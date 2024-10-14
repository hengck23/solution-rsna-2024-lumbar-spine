import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import timm
#from timm.models.convnext import *

from decoder import *

#------------------------------------------------
# processing

def heatmap_to_coord(heatmap):
    device = heatmap.device
    B, num_point, H, W = heatmap.shape

    # create coordinates grid.
    x = torch.linspace(0, W - 1, W, device=device)
    y = torch.linspace(0, H - 1, H, device=device)
    # if normalized_coordinates:
    #     xs = (xs / (width - 1) - 0.5) * 2
    #     ys = (ys / (height - 1) - 0.5) * 2
    grid = torch.meshgrid([x, y], indexing='xy')
    pos_x = grid[0].reshape(1,1,-1)
    pos_y = grid[1].reshape(1,1,-1)

    h = heatmap.reshape(B, num_point, -1)
    y = torch.sum(pos_y * h, -1, keepdim=True)
    x = torch.sum(pos_x * h, -1, keepdim=True)
    xy = torch.cat([x, y], -1)
    xy = xy.reshape(B,num_point,2)
    return xy


def heatmap_to_grade(grade_mask, level_mask):
    num_image, num_level, h, w = level_mask.shape
    num_image, num_grade, h, w = grade_mask.shape

    e = level_mask.reshape(num_image, num_level, 1, h, w)
    g = grade_mask.reshape(num_image, 1, num_grade, h, w)
    grade = (e * g).sum(dim=(-2, -1))
    return grade

# dynamic matching
def do_dynamic_match_truth(xy, truth_xyz, threshold=3):
    num_image, num_point, _2_ = xy.shape
    t = truth_xyz[:, :, 1].reshape(num_image, 1, 5)
    p = xy[:, :, 1].reshape(num_image, 5, 1)
    diff = torch.abs(p - t)
    value, index = diff.min(-1)
    valid = (value < threshold)
    return index.detach(), valid.detach()


#------------------------------------------------

class Net(nn.Module):
    def __init__(self, pretrained=False, cfg=None):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss', ]
        self.register_buffer('D', torch.tensor(0))

        num_grade=3
        num_level=5

        self.arch = 'pvt_v2_b4'
        if cfg is not None:
            self.arch = cfg.arch

        encoder_dim = {
            'resnet18': [64, 64, 128, 256, 512, ],
            'resnet18d': [64, 64, 128, 256, 512, ],
            'resnet34': [64, 64, 128, 256, 512, ],
            'resnet50d': [64, 256, 512, 1024, 2048, ],
            'seresnext26d_32x4d': [64, 256, 512, 1024, 2048, ],
            'convnext_small.fb_in22k': [96, 192, 384, 768],
            'convnext_tiny.fb_in22k': [96, 192, 384, 768],
            'convnext_base.fb_in22k': [128, 256, 512, 1024],
            'tf_efficientnet_b4.ns_jft_in1k':[32, 56, 160, 448],
            'tf_efficientnet_b5.ns_jft_in1k':[40, 64, 176, 512],
            'pvt_v2_b1': [64, 128, 320, 512],
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
        }.get(self.arch, [768])
        decoder_dim = \
              [384, 192, 96]

        self.encoder = timm.create_model(
            model_name=self.arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool='', features_only=True,
        )
        self.decoder = MyUnetDecoder(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1],
            out_channel=decoder_dim,
        )
        self.z_mask = nn.Sequential(
            nn.Linear(decoder_dim[-1],64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_level),
        )
        self.level_mask = nn.Sequential(
            nn.Conv2d(decoder_dim[-1], 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, num_level, kernel_size=1),
        )
        self.grade_mask = nn.Sequential(
            nn.Conv2d(decoder_dim[-1], 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, num_grade, kernel_size=1),
        )

    def forward(self, batch):
        device = self.D.device

        image = batch['image'].to(device)
        D = batch['D']
        num_image = len(D)
        B, H, W = image.shape
        image = image.reshape(B, 1, H, W)

        x = (image.float() / 255 - 0.5) / 0.5
        x = x.expand(-1, 3, -1, -1)

        encode = self.encoder(x)[-4:]
        # [print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1]
        )
        z_pool = last.mean(dim=(2, 3))
        z_mask = self.z_mask(z_pool)
        z_mask = torch.cat([torch.softmax(z, 0) for z in torch.split_with_sizes(z_mask, D, 0)])
        # print('z_logit', z_logit.shape)

        # ---
        xy_pool = []
        for f, z in zip(
                torch.split_with_sizes(last, D, 0),
                torch.split_with_sizes(z_mask, D, 0)):
            d = len(f)
            z = z.mean(1).detach()
            z = (z*torch.arange(d,device=device)).sum().item()
            z = int(round(z))
            zmin = max(0,z-3)  #crop 7 neighbour slices
            zmax = min(d,z+3+1)
            pool = f[zmin:zmax].mean(0)
            xy_pool.append(pool)
        xy_pool = torch.stack(xy_pool)


        # pool level mask to xy ---
        level_mask = self.level_mask(xy_pool)
        num_image, _5_, h, w = level_mask.shape
        level_mask = level_mask.reshape(num_image, 5, -1)
        level_mask = F.softmax(level_mask, dim=-1)
        level_mask = level_mask.reshape(num_image, 5, h, w )
        xy = heatmap_to_coord(level_mask)

        # pool grade mask to grade ---
        grade_mask = self.grade_mask(xy_pool)
        grade_mask = F.softmax(grade_mask, 1)
        grade = heatmap_to_grade(grade_mask, level_mask)

        output = {}
        if 'loss' in self.output_type:
            output['z_mask_loss'] = F_z_mask_loss(z_mask, batch['xyz'].to(device),D)
            output['xy_loss'] = F.mse_loss(xy, batch['xyz'].to(device)[...,:2])
            output['level_mask_loss'] = F_level_mask_loss(level_mask, batch['level_mask'].to(device))

            output['grade_loss'] = F_grade_loss(grade,  batch['grade'].to(device))
            if 1:
                #pool grade mask to grade (truth)
                xyz_truth = batch['xyz'].to(device).reshape(-1,3)
                ii = torch.arange(num_image).to(device).reshape(num_image,1).repeat(1,5).reshape(-1)
                h,w = grade_mask.shape[-2:]
                for dx,dy in [
                          (-1,0),
                    (0,-1),(0,0),(0,1),
                           (1,0),
                ]:
                    xx = torch.round(xyz_truth[:,0]+dx*0.5).long()
                    yy = torch.round(xyz_truth[:,1]+dy*0.5).long()
                    xx = torch.clamp(xx,0,w-1)
                    yy = torch.clamp(yy,0,h-1)
                    grade_from_truth = grade_mask[ii,:,yy,xx]
                    weight = 0.500 if (dx==0)&(dy==0) else 0.125
                    output['grade_loss'] += weight* F_grade_loss(grade_from_truth, batch['grade'].to(device))


        if 'infer' in self.output_type:
            output['level_mask'] = level_mask
            output['grade'] = grade
            output['z_mask'] = z_mask
            output['xy'] = xy
            output['z'] = torch.stack([z.argmax(0) for z in torch.split_with_sizes(z_mask, D, 0)])

        return output


#--------------------------------------------------------------------------
 #https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/11

def F_grade_loss(grade, truth):
    eps = 1e-5
    weight = torch.FloatTensor([1,2,4]).to(grade.device)
    t = truth.reshape(-1)
    g = grade.reshape(-1,3)
    loss = F.nll_loss(torch.clamp(g, eps, 1-eps).log(), t,weight=weight, ignore_index=-1)
    return loss

def F_level_mask_loss(level_mask, truth):
    p,q = truth, level_mask
    B,_5_,h,w = p.shape

    eps = 1e-8
    p = torch.clamp(p.transpose(1,0).reshape(-1,h*w),eps,1-eps)
    q = torch.clamp(q.transpose(1,0).reshape(-1,h*w),eps,1-eps)
    m = (0.5 * (p + q)).log()

    kl = lambda x,t: F.kl_div(x,t, reduction='batchmean', log_target=True)
    loss = 0.5 * (kl(m, p.log()) + kl(m, q.log()))
    return loss

def F_z_mask_loss(z_mask, truth, D):
    eps = 1e-8
    z_mask =  torch.split_with_sizes(z_mask, D, 0)
    num_image = len(D)

    loss = 0
    for i in range(num_image):
        g = z_mask[i].transpose(1,0)
        t =  truth[i,:,2].long()
        #loss += F.cross_entropy(g, t, ignore_index=-1,)# label_smoothing=0.1)
        loss += F.nll_loss(torch.clamp(g, eps, 1 - eps).log(), t, ignore_index=-1)
    loss = loss/num_image
    return loss


#------------------------------------------------------------------------
def run_check_net():

    D = [7, 6, 9, 11, 10, 14, 15]
    num_image  = len(D)
    B = sum(D)

    image_size = 320
    mask_size  = 320//4
    num_grade = 3
    num_level = 5

    batch = {
        'D': D,
        'image': torch.from_numpy(np.random.choice(256, (B, image_size, image_size))).byte(),
        'level_mask': torch.from_numpy(np.random.choice(1, (num_image,  num_level, mask_size, mask_size))).float(),
        'grade_mask': torch.from_numpy(np.random.choice(1, (B, mask_size, mask_size))).long(),
        'grade': torch.from_numpy(np.random.choice(3, (num_image, num_level))).long(),
        'xyz': torch.from_numpy(np.random.choice(min(D), (num_image, num_level, 3))).float(),
    }

    net = Net(pretrained=True, cfg=None).cuda()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            output = net(batch)
    # ---
    print('batch')
    for k, v in batch.items():
        if k == 'D':
            print(f'{k:>32} : {v} ')
        else:
            print(f'{k:>32} : {v.shape} ')

    print('output')
    for k, v in output.items():
        if 'loss' not in k:
            print(f'{k:>32} : {v.shape} ')
    print('loss')
    for k, v in output.items():
        if 'loss' in k:
            print(f'{k:>32} : {v.item()} ')

# main #################################################################
if __name__ == '__main__':
    run_check_net()
