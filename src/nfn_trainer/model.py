import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
print('timm:',timm.__version__)

#------------------------------------------------
from decoder import *

def pvtv2_encode(x, e):
    encode = []
    x = e.patch_embed(x)
    for stage in e.stages:
        x = stage(x); encode.append(x)
    return encode



#------------------------------------------------
#dsnt
#https://github.com/kornia/kornia/blob/93114bf3f499eaac7c5f0f25f3e53ec356b191e2/kornia/geometry/subpix/dsnt.py

def heatmap_to_coord(heatmap):
    num_image = len(heatmap)
    device = heatmap[0].device
    _,_, H, W = heatmap[0].shape
    D = max([h.shape[1] for h in heatmap])

    # create coordinates grid.
    x = torch.linspace(0, W - 1, W, device=device)
    y = torch.linspace(0, H - 1, H, device=device)
    z = torch.linspace(0, D - 1, D, device=device)

    point_xy=[]
    point_z =[]
    for i in range(num_image):
        num_point, D, H, W = heatmap[i].shape
        pos_x = x.reshape(1,1,1,W)
        pos_y = y.reshape(1,1,H,1)
        pos_z = z[:D].reshape(1,D,1,1)

        py = torch.sum(pos_y * heatmap[i], dim=(1,2,3))
        px = torch.sum(pos_x * heatmap[i], dim=(1,2,3))
        pz = torch.sum(pos_z * heatmap[i], dim=(1,2,3))

        point_xy.append(torch.stack([px,py]).T)
        point_z.append(pz)

    xy = torch.stack(point_xy)
    z = torch.stack(point_z)
    return xy, z

def heatmap_to_grade(heatmap, grade_mask):
    num_image = len(heatmap)
    grade = []
    for i in range(num_image):
        num_point, D, H, W = heatmap[i].shape
        C, D, H, W = grade_mask[i].shape
        g = grade_mask[i].reshape(1,C,D,H,W)#.detach()
        h = heatmap[i].reshape(num_point,1,D,H,W)#.detach()
        g = (h*g).sum(dim=(2,3,4))
        grade.append(g)
    grade = torch.stack(grade)
    return grade

# ----dynamic matching ---
def do_dynmaic_match_truth(xy, truth_xy, threshold=3):
    num_image, num_point, _2_ = xy.shape
    t = truth_xy[:, :5, 1].reshape(num_image, 5, 1)
    p = xy[:, :5, 1].reshape(num_image, 1, 5)
    diff = torch.abs(p - t)
    left, left_i = diff.min(-1)
    left_t = (left < threshold)

    # closest_i = left_i.tolist()
    # for j in range(num_image):
    #     if closest_i[j] !=[0,1,2,3,4]: print('left',closest_i[j], valid[j])

    t = truth_xy[:, 5:, 1].reshape(num_image, 5, 1)
    p = xy[:, 5:, 1].reshape(num_image, 1, 5)
    diff = torch.abs(p - t)
    right, right_i = diff.min(-1)
    right_t = (right < threshold)

    # closest_i = right_i.tolist()
    # for j in range(num_image):
    #     if closest_i[j] !=[0,1,2,3,4]: print('right',closest_i[j], valid[j])

    index = torch.cat([left_i, right_i + 5], 1).detach()
    valid = torch.cat([left_t, right_t], 1).detach()
    return index, valid


class Net(nn.Module):
    def __init__(self, pretrained=False, cfg=None):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))

        num_grade=3
        num_level=5

        self.arch = 'pvt_v2_b4'
        if cfg is not None:
            self.arch = cfg.arch

        encoder_dim = {
            'resnet18': [64, 64, 128, 256, 512, ],
            'resnet18d': [64, 64, 128, 256, 512, ],
            'resnet34': [ 64, 128, 256, 512, ],
            'resnet50d': [256, 512, 1024, 2048, ],
            'seresnext26d_32x4d': [256, 512, 1024, 2048,],
            'convnext_small.fb_in22k': [96, 192, 384, 768],
            'convnext_tiny.fb_in22k': [96, 192, 384, 768],
            'convnext_base.fb_in22k': [96, 192, 384, 768],
            'tf_efficientnet_b4.ns_jft_in1k':[32, 56, 160, 448],
            'tf_efficientnet_b5.ns_jft_in1k':[40, 64, 176, 512],
            'pvt_v2_b1': [64, 128, 320, 512],
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b3': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
        }.get( self.arch,None)

        decoder_dim =  [384, 192, 96]

        if  self.arch == 'pvt_v2_b4':
            #legacy code
            self.encoder = timm.create_model(
                model_name=self.arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool=''
            )
        else:
            self.encoder = timm.create_model(
                model_name=self.arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool='', features_only=True
            )

        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1],
            out_channel=decoder_dim,
        )

        self.zxy_mask = nn.Conv3d(decoder_dim[-1], 10, kernel_size=1)
        self.grade_mask = nn.Conv3d(decoder_dim[-1], 128, kernel_size=1)
        self.grade = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
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

        #---
        if self.arch == 'pvt_v2_b4':
            #legacy code
            encode = pvtv2_encode(x, self.encoder)
        else:
            encode = self.encoder(x)[-4:]

        ##[print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]
        encode = [ torch.split_with_sizes(e, D, 0) for e in encode ]

        zxy_mask_logit = []
        zxy_mask_prob  = []
        grade_mask_logit = []
        for i in range(num_image):
            e = [ encode[s][i].transpose(1,0).unsqueeze(0) for s in range(4) ]
            l, d = self.decoder(
                feature=e[-1], skip=e[:-1][::-1]
            )

            g = self.grade_mask(l).squeeze(0)
            zxy = self.zxy_mask(l).squeeze(0)
            grade_mask_logit.append(g)
            zxy_mask_logit.append(zxy)

            _,d,h,w = zxy.shape
            zxy_p = zxy.flatten(1).softmax(-1).reshape(-1,d,h,w)
            zxy_mask_prob.append(zxy_p)

        xy, z = heatmap_to_coord(zxy_mask_prob)
        #---
        num_point = xy.shape[1]
        grade = heatmap_to_grade(zxy_mask_prob, grade_mask_logit)
        grade = grade.reshape(num_image*num_point,-1)
        grade = self.grade(grade)
        grade = grade.reshape(num_image,num_point,3)

        #---
        zxy_mask = torch.cat(zxy_mask_prob, 1).transpose(1, 0)


        output = {}
        if 'loss' in self.output_type:
            output['zxy_loss'] = F_zxy_loss(z, xy, batch['z'].to(device), batch['xy'].to(device))
            output['zxy_mask_loss'] = F_divergence_loss(zxy_mask, batch['zxy_mask'].to(device), D)

            #output['grade_loss'] = F_grade_loss(grade,  batch['grade'].to(device))
            if 1:
                index, valid = do_dynmaic_match_truth(xy, batch['xy'].to(device))
                truth = batch['grade'].to(device)
                truth_matched = []
                for i in range(num_image):
                    truth_matched.append(truth[i][index[i]])
                truth_matched = torch.stack(truth_matched)
                output['grade_loss'] = F_grade_loss(grade[valid],  truth_matched[valid])

        if 'infer' in self.output_type:
            output['grade'] = F.softmax(grade,-1)
            output['zxy_mask'] = zxy_mask
            output['xy'] = xy
            output['z'] = z

        return output

#--------------------------------------------------------------------------
def F_zxy_loss(z, xy,  z_truth, xy_truth):
    m = z_truth!=-1
    z_truth = z_truth.float()
    loss = (
        F.mse_loss(z[m], z_truth[m]) + F.mse_loss(xy[m], xy_truth[m])
    )
    return loss

def F_grade_loss(grade, truth):
    eps = 1e-5
    weight = torch.FloatTensor([1,2,4]).to(grade.device)
    t = truth.reshape(-1)
    g = grade.reshape(-1,3)
    #loss = F.nll_loss( torch.clamp(g, eps, 1-eps).log(), t,weight=weight, ignore_index=-1)
    loss = F.cross_entropy(g, t,weight=weight, ignore_index=-1)
    return loss

#https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/11
#Jensen-Shannon divergence
def F_divergence_loss(heatmap, truth, D):
    heatmap =  torch.split_with_sizes(heatmap, D, 0)
    truth =  torch.split_with_sizes(truth, D, 0)
    num_image = len(heatmap)

    loss =0
    for i in range(num_image):
        p,q = truth[i], heatmap[i]
        D,num_point,H,W = p.shape

        eps = 1e-8
        p = torch.clamp(p.transpose(1,0).flatten(1),eps,1-eps)
        q = torch.clamp(q.transpose(1,0).flatten(1),eps,1-eps)
        m = (0.5 * (p + q)).log()

        kl = lambda x,t: F.kl_div(x,t, reduction='batchmean', log_target=True)
        loss += 0.5 * (kl(m, p.log()) + kl(m, q.log()))
    loss = loss/num_image
    return loss


###################################################################################################################
def run_check_net():
    D = [6, 7, 9, 11, 3, 4, 5]
    image_size = 320
    mask_size  = image_size//4
    num_image  = len(D)
    B = sum(D)
    num_point = 10

    batch = {
        'D': D,
        'image': torch.from_numpy( np.random.uniform(-1, 1, ( B, image_size, image_size))).byte(),
        'z': torch.from_numpy(np.random.choice(min(D), (num_image, num_point))).long(),
        'xy': torch.from_numpy(np.random.choice(image_size, (num_image, num_point, 2))).float(),
        'grade': torch.from_numpy(np.random.choice(3, (num_image, num_point))).long(),
        'zxy_mask': torch.from_numpy(np.random.uniform(0,1,(B, num_point, mask_size, mask_size))).float(),
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
