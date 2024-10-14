import sys
sys.path.append('..')
sys.path.append('../data_process')
from common import *
from _dir_setting_ import *
from data import *
from configure import *

import torch
from functools import *

import albumentations as A
from albumentations.pytorch import ToTensorV2
from augmentation import *

import numpy as np
import pandas as pd


###################################################################################################
def load_csv():
    csv_file = f'{DATA_PROCESSED_DIR}/scs_sag_t2_processed_df.csv'
    processed_df = pd.read_csv(csv_file)
    processed_df = do_clean_by_eval_df(processed_df, col=['grade','instance_number','xyz', 'level'])
    return processed_df

def make_level_mask(xyz, level, sigma=0, mask_shape=(5,80,80)):

    D,H,W = mask_shape
    num_xyz = len(xyz)
    if sigma==0:
        mask = np.full( (5,H,W), fill_value=0, dtype=np.float32)
        for i in range(num_xyz):
            x,y, z = xyz[i]
            x = int(round(x))
            y = int(round(y))
            z = int(round(z))
            if z!=-1:
                mask[level[i], y, x] = 1
    else:
        mask = np.full( (5,H,W), fill_value=0, dtype=np.float32)
        for i in range(num_xyz):
            x,y, z = xyz[i]
            pos_x = np.linspace(0, W - 1, W).reshape(1, 1, W)
            pos_y = np.linspace(0, H - 1, H).reshape(1, H, 1)

            # Gaussian PDF = exp(-(x - \mu)^2 / (2 \sigma^2))
            #              = exp(dists * ks),
            #                where dists = (x - \mu)^2 and ks = -1 / (2 \sigma^2)

            dist_x = (pos_x - x) ** 2
            dist_y = (pos_y - y) ** 2

            exp_x = np.exp( -0.5 * dist_x / (sigma ** 2))
            exp_y = np.exp( -0.5 * dist_y / (sigma ** 2))
            gauss = exp_x*exp_y
            gauss = gauss/(gauss.sum(keepdims=True)+ 1e-16)
            mask[level[i]] = gauss
    return mask


######################################################################3

def make_valid_augment(cfg):
    transform = A.Compose([
            A.LongestMaxSize(max_size=cfg.image_size, interpolation=1),
            A.PadIfNeeded(min_height=cfg.image_size, min_width=cfg.image_size, border_mode=0, value=(0, 0, 0)),
        ],
        keypoint_params=A.KeypointParams(format='xy',),
        p=1.)
    return transform

#https://github.com/albumentations-team/albumentations/issues/718
def make_train_augment(cfg):

    transform = A.Compose([
            A.LongestMaxSize(max_size=cfg.image_size, interpolation=1),
            A.PadIfNeeded(min_height=cfg.image_size, min_width=cfg.image_size, border_mode=0, value=(0, 0, 0)),

            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                p=0.75
            ),

            A.OneOf([
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(var_limit=(10.0, 50.0)),
            ], p=0.5),
        ],
        keypoint_params=A.KeypointParams(format='xy',remove_invisible=False),
        p=1.)
    return transform

#---------------------------------
def make_random_split( fold=1, num_fold=5):

    all_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train.csv')
    all_id = sorted(all_df.study_id.values.tolist())
    all_id = np.array(all_id)

    rng   = np.random.RandomState(42)
    choice = rng.choice(num_fold, len(all_id))
    train_id = all_id[np.where(choice!=fold)[0]]
    valid_id = all_id[np.where(choice==fold)[0]]
    train_id = np.sort(train_id)
    valid_id = np.sort(valid_id)
    return train_id, valid_id



class SplineDataset(Dataset):
    def __init__(self, processed_df, sample_id, cfg, augment=make_valid_augment, mode='train'):
        error_id = [
            # (study_id,series_id)
            (3637444890, 3892989905),
        ]
        exlude = [ study_id for study_id, series_id in error_id]
        sample_id =[ s for s in sample_id if s not in exlude]

        self.mode = mode
        self.cfg = cfg
        self.df = processed_df[processed_df.study_id.isin(sample_id)].reset_index(drop=True)
        self.sample_id = sample_id
        self.augment = augment(cfg)
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __str__(self):
        text = ''
        text += f'\tlen = {len(self)}\n'
        text += f'\t\tnum_study_id : {len(self.df.study_id.unique())}\n'
        text += f'\t\tnum_series_id : {len(self.df.series_id.unique())}\n'
        text += f'\t\tnum_points: {len(self.df.series_id.unique())*10}\n'

        if 1:
            gradename = [
                'normal/mild', 'moderate', 'severe'
            ]
            weight = [1, 2, 4]

            grade = np.array(self.df.grade.tolist())
            count = [(grade == i).sum() for i in [0, 1, 2]]
            L = len(grade)
            wL = sum(weight[i] * count[i] for i in [0, 1, 2])
            for i in [0, 1, 2]:
                text += f'\t\t{i} {gradename[i]:>16}: {count[i]:5d} {count[i] / L:0.3f}  ({weight[i] * count[i] / wL:0.3f})\n'
        return text

    def __getitem__(self, index):
        d = self.df.iloc[index]
        volume = np.load(f'{DATA_PROCESSED_DIR}/mini-clean5.0/{d.study_id}/{d.series_id}/volume.npz')['volume']
        vxyz   = np.array(d.xyz, dtype=np.float32)
        grade  = np.array(d.grade, dtype=np.int32)
        level  = np.array([ int(l[1])-1 for l in d.level], dtype=np.int32)
        assert (np.sum(vxyz==-1)==0)

        image = np.ascontiguousarray(volume.transpose(1,2,0))
        point = vxyz[:,[0,1]]
        image, point = do_resize_and_center(image, point, reference_size=512)

        # start of uagmentation -------------------------------------------------------
        if self.mode == 'train':
            if np.random.rand() < 0.5: #flip
                vxyz[...,2] =  image.shape[2] - 1 -vxyz[...,2]
                image = np.ascontiguousarray(image[..., ::-1])


            u = np.random.choice([0,1,2,],p=[0.1,0.7,0.2])
            if u==1:
                mean_512_shape = np.load(f'{DATA_PROCESSED_DIR}/scs_sag_t2_mean.512.npy')
                mean_512_shape = mean_512_shape.reshape(-1, 2) + [[256, 256]]
                mean_512_shape = mean_512_shape[1::2]
                mean_512_shape = np.ascontiguousarray(mean_512_shape).astype(np.float32)

                mat = get_rotate_scale_by_reference_mat(
                    point, (512,512), mean_512_shape,
                    scale_limit=(-0.20, 0.20),
                    rotate_limit=(-15, 15),
                    shift_limit=(5, 5),
                    border=10
                )
                image, point = apply_affine( image, point, mat )

            if u==2:
                mat = get_safe_custom_mat(
                    point, (512,512),
                    affline_limit = (-0.2,0.2),
                    border=5
                )
                image, point = apply_perspective( image, point, mat )

            #-------
            # if np.random.random() < 0.5:
            #     image = do_random_cutout(image, point)

        if self.mode=='train':
            r = self.augment(image=image.copy(), keypoints=point)

        if self.mode=='valid':
            r = self.augment(image=image.copy(), keypoints=point)

        # ----
        point = r['keypoints']
        image = r['image']
        image = np.ascontiguousarray(image.transpose(2,0,1))

        # end of uagmentation -------------------------------------------------------


        #make attention mask for supervision
        D,h,w = image.shape
        xy = np.array(point)*[[self.cfg.mask_size/w, self.cfg.mask_size/h ]]
        z = vxyz[:,[2]]
        xyz = np.concatenate([xy,z],-1)

        mask_shape = [len(image),self.cfg.mask_size, self.cfg.mask_size]
        level_mask = make_level_mask(xyz, level, sigma=self.cfg.level_sigma, mask_shape=mask_shape)


        r = {}
        r['index'] = index
        r['d'] = d
        r['D'] = D
        r['image'  ] = torch.from_numpy(image)
        r['level_mask'] = torch.from_numpy(level_mask)
        r['xyz' ] = torch.from_numpy(xyz)
        r['grade'] = torch.from_numpy(grade)
        return r


tensor_key = ['image','level_mask', 'grade', 'xyz']
def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        d[k] = [b[k] for b in batch]

    d['image'] = torch.cat(d['image']).byte()
    d['level_mask'] = torch.stack(d['level_mask']).float()
    d['xyz'] = torch.stack(d['xyz']).float()
    d['grade'] = torch.stack(d['grade']).long()
    return d

##########################################################################################3

def run_check_dataset():
    from configure import default_cfg as cfg

    processed_df = load_csv()
    train_id, valid_id = make_random_split()
    #dataset = SplineDataset(processed_df, train_id, cfg=default_cfg, augment=make_valid_augment, mode='valid')
    dataset = SplineDataset(processed_df, train_id, cfg=cfg, augment=make_train_augment, mode='train')
    print(dataset)

    for i in range(5):
        r = dataset[i]

        print(r['index'], '--------------------')
        for k in tensor_key:
            v = r[k]
            print(k)
            print('\t', 'dtype:', v.dtype)
            print('\t', 'shape:', v.shape)
            if len(v) != 0:
                print('\t', 'min/max:', v.min().item(), '/', v.max().item())
                print('\t', 'sum:', v.sum().item())
                print('\t', 'is_contiguous:', v.is_contiguous())
                print('\t', 'values:')
                print('\t\t', v.reshape(-1)[:3].data.numpy().tolist(), '...', v.reshape(-1)[-3:].data.numpy().tolist())
        print('')

        if 1:
            # draw
            #print(r['d'])
            image = r['image'].data.cpu().numpy()
            mask  = r['level_mask'].data.cpu().numpy()

            image = image.mean(0)
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            resize = cfg.image_size// cfg.mask_size
            mask = mask.sum((0)).astype(np.float32)
            mask = cv2.resize(mask, (cfg.image_size,cfg.image_size))
            mask = mask/mask.max()
            image[...,2] =255-(1-mask)*(255-image[...,2])

            image_show_norm('image', image)#, type='rgb')
            image_show_norm('mask',mask,resize=1)
            cv2.waitKey(0)


    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=7,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=null_collate,
    )
    print(f'batch_size   : {loader.batch_size}')
    print(f'len(loader)  : {len(loader)}')
    print(f'len(dataset) : {len(dataset)}')
    print('')

    for t, batch in enumerate(loader):
        if t > 5: break
        print('batch ', t, '===================')
        print('index', batch['index'])
        print('D', batch['D'])

        for k in tensor_key:
            v = batch[k]
            print(k)
            print('\t', v.data.shape)
            print('\t', 'is_contiguous:', v.is_contiguous())

        print('')
# main #################################################################
if __name__ == '__main__':
    run_check_dataset()
