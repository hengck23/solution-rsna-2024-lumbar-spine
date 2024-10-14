import pandas as pd
import numpy as np
import pydicom
import glob
import cv2

import sys
import os
from ast import literal_eval
from natsort import natsorted


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

##############################################################################333
# read data
def np_dot(a, b):
    return np.sum(a * b, 1)


def normalise_to_8bit(x, lower=0.1, upper=99.9):
    lower, upper = np.percentile(x, (lower, upper))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype(np.uint8)


# https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/537339
def heng_read_series(study_id, series_id, series_description, image_dir):
    dicom_dir = f'{image_dir}/{study_id}/{series_id}'

    # read dicom file
    dicom_file = natsorted(glob.glob(f'{dicom_dir}/*.dcm'))
    if len(dicom_file) == 0:
        return None, None, ['empty-dir']

    instance_number = [int(f.split('/')[-1].split('.')[0]) for f in dicom_file]
    dicom = [pydicom.dcmread(f) for f in dicom_file]

    # make dicom header df
    dicom_df = []
    for i, d in zip(instance_number, dicom):  # d__.dict__
        dicom_df.append(
            dotdict(
                study_id=study_id,
                series_id=series_id,
                series_description=series_description,
                instance_number=i,

                H=d.pixel_array.shape[0],
                W=d.pixel_array.shape[1],

                ImagePositionPatient=[float(v) for v in d.ImagePositionPatient],
                ImageOrientationPatient=[float(v) for v in d.ImageOrientationPatient],
                PixelSpacing=[float(v) for v in d.PixelSpacing],
                grouping=str([round(float(v), 3) for v in d.ImageOrientationPatient]),

                # error for hidden test
                ##  SpacingBetweenSlices=float(d.SpacingBetweenSlices),
                ##  SliceThickness=float(d.SliceThickness),
            )
        )
    dicom_df = pd.DataFrame(dicom_df)
    # dicom_df.to_csv('dicom_df.csv',index=False)

    # ----
    Wmax = dicom_df.W.max()
    Hmax = dicom_df.H.max()

    error_code = []
    if ((dicom_df.W.nunique() != 1) or (dicom_df.H.nunique() != 1)):
        error_code.append('multi-shape')

        # sort slices
    dicom_df = [d for _, d in dicom_df.groupby('grouping')]

    data = []
    sort_data_by_group = []
    for df in dicom_df:
        position = np.array(df['ImagePositionPatient'].values.tolist())
        orientation = np.array(df['ImageOrientationPatient'].values.tolist())
        normal = np.cross(orientation[:, :3], orientation[:, 3:])
        projection = np_dot(normal, position)
        df.loc[:, 'projection'] = projection
        df = df.sort_values('projection')

        volume = []
        for i in df.instance_number:
            v = dicom[instance_number.index(i)].pixel_array
            if 'multi-shape' in error_code:
                H, W = v.shape
                v = np.pad(v, [(0, Hmax - H), (0, Wmax - W)], 'reflect')
            volume.append(v)

        volume = np.stack(volume)
        volume = normalise_to_8bit(volume)

        data.append(dotdict(
            df=df,
            volume=volume,
        ))

        if 'sagittal' in series_description.lower():
            sort_data_by_group.append(position[0, 0])  # x
        if 'axial' in series_description.lower():
            sort_data_by_group.append(position[0, 2])  # z

    data = [r for _, r in sorted(zip(sort_data_by_group, data))]
    for i, r in enumerate(data):
        r.df.loc[:, 'group'] = i

    df = pd.concat([r.df for r in data])
    df.loc[:, 'z'] = np.arange(len(df))
    volume = np.concatenate([r.volume for r in data])
    return volume, df, error_code


def do_resize_and_center(
        image, reference_size
):
    H, W = image.shape[:2]
    if (W == reference_size) & (H == reference_size):
        return image, [1, 0, 0]

    s = reference_size / max(H, W)
    m = cv2.resize(image, dsize=None, fx=s, fy=s)
    h, w = m.shape[:2]
    padx0 = (reference_size - w) // 2
    padx1 = reference_size - w - padx0
    pady0 = (reference_size - h) // 2
    pady1 = reference_size - h - pady0

    m = np.pad(m, [[pady0, pady1], [padx0, padx1], [0, 0]], mode='constant', constant_values=0)
    # p = point * s +[[padx0,pady0]]
    resize_param = [s, padx0, pady0]
    return m, resize_param




