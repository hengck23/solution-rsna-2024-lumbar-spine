import pandas as pd
import numpy as np
import pydicom
import glob
import cv2

import sys
import os
from ast import literal_eval
from natsort import natsorted
import sklearn

class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

## data evaluation #########################################################

def do_local_lb(probability,truth, is_any=False):
	# probability : N,2,5,3    N x condition x level x grade
	# truth: N,2,5

	p = probability.reshape(-1,3)
	t = truth.reshape(-1)

	available = t!=-1
	p = p[available]
	t = t[available]

	loss  = []
	count = []
	for i in [0, 1, 2]: #3 grade
		l = -np.log(p[t == i][:, i])
		L = len(l)
		if L == 0:
			count.append(0)
			loss.append(0)
		else:
			count.append(L)
			loss.append(l.mean())

	weight=[1,2,4]
	weighted_loss = (
		 (weight[0]*count[0] * loss[0]  + weight[1]*count[1] * loss[1]  + weight[2]*count[2] * loss[2] ) /
		 (weight[0]*count[0] + weight[1]*count[1] + weight[2]*count[2] )
	)
	if is_any==False:
		return loss, weighted_loss

	#---
	if 1:
		any_truth = truth.reshape(-1, 5)
		any_prob  = probability.reshape(-1, 5, 3)

		t  = (any_truth.reshape(-1, 5) == 2).max(-1).astype(int)
		p  = (any_prob.reshape(-1, 5, 3)[..., 2]).max(-1)
		weight = (t == 1) * 4 + (t != 1) * 1
		any_loss = sklearn.metrics.log_loss(
			y_true=t,
			y_pred=p,
			sample_weight=weight,
		)
	return loss, weighted_loss, any_loss


def do_compute_point_error(xy, z, xyz_truth, threshold = [1,2,3,5]):
	xyz_truth = xyz_truth.reshape(-1,3)
	xy = xy.reshape(-1,2)
	z = z.reshape(-1)

	x_t,y_t,z_t = xyz_truth.T
	x,y = xy.T

	x_diff = np.abs(x-x_t)
	y_diff = np.abs(y-y_t)
	z_diff = np.abs(z-z_t)
	x_err = [(x_diff<=th).mean() for th in threshold]
	y_err = [(y_diff<=th).mean() for th in threshold]
	z_err = [(z_diff<=th).mean() for th in threshold]

	return x_err,y_err,z_err,threshold


## read/write data #########################################################

grade_map = {
	'Missing': -1,
	'Normal/Mild': 0,
	'Moderate': 1,
	'Severe': 2,
}
condition_map = {  # follow sample submission order
	'Left Neural Foraminal Narrowing': 'left_neural_foraminal_narrowing',
	'Left Subarticular Stenosis': 'left_subarticular_stenosis',
	'Right Neural Foraminal Narrowing': 'right_neural_foraminal_narrowing',
	'Right Subarticular Stenosis': 'right_subarticular_stenosis',
	'Spinal Canal Stenosis': 'spinal_canal_stenosis',
}
level_map = {
	'L1/L2': 'l1_l2',
	'L2/L3': 'l2_l3',
	'L3/L4': 'l3_l4',
	'L4/L5': 'l4_l5',
	'L5/S1': 'l5_s1',
}
description_map = {
	'Sagittal T2/STIR': 'sagittal_t2',
	'Sagittal T1': 'sagittal_t1',
	'Axial T2': 'axial_t2',
}

level_col = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

condition_level_col = [  # follow sample submission order
	'left_neural_foraminal_narrowing_l1_l2',
	'left_neural_foraminal_narrowing_l2_l3',
	'left_neural_foraminal_narrowing_l3_l4',
	'left_neural_foraminal_narrowing_l4_l5',
	'left_neural_foraminal_narrowing_l5_s1',
	'left_subarticular_stenosis_l1_l2',
	'left_subarticular_stenosis_l2_l3',
	'left_subarticular_stenosis_l3_l4',
	'left_subarticular_stenosis_l4_l5',
	'left_subarticular_stenosis_l5_s1',
	'right_neural_foraminal_narrowing_l1_l2',
	'right_neural_foraminal_narrowing_l2_l3',
	'right_neural_foraminal_narrowing_l3_l4',
	'right_neural_foraminal_narrowing_l4_l5',
	'right_neural_foraminal_narrowing_l5_s1',
	'right_subarticular_stenosis_l1_l2',
	'right_subarticular_stenosis_l2_l3',
	'right_subarticular_stenosis_l3_l4',
	'right_subarticular_stenosis_l4_l5',
	'right_subarticular_stenosis_l5_s1',
	'spinal_canal_stenosis_l1_l2',
	'spinal_canal_stenosis_l2_l3',
	'spinal_canal_stenosis_l3_l4',
	'spinal_canal_stenosis_l4_l5',
	'spinal_canal_stenosis_l5_s1',
]

def load_kaggle_csv(DATA_KAGGLE_DIR):
	id_df    = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_series_descriptions.csv')
	grade_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train.csv')
	coord_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_label_coordinates.csv')

	id_df.loc[:, 'series_description'] = id_df['series_description'].map(description_map)
	grade_df = grade_df.fillna(value='Missing')
	grade_df = grade_df.set_index('study_id')
	grade_df = grade_df[condition_level_col]
	grade_df = grade_df.map(lambda x: grade_map[x])
	grade_df = grade_df.reset_index(drop=False)
	coord_df.loc[:, 'condition'] = coord_df['condition'].map(condition_map)
	coord_df.loc[:, 'level'] = coord_df['level'].map(level_map)
	return id_df,grade_df,coord_df

#EVAL_COL=['ImagePositionPatient','ImageOrientationPatient','PixelSpacing']
def do_clean_by_eval_df(df, col):
	for c in col:
		try:
			df.loc[:,c] = df[c].apply(lambda x: literal_eval(x))
		except:
			continue
	return df

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




