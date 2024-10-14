import sys
sys.path.append('..')
from common import *
from _dir_setting_ import *
from data import *


# convert dicom data to np array and dicom header to csv file
def run_make_series_data():
	id_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_series_descriptions.csv')
	image_dir = f'{DATA_KAGGLE_DIR}/train_images'

	for i,d in id_df.iterrows():
		print(i,d.study_id, d.series_id )
		#if i==30: exit(0)
		volume, df, error_code = heng_read_series(d.study_id, d.series_id, d.series_description, image_dir)

		data_dir = f'{DATA_PROCESSED_DIR}/mini-clean5.0/{d.study_id}/{d.series_id}'
		os.makedirs(data_dir, exist_ok=True)
		df.to_csv(f'{data_dir}/df.csv', index=False)
		np.savez_compressed(f'{data_dir}/volume.npz', volume=volume)

		# os.makedirs(f'{data_dir}/png', exist_ok=True)
		# for z, v in enumerate(volume):
		#     cv2.imwrite(f'{data_dir}/png/{z:02}.png', v)

# nfn train data : processed_df
def run_make_nfn_data():
	error_series_id = [
		#error in left point
		8785691, 3355993164, 4066376844,
		#left/right point overlap
		3230157587, 692927423, 2289719834, 4030602643, 856763877,
	]

	target_condition   = ['left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing']
	target_description = 'sagittal_t1'
	id_df, grade_df,coord_df = load_kaggle_csv(DATA_KAGGLE_DIR)
	id_df = id_df[id_df.series_description==target_description]

	processed_df = []
	for i, d in id_df.iterrows():
		if d['series_id'] in error_series_id:  continue
		print(i, d.series_id )

		dicom_df = pd.read_csv(f'{DATA_PROCESSED_DIR}/mini-clean5.0/{d.study_id}/{d.series_id}/df.csv')
		instance_number_to_z_map = {
			n: z for (n, z) in dicom_df[['instance_number', 'z']].values
		}

		this_left_coord_df = coord_df[
			  (coord_df.study_id == d.study_id)
			& (coord_df.series_id == d.series_id)
			& (coord_df.condition == 'left_neural_foraminal_narrowing')
		]
		this_right_coord_df = coord_df[
			  (coord_df.study_id == d.study_id)
			& (coord_df.series_id == d.series_id)
			& (coord_df.condition == 'right_neural_foraminal_narrowing')
		]
		this_grade_df = grade_df[
			(grade_df.study_id == d.study_id)
		]
		if not ((len(this_right_coord_df) == 5) & (len(this_left_coord_df) == 5)): continue

		zz = 0
		this_left_coord_df  = this_left_coord_df.sort_values('level')
		this_right_coord_df = this_right_coord_df.sort_values('level')
		this_coord_df = pd.concat([this_left_coord_df, this_right_coord_df])

		instance_number = this_coord_df['instance_number'].tolist()
		xy = this_coord_df[['x', 'y']].values.tolist()
		z = [instance_number_to_z_map[n] for n in instance_number]

		# check same xyz in left right
		left_xyz  = np.array([[xy[i][0], xy[i][1], z[i]] for i in range(0, 5)])
		right_xyz = np.array([[xy[i][0], xy[i][1], z[i]] for i in range(5, 10)])
		diff = np.fabs(left_xyz.reshape(1, 5, 3) - right_xyz.reshape(5, 1, 3)).sum(-1)
		if (diff < 2).any():
			print('error : same left/right ???', i, d)
			# 3230157587,692927423,2289719834,4030602643,856763877
			raise NotImplementedError

		grade = this_grade_df[
			  ['left_neural_foraminal_narrowing_' + l for l in level_col]
			+ ['right_neural_foraminal_narrowing_' + l for l in level_col]
		].values[0].tolist()

		one_row = dotdict(
			study_id=d.study_id,
			series_id=d.series_id,
			series_description=target_description,
			grade=grade,
			instance_number=instance_number,
			z=z,
			xy=xy,
		)
		processed_df.append(one_row)

	processed_df = pd.DataFrame(processed_df)
	print(processed_df)
	processed_df = processed_df.reset_index(drop=True)

	csv_file = f'{DATA_PROCESSED_DIR}/nfn_sag_t1_processed_df.csv'
	processed_df.to_csv(csv_file, index=False)
	print('saved:', csv_file)
	#[1959 rows x 7 columns]

def run_make_scs_data():
	error_id = [  # (study_id,series_id)
		(3637444890, 3892989905),
	]

	target_condition = ['spinal_canal_stenosis']
	target_description = 'sagittal_t2'
	id_df, grade_df, coord_df = load_kaggle_csv(DATA_KAGGLE_DIR)

	#hand corrected level points for SCS from team mate @lhwcv
	coord_df = pd.read_csv(f'{DATA_PROCESSED_DIR}/train_label_coordinates.fix01b.csv')
	coord_df = coord_df.sort_values(['study_id', 'series_id', 'level', 'condition', ])

	id_df = id_df[id_df.series_description==target_description]
	processed_df = []
	for i,d in id_df.iterrows():

		dicom_df = pd.read_csv(f'{DATA_PROCESSED_DIR}/mini-clean5.0/{d.study_id}/{d.series_id}/df.csv')
		instance_number_to_z_map={
			n:z for (n,z) in dicom_df[['instance_number', 'z']].values
		}
		this_coord_df = coord_df[
				(coord_df.study_id == d.study_id)
				& (coord_df.series_id == d.series_id)
				& (coord_df.condition == 'spinal_canal_stenosis')
			]
		this_grade_df = grade_df[
			(grade_df.study_id == d.study_id)
		]
		if not((len(this_coord_df)==5)):
			print('skipping',d.study_id, d.series_id)
			#skipping 3637444890
			continue

		this_coord_df = this_coord_df.sort_values('level')
		instance_number = this_coord_df['instance_number'].tolist()
		z = [instance_number_to_z_map[n] for n in instance_number]
		xy = this_coord_df[['x','y']].values.tolist()
		xyz = [[x,y,s] for s,(x,y) in zip(z,xy)]

		grade = this_grade_df[
			['spinal_canal_stenosis_' + l for l in level_col]
		].values[0].tolist()
		level = level_col

		one_row = dotdict(
			study_id=d.study_id,
			series_id=d.series_id,
			series_description=target_description,
			instance_number = instance_number,
			level = level,
			grade = grade,
			xyz = xyz,
		)
		processed_df.append(one_row)

	processed_df= pd.DataFrame(processed_df)
	print(processed_df)
	processed_df = processed_df.reset_index(drop=True)

	csv_file = f'{DATA_PROCESSED_DIR}/scs_sag_t2_processed_df.csv'
	processed_df.to_csv(csv_file, index=False)
	print('saved:', csv_file)
	#[1973 rows x 7 columns]



# main #################################################################
if __name__ == '__main__':
	run_make_series_data()
	run_make_nfn_data()
	run_make_scs_data()
