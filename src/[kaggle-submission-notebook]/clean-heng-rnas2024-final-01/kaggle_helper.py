import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import sklearn.metrics

## submission tools -----
level_color = [
    [255, 0, 0],
    [0, 255, 0],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
]

grade_col = [
    'normal_mild', 'moderate', 'severe'
]
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
grade_map = {
    'Missing': -1,
    'Normal/Mild': 0,
    'Moderate': 1,
    'Severe': 2,
}

def make_dummy_submit(valid_df, dummy_grade=None):

    study_id = valid_df.study_id.unique()
    if dummy_grade is None:
        dummy_grade = \
            np.full((len(study_id) * len(condition_level_col), 3), fill_value=1/3).astype(np.float32)

    submit_df = pd.DataFrame(dummy_grade, columns=grade_col)
    submit_df.loc[:, 'row_id'] = [f'{s}_{c}' for s in study_id for c in condition_level_col]
    submit_df = submit_df.set_index('row_id')
    print('make_dummy_submit():')
    print('\t',submit_df.head(20))
    print('\t',submit_df.shape, submit_df.dtypes)
    print('')
    return submit_df

def make_scs_grade_submit(result):

    study_id = list(result.keys())
    scs = np.stack(list([np.array(v.grade) for k,v in result.items()])).astype(np.float32)
    print('scs', scs.shape)
    scs_df = pd.DataFrame(scs.reshape(-1, 3), columns=grade_col)
    scs_df.loc[:, 'row_id'] = [f'{s}_{c}' for s in study_id for c in [
        'spinal_canal_stenosis_l1_l2',
        'spinal_canal_stenosis_l2_l3',
        'spinal_canal_stenosis_l3_l4',
        'spinal_canal_stenosis_l4_l5',
        'spinal_canal_stenosis_l5_s1',
    ]]
    scs_df = scs_df.set_index('row_id')

    print('make_scs_submit():')
    print('\t',scs_df.head(5))
    print('\t',scs_df.shape, scs_df.dtypes)
    print('')
    return scs_df

def make_nfn_grade_submit(result):

    study_id = list(result.keys())
    nfn = np.stack(list([np.array(v.grade) for k,v in result.items()])).astype(np.float32)
    print('nfn', nfn.shape)
    nfn_df = pd.DataFrame(nfn.reshape(-1, 3), columns=grade_col)
    nfn_df.loc[:, 'row_id'] = [f'{s}_{c}' for s in study_id for c in [
        'left_neural_foraminal_narrowing_l1_l2',
        'left_neural_foraminal_narrowing_l2_l3',
        'left_neural_foraminal_narrowing_l3_l4',
        'left_neural_foraminal_narrowing_l4_l5',
        'left_neural_foraminal_narrowing_l5_s1',
        'right_neural_foraminal_narrowing_l1_l2',
        'right_neural_foraminal_narrowing_l2_l3',
        'right_neural_foraminal_narrowing_l3_l4',
        'right_neural_foraminal_narrowing_l4_l5',
        'right_neural_foraminal_narrowing_l5_s1',
    ]]
    nfn_df = nfn_df.set_index('row_id')

    print('make_nfn_submit():')
    print('\t',nfn_df.head(10))
    print('\t',nfn_df.shape, nfn_df.dtypes)
    print('')
    return nfn_df

def make_scs_grade_truth(result, grade_truth_df):
    truth_df = grade_truth_df.set_index('study_id')
    study_id = list(result.keys())

    truth_df = truth_df.loc[study_id,[
        'spinal_canal_stenosis_l1_l2',
        'spinal_canal_stenosis_l2_l3',
        'spinal_canal_stenosis_l3_l4',
        'spinal_canal_stenosis_l4_l5',
        'spinal_canal_stenosis_l5_s1',
    ]]
    truth_df = truth_df.map(lambda x: grade_map[x])
    return truth_df

def make_nfn_grade_truth(result, grade_truth_df):
    truth_df = grade_truth_df.set_index('study_id')
    study_id = list(result.keys())

    truth_df = truth_df.loc[study_id,[
        'left_neural_foraminal_narrowing_l1_l2',
        'left_neural_foraminal_narrowing_l2_l3',
        'left_neural_foraminal_narrowing_l3_l4',
        'left_neural_foraminal_narrowing_l4_l5',
        'left_neural_foraminal_narrowing_l5_s1',
        'right_neural_foraminal_narrowing_l1_l2',
        'right_neural_foraminal_narrowing_l2_l3',
        'right_neural_foraminal_narrowing_l3_l4',
        'right_neural_foraminal_narrowing_l4_l5',
        'right_neural_foraminal_narrowing_l5_s1',
    ]]
    truth_df = truth_df.map(lambda x: grade_map[x])
    return truth_df

def do_local_lb(probability, truth, is_any=False):
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

	#---
	any_loss=-1
	if is_any:
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

	#---
	return loss, weighted_loss, any_loss
