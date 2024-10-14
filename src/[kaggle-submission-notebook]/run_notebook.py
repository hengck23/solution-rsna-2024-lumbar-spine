'''
# download installation wheel for kaggle notebook
if 0:
    !pip download natsort -d /kaggle/working/
    from IPython.display import FileLink
    FileLink(r'./natsort-8.4.0-py3-none-any.whl')
'''

#### begining of new cell ############################################################################

try:
    import natsort
except:
    pass
    # !pip install natsort
    # !pip install natsort --no-index --find-links=file://///kaggle/input/heng-rnas2024-final-01/

import sys
import os
sys.path.append(
    '/home/hp/work/2024/kaggle/rsna2024-lumbar-spine/[final-submit]/code/src/kaggle-submission-notebook/clean-heng-rnas2024-final-01'
    #'/kaggle/input/clean-heng-rnas2024-final-01'
)

from data import *
from kaggle_helper import *

import pandas as pd
from natsort import natsorted
import matplotlib
import matplotlib.pyplot as plt

pd.set_option('display.width', 1000)
print('IMPORT OK !!!!!!!!!!!!!')

#### begining of new cell ############################################################################

KAGGLE_DATA_DIR = (
    # '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
    '/home/hp/work/2024/kaggle/rsna2024-lumbar-spine/data/kaggle/rsna-2024-lumbar-spine-degenerative-classification'
)
WEIGHT_DIR =(
    '/home/hp/work/2024/kaggle/rsna2024-lumbar-spine/[final-submit]/weight'
)

MODE   = 'submit'  # local #submit
DEVICE = 'cuda'   # 'cpu' 'cuda'

if MODE == 'local':
    IMAGE_DIR = f'{KAGGLE_DATA_DIR}/train_images'
    valid_df = pd.read_csv(f'{KAGGLE_DATA_DIR}/train_series_descriptions.csv')
if MODE == 'submit':
    IMAGE_DIR = f'{KAGGLE_DATA_DIR}/test_images'
    valid_df = pd.read_csv(f'{KAGGLE_DATA_DIR}/test_series_descriptions.csv')

# example submission file
submit_df = make_dummy_submit(valid_df)


nfn_cfg = dotdict(
    bugged_pvt_v2_b4 = dotdict(
        arch='pvt_v2_b4',
        image_size=320,
        flip_tta=False,
        checkpoint=[
            f'{WEIGHT_DIR}/nfn/nfn_model_pvtv2_b4_bugged_weight/fold0-00032592.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_pvtv2_b4_bugged_weight/fold0-00032592.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_pvtv2_b4_bugged_weight/fold0-00032592.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_pvtv2_b4_bugged_weight/fold0-00032592.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_pvtv2_b4_bugged_weight/fold0-00032592.pth',
        ],
    ),
    pvt_v2_b4 = dotdict(
        arch='pvt_v2_b4',
        image_size=320,
        flip_tta=True,
        checkpoint=[
            f'{WEIGHT_DIR}/nfn/nfn_model_pvtv2_b4_fixed_weight/fold0-fix-flip-aug-00032204.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_pvtv2_b4_fixed_weight/fold1-fix-flip-aug-00029400.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_pvtv2_b4_fixed_weight/fold2-fix-flip-aug-00015086.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_pvtv2_b4_fixed_weight/fold3-fix-flip-aug-00031047.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_pvtv2_b4_fixed_weight/fold4-fix-flip-aug-00028712.pth',
        ],
    ),
    convnext_small = dotdict(
        arch='convnext_small.fb_in22k',
        image_size=320,
        flip_tta=True,
        checkpoint=[
            f'{WEIGHT_DIR}/nfn/nfn_model_convnext_small_weight/fold0-00022892.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_convnext_small_weight/fold1-00022736.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_convnext_small_weight/fold2-00019850.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_convnext_small_weight/fold3-00023580.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_convnext_small_weight/fold4-00020176.pth',
        ],
    ),
    effnet_b5=dotdict(
        arch='tf_efficientnet_b5.ns_jft_in1k',
        image_size=384,
        flip_tta=True,
        checkpoint=[
            f'{WEIGHT_DIR}/nfn/nfn_model_effb5_weight/fold0-00023782.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_effb5_weight/fold1-00027666.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_effb5_weight/fold2-00031211.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_effb5_weight/fold3-00026724.pth',
            f'{WEIGHT_DIR}/nfn/nfn_model_effb5_weight/fold4-00032116.pth',
        ],
    ),
)

scs_cfg = dotdict(
    pvt_v2_b4 = dotdict(
        arch='pvt_v2_b4',
        image_size=320,
        checkpoint=[
            f'{WEIGHT_DIR}/scs/scs_model_pvtv2_b4_weight/fold0-00009899.pth',
            f'{WEIGHT_DIR}/scs/scs_model_pvtv2_b4_weight/fold1-00008416.pth',
            f'{WEIGHT_DIR}/scs/scs_model_pvtv2_b4_weight/fold2-00012259.pth',
            f'{WEIGHT_DIR}/scs/scs_model_pvtv2_b4_weight/fold3-00009486.pth',
            f'{WEIGHT_DIR}/scs/scs_model_pvtv2_b4_weight/fold4-00006276.pth',
        ],
    ),
    convnext_base = dotdict(
        arch='convnext_base.fb_in22k',
        image_size=320,
        checkpoint=[
            f'{WEIGHT_DIR}/scs/scs_model_convnext_base_weight/fold0-00010941.pth',
            f'{WEIGHT_DIR}/scs/scs_model_convnext_base_weight/fold1-00007890.pth',
            f'{WEIGHT_DIR}/scs/scs_model_convnext_base_weight/fold2-00010127.pth',
            f'{WEIGHT_DIR}/scs/scs_model_convnext_base_weight/fold3-00008959.pth',
            f'{WEIGHT_DIR}/scs/scs_model_convnext_base_weight/fold4-00008891.pth',
        ],
    ),
    effnet_b4=dotdict(
        arch='tf_efficientnet_b4.ns_jft_in1k',
        image_size=384,
        checkpoint=[
            f'{WEIGHT_DIR}/scs/scs_model_effb4_weight/fold0-00015109.pth',
            f'{WEIGHT_DIR}/scs/scs_model_effb4_weight/fold1-00014728.pth',
            f'{WEIGHT_DIR}/scs/scs_model_effb4_weight/fold2-00019721.pth',
            f'{WEIGHT_DIR}/scs/scs_model_effb4_weight/fold3-00011594.pth',
            f'{WEIGHT_DIR}/scs/scs_model_effb4_weight/fold4-00013075.pth',
        ],
    ),
)


print('SETTING OK !!!!!!!!!!!!!')


#### begining of new cell ############################################################################


if 1:
    from nfn_sag_t1_infer import *

    #nfn_result = run_nfn( valid_df, nfn_cfg.bugged_pvt_v2_b4, IMAGE_DIR, MODE, DEVICE )
    #nfn_df = make_nfn_grade_submit(nfn_result)

    nfn_df1 = make_nfn_grade_submit(run_nfn( valid_df, nfn_cfg.pvt_v2_b4, IMAGE_DIR, MODE, DEVICE ))
    nfn_df2 = make_nfn_grade_submit(run_nfn( valid_df, nfn_cfg.convnext_small, IMAGE_DIR, MODE, DEVICE ))
    nfn_df3 = make_nfn_grade_submit(run_nfn( valid_df, nfn_cfg.effnet_b5, IMAGE_DIR, MODE, DEVICE ))
    nfn_df = (nfn_df1+nfn_df2+nfn_df3)/3

    # ----
    # merge
    submit_df.loc[nfn_df.index, grade_col] = nfn_df

    # save
    submit_df.to_csv('submission.csv', index=True)
    print('** FINAL SUBMIT **')
    print(submit_df.head(30))
    print(submit_df.shape)

    print('SUBMIT OK!!!!!')
    if MODE == 'local':
        grade_truth_df = pd.read_csv(f'{KAGGLE_DATA_DIR}/train.csv')
        truth_df = make_nfn_grade_truth(nfn_result, grade_truth_df)
        truth = np.array(truth_df.values.tolist())
        probability = np.array(nfn_df.values.tolist())
        print('nfn:')
        print(do_local_lb(probability, truth, False))

#### begining of new cell ############################################################################
if 1:
    from scs_sag_t2_infer import *

    # scs_result = run_scs(valid_df, scs_cfg.pvt_v2_b4, IMAGE_DIR, MODE, DEVICE)
    # scs_result = run_scs( valid_df, scs_cfg.convnext_base, IMAGE_DIR, MODE, DEVICE )
    scs_result = run_scs( valid_df, scs_cfg.effnet_b4, IMAGE_DIR, MODE, DEVICE )
    scs_df = make_scs_grade_submit(scs_result)

    # ----

    # merge
    submit_df.loc[scs_df.index, grade_col] = scs_df

    # save
    submit_df.to_csv('submission.csv', index=True)
    print('** FINAL SUBMIT **')
    print(submit_df.head(30))
    print(submit_df.shape)

    print('SUBMIT OK!!!!!')
    if MODE == 'local':
        grade_truth_df = pd.read_csv(f'{KAGGLE_DATA_DIR}/train.csv')
        truth_df = make_scs_grade_truth(scs_result, grade_truth_df)
        truth = np.array(truth_df.values.tolist())
        probability = np.array(scs_df.values.tolist())
        print('scs:')
        print(do_local_lb(probability, truth, True))

'''
for debug verification 
nfn_cfg.bugged_pvt_v2_b4

study_id 4003253
series_id 1054713880
volume (15, 384, 384)
image (15, 320, 320)
dicom_df (15, 13)
grade (10, 3)
'grade': [[0.999079704284668, 0.0009172920254059136, 3.026959348062519e-06]

study_id 4646740
series_id 3486248476
volume (17, 540, 384)
image (17, 320, 320)
dicom_df (17, 13)
grade (10, 3)
{'grade': [[0.9185006022453308, 0.08072932809591293, 0.0007700947462581098],

nfn_cfg.bugged_pvt_v2_b4
([0.22723701218767847, 0.8666631775640278, 1.0238806241838554], 0.5134474212608597, -1)
nfn_cfg.pvt_v2_b4
([0.22682431068534087, 0.8538277783151468, 0.7790620735079907], 0.483031911976068, -1)
nfn_cfg.convnext_small
([0.21719899504804527, 0.8553222909222131, 0.924566630632426], 0.49346769173434274, -1)
nfn_cfg.effnet_b5
([0.2265785367620108, 0.888213609166505, 0.8170621529929115], 0.4977641791941786, -1)
---
scs_cfg.pvt_v2_b4
([0.05060841049739512, 1.0678985070648728, 0.6358080051983124], 0.26964103321385474, 0.2644830739771569)
scs_cfg.convnext_base
([0.045310714958383604, 1.110966942427944, 0.5935163620346083], 0.2622622530060437, 0.2632618406403011)
scs_cfg.effnet_b4
([0.05008578044557778, 1.1748753784445447, 0.5871180228694474], 0.2710579779729628, 0.28807300030308763)

'''

