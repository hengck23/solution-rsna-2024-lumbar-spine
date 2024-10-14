import sys
sys.path.append('..')
from common import *

from _dir_setting_ import *
from my_lib.other import *


#====================================================================================
default_cfg = dotdict(

	# --- envir ---
	is_amp=True, #fp16
	is_torch_compile=False,

	# --- experiment ---
	fold=0,
	seed=1234,
	experiment_name = 'resnet50d-nfn-xxx',
	fold_dir = f'{RESULT_DIR}/default',
		       #f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'

	# --- dataset ---
	image_size=320,
	mask_size=320//4,
	point_sigma=0,

	train_num_worker=32,
	train_batch_size=4,
	valid_num_worker=32,
	valid_batch_size=32,

	# --- model ---
	lr=1e-4,

	# --- loop ---
	resume_from=dotdict(
		iteration=-1,
		checkpoint= None, #cfg.fold_dir + '/checkpoint/00009416.pth'
	),

	num_epoch  = 30,
	epoch_valid = 1,
	epoch_log   = 1,
	epoch_save  = 1,
)








