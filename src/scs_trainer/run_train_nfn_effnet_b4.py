from trainer import *
from configure import *

# start here ###################################################################################

cfg = deepcopy(default_cfg)
cfg.experiment_name = 'one-stage-scs/effnet_b4-decoder2d-01'
cfg.arch = 'tf_efficientnet_b4.ns_jft_in1k'
cfg.comment  = 'xxx'
cfg.image_size=384
cfg.mask_size=384//4
cfg.train_batch_size=3
cfg.valid_batch_size=8

if 1:
	for f in [0,1,2,3,4]:
		cfg.fold = f
		cfg.fold_dir = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.lr        =  5e-5
		cfg.num_epoch =  31

		cfg.resume_from.iteration  = -1
		cfg.resume_from.checkpoint =  None    # cfg.fold_dir + '/checkpoint/00015720.pth'
		run_trainer(cfg)
