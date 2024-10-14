from trainer import *
from configure import *

# start here ###################################################################################

cfg = deepcopy(default_cfg)
cfg.experiment_name = 'one-stage-scs/pvt_v2_b4-decoder2d-01'
cfg.arch = 'pvt_v2_b4'
cfg.comment  = 'xxx'

cfg.train_batch_size = 3
cfg.valid_batch_size = 2
cfg.level_sigma=1
cfg.image_size=320
cfg.mask_size=320//4

if 1: #early training
	for f in [0,1,2,3,4,5]:
		cfg.fold = f
		cfg.fold_dir = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.lr        =  1e-4
		cfg.num_epoch =  [11,13,6,24,12][f]

		cfg.resume_from.iteration  = -1
		cfg.resume_from.checkpoint =  None    # cfg.fold_dir + '/checkpoint/00015720.pth'
		run_trainer(cfg)

if 1: #finetuning
	for f in  [0,1,2,3,4,5]:
		cfg.fold = f
		cfg.fold_dir = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.lr        =  5e-5
		cfg.num_epoch =  [21,21,28,28,24,][f]

		cfg.resume_from.iteration  = -1
		cfg.resume_from.checkpoint =  [
			cfg.fold_dir + '/checkpoint/00005210.pth',
			cfg.fold_dir + '/checkpoint/00006312.pth',
			cfg.fold_dir + '/checkpoint/00002665.pth',
			cfg.fold_dir + '/checkpoint/00011067.pth',
			cfg.fold_dir + '/checkpoint/00005230.pth',
		][f]
		run_trainer(cfg)
