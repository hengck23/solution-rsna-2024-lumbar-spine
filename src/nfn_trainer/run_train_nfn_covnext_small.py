from trainer import *
from configure import *

# start here ###################################################################################

cfg = deepcopy(default_cfg)
cfg.experiment_name = 'one-stage-nfn-fixed/convnext_small-decoder3d-01'
cfg.arch = 'convnext_small.fb_in22k'
cfg.comment  = 'xxx'

if 1:
	for f in [0,1,2,3,4]:
		cfg.fold = f
		cfg.fold_dir = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.lr        =  5e-5
		cfg.num_epoch =  70

		cfg.resume_from.iteration  = -1
		cfg.resume_from.checkpoint =  None    # cfg.fold_dir + '/checkpoint/00015720.pth'
		run_trainer(cfg)
