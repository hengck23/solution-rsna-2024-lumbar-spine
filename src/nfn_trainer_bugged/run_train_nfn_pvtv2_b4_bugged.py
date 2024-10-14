from trainer import *
from configure import *

# start here ###################################################################################

# here we reuse the parameters to produce the results of winning kaggle submission.
# at that time, we are in early experiments, training different folds with different
# learning rate, etc ...
#
# later we find that it is sufficient just to use 1e-5 for all folds with 100 epoch,
# then select the model with the est lb metric from the train log.

cfg = deepcopy(default_cfg)
cfg.experiment_name = 'one-stage-nfn-bugged/pvt_v2_b4-decoder3d-01'
cfg.arch = 'pvt_v2_b4'
cfg.comment  = 'xxx'

if 1:  #early training
	for f in [0,1,2,3,4]:
		cfg.fold = f
		cfg.fold_dir = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.lr        =  [1e-4, 1e-4, 1e-5, 1e-5, 1e-5][f]
		cfg.num_epoch =  [40, 40, 30, 45, 45][f]

		cfg.resume_from.iteration  = -1
		cfg.resume_from.checkpoint =  None    # cfg.fold_dir + '/checkpoint/00015720.pth'
		run_trainer(cfg)

if 1:  #finetune with smaller rate
	for f in [0,1,2,3,4]:
		cfg.fold = f
		cfg.lr        =  1e-5
		cfg.num_epoch =  100

		cfg.resume_from.iteration  = -1
		cfg.resume_from.checkpoint =  [
			cfg.fold_dir + '/checkpoint/00010850.pth',
			cfg.fold_dir + '/checkpoint/00012544.pth',
			cfg.fold_dir + '/checkpoint/00009925.pth',
			cfg.fold_dir + '/checkpoint/00015720.pth',
			cfg.fold_dir + '/checkpoint/00015520.pth',
		][f]
		run_trainer(cfg)





