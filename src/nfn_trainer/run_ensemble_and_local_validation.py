import warnings

import pandas as pd

warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dataset import *
from model import *

from my_lib.runner import *
from my_lib.file import *
from my_lib.net.rate import get_learning_rate
from my_lib.draw import *

from sklearn.metrics import roc_curve
import sklearn.metrics

#######################################################33

def run_infer_and_save(cfg, ensemble_dir):
	os.makedirs(ensemble_dir, exist_ok=True)

	log = Logger()
	log.open(ensemble_dir + '/log.infer.txt', mode='a')
	log.write(f'\n--- [START {log.timestamp()}] {"-" * 64}')
	log.write(f'__file__ = {__file__}\n')
	log.write(f'cfg:\n{format_dict(cfg)}')
	log.write(f'')

	# --- dataset ---
	processed_df = load_csv()
	train_id, valid_id = make_random_split(fold=cfg.fold)
	valid_dataset = SplineDataset(processed_df, valid_id, cfg=cfg, augment=make_valid_augment, mode='valid')

	valid_loader = DataLoader(
		valid_dataset,
		sampler=SequentialSampler(valid_dataset),
		batch_size=cfg.valid_batch_size,
		drop_last=False,
		num_workers=cfg.valid_num_worker,
		pin_memory=True,
		collate_fn=null_collate,
	)
	log.write(f'fold = {cfg.fold}')
	log.write(f'valid_dataset : \n{str(valid_dataset)}')
	log.write('\n')

	# ---model ---
	scaler = torch.cuda.amp.GradScaler(enabled=cfg.is_amp)
	net = Net(pretrained=True, cfg=cfg)
	net.cuda()
	log.write(f'net:\n\t{str(net.arch)}')

	if cfg.resume_from.checkpoint is not None:
		f = torch.load(cfg.resume_from.checkpoint, map_location=lambda storage, loc: storage)
		state_dict = f['state_dict']
		print(net.load_state_dict(state_dict, strict=False))  # True
		iteration = f.get('iteration', 0)

	### start here! ################################################
	if 1:
		result = dotdict(
			D=[],
			grade_truth=[],
			grade=[],
			xyz_truth=[],
			xy=[],
			z=[],
		)
		num_valid = 0
		start_timer = timer()

		net.cuda()
		net.eval()
		net.output_type = ['loss', 'infer']

		for t, batch in enumerate(valid_loader):
			with torch.cuda.amp.autocast(enabled=cfg.is_amp):
				with torch.no_grad():
					output = net(batch)

			B = len(batch['index'])
			num_valid += B
			result.grade_truth.append(batch['grade'].data.cpu().numpy())
			result.grade.append(output['grade'].data.cpu().numpy())
			result.xyz_truth.append(torch.cat([
				batch['xy'],batch['z'].unsqueeze(2)],2).data.cpu().numpy())
			result.xy.append(output['xy'].data.cpu().numpy())
			result.z.append(output['z'].data.cpu().numpy())

			print(f'\r validation: {num_valid}/{len(valid_dataset)}', time_to_str(timer() - start_timer, 'min'),
				  end='', flush=True)

		print('')
		#-------------
		xyz_truth = np.concatenate(result.xyz_truth)
		z = np.concatenate(result.z)
		xy = np.concatenate(result.xy)
		grade = np.concatenate(result.grade)
		grade_truth = np.concatenate(result.grade_truth)

		np.savez(f'{ensemble_dir}/{net.arch}-fold{cfg.fold}-{iteration:08d}.result.npz',
				 grade=grade, grade_truth=grade_truth, xy=xy, z=z, xyz_truth=xyz_truth)

		log.write('do_local_lb():')
		log.write(str(do_local_lb(grade, grade_truth, False)))
		log.write('do_compute_point_error():')
		log.write(str(do_compute_point_error(xy, z, xyz_truth,)))
		log.write('\n\n')


def run_ensemble_and_eval(ensemble_dir, name):

	df=[]
	for fold in [0,1,2,3,4]:
		avg_grade,avg_z, avg_xy = 0,0,0
		num_name= len(name)
		for n in name:
			npz_file = glob(f'{ensemble_dir}/{n}-fold{fold}*.result.npz'.replace('[', '[[]'))[0]
			npz =  np.load(npz_file)

			npz_file = npz_file.split('/')[-1]
			print(npz_file)

			loss, weighted_loss, = do_local_lb(npz['grade'], npz['grade_truth'], False)
			x_err,y_err,z_err,threshold = do_compute_point_error(npz['xy'], npz['z'], npz['xyz_truth'] )
			one_row = {
				#'weight':npz_file,
				'name' : f'{n}',
				'fold': fold,
				'lb': weighted_loss,
				'x<1': x_err[0],
				'x<2': x_err[1],
				'x<5': x_err[3],
				'y<1': y_err[0],
				'y<2': y_err[1],
				'y<5': y_err[3],
				'z<1': z_err[0],
				'z<2': z_err[1],
				'z<5': z_err[3],
			}
			df.append(one_row)
			avg_grade += npz['grade']
			avg_xy += npz['xy']
			avg_z += npz['z']

		#----------------------------------------
		avg_grade /=num_name
		avg_xy /=num_name
		avg_z /=num_name

		loss, weighted_loss, = do_local_lb(avg_grade, npz['grade_truth'], False)
		x_err, y_err, z_err, threshold = do_compute_point_error(avg_xy, avg_z, npz['xyz_truth'])
		one_row = {
			'name' : f'ensemble',
			'fold': fold,
			'lb': weighted_loss,
			'x<1': x_err[0],
			'x<2': x_err[1],
			'x<5': x_err[3],
			'y<1': y_err[0],
			'y<2': y_err[1],
			'y<5': y_err[3],
			'z<1': z_err[0],
			'z<2': z_err[1],
			'z<5': z_err[3],
		}
		df.append(one_row)
	#-----------------------------
	print('')
	df = pd.DataFrame(df)
	df = df.sort_values(by=['name','fold'])
	df.to_csv(f'{ensemble_dir}/ensemble.csv', index=False)
	print(df)



# main #################################################################
if __name__ == '__main__':
	from configure import *

	cfg = deepcopy(default_cfg)
	ensemble_dir = f'{RESULT_DIR}/one-stage-nfn-fixed/ensemble'
	if 1:
		cfg.experiment_name = 'one-stage-nfn-fixed/pvt_v2_b4-decoder3d-01'
		cfg.image_size = 320
		cfg.mask_size  = 320//4
		cfg.arch = 'pvt_v2_b4'
		for f in [0,1,2,3,4]:
			cfg.fold = f
			cfg.resume_from.checkpoint = (f'{RESULT_DIR}/{cfg.experiment_name}' + [
				'/fold0-fix-flip-aug-00032204.pth',
				'/fold1-fix-flip-aug-00029400.pth',
				'/fold2-fix-flip-aug-00015086.pth',
				'/fold3-fix-flip-aug-00031047.pth',
				'/fold4-fix-flip-aug-00028712.pth',
			][f])
			# cfg.fold_dir = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
			# cfg.resume_from.checkpoint = (cfg.fold_dir + [
			#
			# 	'/checkpoint/00032592.pth',
			# 	'/checkpoint/00032144.pth',
			# 	'/checkpoint/00036524.pth',
			# 	'/checkpoint/00032619.pth',
			# 	'/checkpoint/00029488.pth',
			# ][f])
			run_infer_and_save(cfg, ensemble_dir)

	if 1:
		cfg.experiment_name = 'one-stage-nfn-fixed/convnext_small-decoder3d-01'
		cfg.image_size = 320
		cfg.mask_size  = 320//4
		cfg.arch = 'convnext_small.fb_in22k'
		for f in [0,1,2,3,4]:
			cfg.fold = f
			cfg.resume_from.checkpoint = (f'{RESULT_DIR}/{cfg.experiment_name}' + [
				'/fold0-00022892.pth',
				'/fold1-00022736.pth',
				'/fold2-00019850.pth',
				'/fold3-00023580.pth',
				'/fold4-00020176.pth',
			][f])
			run_infer_and_save(cfg, ensemble_dir)
	if 1:
		cfg.experiment_name = 'one-stage-nfn-fixed/effnet_b5-decoder3d-01'
		cfg.image_size = 384
		cfg.mask_size  = 384//4
		cfg.arch = 'tf_efficientnet_b5.ns_jft_in1k'
		cfg.valid_batch_size=16
		for f in [0,1,2,3,4]:
			cfg.fold = f
			cfg.resume_from.checkpoint = (f'{RESULT_DIR}/{cfg.experiment_name}' + [
				'/fold0-00023782.pth',
				'/fold1-00027666.pth',
				'/fold2-00031211.pth',
				'/fold3-00026724.pth',
				'/fold4-00032116.pth',
			][f])
			run_infer_and_save(cfg, ensemble_dir)

	#exit(0)
	run_ensemble_and_eval(ensemble_dir,name=['pvt_v2_b4','convnext_small.fb_in22k','tf_efficientnet_b5.ns_jft_in1k'])

