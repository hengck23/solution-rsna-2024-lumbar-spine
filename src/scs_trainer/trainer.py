import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from dataset import *
from model import *

from my_lib.runner import *
from my_lib.file import *
from my_lib.net.rate import get_learning_rate
from my_lib.draw import *

from sklearn.metrics import roc_curve
import sklearn.metrics
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from timm.utils import ModelEmaV2

###############################################################################

def run_trainer(cfg):
	# --- setup ---
	seed_everything(cfg.seed)
	os.makedirs(cfg.fold_dir, exist_ok=True)
	for f in ['checkpoint', 'train', 'valid', 'etc']:
		os.makedirs(cfg.fold_dir + '/' + f, exist_ok=True)

	log = Logger()
	log.open(cfg.fold_dir + '/log.train.txt', mode='a')
	log.write(f'\n--- [START {log.timestamp()}] {"-" * 64}')
	log.write(f'__file__ = {__file__}\n')
	log.write(f'cfg:\n{format_dict(cfg)}')
	log.write(f'')

	# --- dataset ---
	processed_df = load_csv()
	train_id, valid_id = make_random_split(fold=cfg.fold)
	train_dataset = SplineDataset(processed_df, train_id, cfg=cfg, augment=make_train_augment, mode='train')#make_train_augment
	valid_dataset = SplineDataset(processed_df, valid_id, cfg=cfg, augment=make_valid_augment, mode='valid')

	train_loader = DataLoader(
		train_dataset,
		sampler=RandomSampler(train_dataset),
		batch_size=cfg.train_batch_size,
		drop_last=True,
		num_workers=cfg.train_num_worker,
		pin_memory=True,
		worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
		collate_fn=null_collate,
	)
	valid_loader = DataLoader(
		valid_dataset,
		sampler=SequentialSampler(valid_dataset),
		batch_size=cfg.valid_batch_size,
		drop_last=False,
		num_workers=cfg.valid_num_worker,
		pin_memory=True,
		collate_fn=null_collate,
	)
	num_train_batch = len(train_loader)

	log.write(f'fold = {cfg.fold}')
	log.write(f'valid_dataset : \n{str(valid_dataset)}')
	log.write(f'train_dataset : \n{str(train_dataset)}')
	log.write(f'num_train_batch : \n{num_train_batch}')
	log.write('\n')

	# ---model ---
	scaler = torch.cuda.amp.GradScaler(enabled=cfg.is_amp)
	net = Net(pretrained=True, cfg=cfg)
	net.cuda()
	ema = ModelEmaV2(net, decay=0.99)
	log.write(f'net:\n\t{str(net.arch)}')

	optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
	log.write(f'optimizer:\n\t{str(optimizer)}')
	log.write('')

	# --- loop ---
	start_iteration = 0
	start_epoch = 0
	if cfg.resume_from.checkpoint is not None:
		f = torch.load(cfg.resume_from.checkpoint, map_location=lambda storage, loc: storage)
		state_dict = f['state_dict']
		print(net.load_state_dict(state_dict, strict=False))  # True
		if cfg.resume_from.iteration < 0:
			start_iteration = f.get('iteration', 0)
			start_epoch = f.get('epoch', 0)
		ema.set(net)

	if cfg.is_torch_compile:
		net = torch.compile(net, dynamic=True)

	iter_save = int(cfg.epoch_save * num_train_batch)
	iter_valid = int(cfg.epoch_valid * num_train_batch)
	iter_log = int(cfg.epoch_log * num_train_batch)
	train_loss = MyMeter(None, min(100, num_train_batch))  # window must be less than num_train_batch
	valid_loss = [0, 0, ]

	# logging
	def message_header():
		text = ''
		text += f'** start training here! **\n'
		text += f'   experiment_name = {cfg.experiment_name} \n'
		text += f'                            |---------- VALID--------------|------ TRAIN/BATCH --------------------\n'
		text += f'                            |        loss                  | loss              |                    \n'
		text += f'rate      iter       epoch  | y_acc  level_mask  grade lb  | level_mask  grade |  time  \n'
		text += f'-------------------------------------------------------------------------------------------------------\n'
			    # 5.00e-5  00000521*    1.00  |  0.994  0.468  1.003  0.672  |  0.442  1.089  |  0 hr 03 min :  35 gb
		text = text[:-1]
		return text

	def message(mode='print'):
		if mode == 'print':
			loss = batch_loss
		if mode == 'log':
			loss = train_loss

		if (iteration % iter_save == 0):
			asterisk = '*'
		else:
			asterisk = ' '

		lr = get_learning_rate(optimizer)[0]
		lr = short_e_format(f'{lr:0.2e}')

		timestamp = time_to_str(timer() - start_timer, 'min')
		text = ''
		text += f'{lr}  {iteration:08d}{asterisk}  {epoch:6.2f}  |  '

		for v in valid_loss:
			text += f'{v:5.3f}  '
		text += f'|  '

		for v in loss:
			text += f'{v:5.3f}  '
		text += f'| '

		text += f'{timestamp} : '
		text += f'{get_used_mem():3d} gb'
		return text

	### start training here! ################################################

	def do_valid(net, iteration):

		result = dotdict(
			D=[],

			grade_loss=0,
			grade_truth=[],
			grade=[],

			level_mask_loss=0,
			zxy_loss=0,
			xy_truth=[],
			z_truth=[],
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
			result.grade_loss += B * output['grade_loss'].item()
			result.level_mask_loss += B * output['level_mask_loss'].item()

			result.grade_truth.append(batch['grade'].data.cpu().numpy())
			result.grade.append(output['grade'].data.cpu().numpy())

			result.xy_truth.append(batch['xyz'][...,[0,1]].data.cpu().numpy())
			result.xy.append(output['xy'].data.cpu().numpy())
			#--------------
			if 0:
				#legacy code (copmute z from level mask)

				D = batch['D']
				level_mask  = output['level_mask'].float().data.cpu().numpy()
				level_mask_truth  = batch['level_mask'].float().data.cpu().numpy()
				B = len(D)
				for b in range(B):
					g  = level_mask[b] #d,_,h,w
					gz = g.sum((2))
					p  = gz.argmax(1) #predict

					g_truth = level_mask_truth[b]
					gz_truth = g_truth.sum((2))
					p_truth = gz_truth.argmax(1)
					valid = g_truth.sum((1,2))!=0

					result.z.append(p[valid])
					result.z_truth.append(p_truth[valid])

			######################################################################

			print(f'\r validation: {num_valid}/{len(valid_dataset)}', time_to_str(timer() - start_timer, 'min'),
				  end='', flush=True)

		# ----
		grade_loss = result.grade_loss/ num_valid
		level_mask_loss = result.level_mask_loss / num_valid

		grade_truth = np.concatenate(result.grade_truth)
		grade = np.concatenate(result.grade)
		loss, weighted_loss = do_local_lb(grade, grade_truth)

		#check if level is correct
		xy = np.concatenate(result.xy)
		xy_truth = np.concatenate(result.xy_truth)
		diff = np.abs(xy[...,1]-xy_truth[...,1])
		y_acc = (diff<=2.5).mean()


		valid_loss = [
			y_acc, level_mask_loss, grade_loss, weighted_loss
		]
		return valid_loss


	# ---------------------------------------
	iteration = start_iteration
	epoch = start_epoch
	start_timer = timer()
	log.write(message_header())

	break_while_loop=0
	while break_while_loop==0:

		for t, batch in enumerate(train_loader):
			# --- start of callback ---
			if iteration % iter_save == 0:
				if iteration != start_iteration:
					torch.save({
						# 'state_dict': net.state_dict(),
						#'state_dict': getattr(net, '_orig_mod', net).state_dict(),
						'state_dict': ema.module.state_dict(),
						'iteration': iteration,
						'epoch': epoch,
					}, f'{cfg.fold_dir}/checkpoint/{iteration:08d}.pth')
					pass

			if iteration % iter_valid == 0:
				valid_loss = do_valid( ema.module, iteration )
				pass

			if (iteration % iter_log == 0) or (iteration % iter_valid == 0):
				print('\r', end='', flush=True)
				log.write(message(mode='log'))

			if break_while_loop: break
			# --- end of callback ----

			net.train()
			net.output_type = ['loss', 'infer']

			if 1:# len(batch['xy']) !=0:
				with torch.cuda.amp.autocast(enabled=cfg.is_amp):
					output = net(batch)

					level_mask_loss = output['level_mask_loss']
					grade_loss = output['grade_loss']
					z_mask_loss = output['z_mask_loss']
					xy_loss = output['xy_loss']
					loss =  level_mask_loss  + xy_loss+ z_mask_loss + grade_loss
					batch_loss = [ level_mask_loss.item(), grade_loss.item(),]

				optimizer.zero_grad()
				# if batch['target'].sum()>0:
				if 1:
					if cfg.is_amp:
						scaler.scale(loss).backward()
						# scaler.unscale_(optimizer)
						# torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
						scaler.step(optimizer)
						scaler.update()
					else:
						loss.backward()
						optimizer.step()

				ema.update(net)
				torch.clear_autocast_cache()

				# print---
				train_loss.step(batch_loss)
				print('\r', end='', flush=True)
				print(message(mode='print'), end='', flush=True)

			iteration += 1
			epoch += 1 / num_train_batch
			if epoch > cfg.num_epoch+2 / num_train_batch:
				break_while_loop=1
		# print('')


# main #################################################################
if __name__ == '__main__':
	from configure import *

	cfg = deepcopy(default_cfg)
	cfg.experiment_name = 'one-stage-scs/pvt_v2_b4-decoder2d-01'
	cfg.lr = 5e-5 # 5e-5 # 1e-4
	cfg.num_epoch = 30
	cfg.comment =  'xxx'

	cfg.train_batch_size = 3
	cfg.valid_batch_size = 2

	cfg.level_sigma=1
	cfg.image_size=320
	cfg.mask_size=320//4
	cfg.arch = 'pvt_v2_b4'
		#'tf_efficientnet_b4.ns_jft_in1k'
	    #'convnext_base.fb_in22k'
     	# 'pvt_v2_b4'

	#for f in [0,1,2,3,4]:
	for f in [0,1,2,3,4]:
		cfg.fold = f
		cfg.fold_dir = f'{RESULT_DIR}/{cfg.experiment_name}/fold-{cfg.fold}'
		cfg.resume_from.iteration  = -1
		cfg.resume_from.checkpoint = \
		    None #cfg.fold_dir + '/checkpoint/00011067.pth' # fold-4

		run_trainer(cfg)
