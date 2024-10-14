import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt

from data import *
from kaggle_helper import *
from scs_sag_t2_all_model import Net as SCSJointNet

def load_net(Net, checkpoint, cfg, device):
    net = Net(pretrained=False, cfg=cfg)  #
    state_dict = torch.load(
        checkpoint,
        map_location=lambda storage, loc: storage, weights_only=True)['state_dict']
    print(net.load_state_dict(state_dict, strict=False))  # True

    net = net.eval()
    net.output_type = ['infer']
    net = net.to(torch.device(device))
    return net


def run_scs( valid_df, cfg, IMAGE_DIR, MODE, DEVICE ):

    net = [
        load_net(SCSJointNet, checkpoint, cfg, DEVICE)
        for checkpoint in cfg.checkpoint
    ]
    num_net = len(net)

    result = {}

    study_id = valid_df.study_id.unique()
    num_study_id = len(study_id)
    for i in range(num_study_id):
        df = valid_df[(valid_df.study_id == study_id[i]) &
                      (valid_df.series_description == 'Sagittal T2/STIR')]
        series_id = df.series_id.tolist()
        num_series_id = len(series_id)

        #if 1:
        try:
            grade = []
            xy, z = [], []
            for j in range(num_series_id):
                print('\r', i, j, study_id[i], series_id[j], end='', flush=True)

                volume, dicom_df, error_code = heng_read_series(study_id[i], series_id[j], 'Sagittal T2/STIR', IMAGE_DIR)
                if volume is None: continue

                image = np.ascontiguousarray(volume.transpose(1, 2, 0))
                image, resize_param1 = do_resize_and_center(image, reference_size=512)  # bug #to solved later
                image, resize_param2 = do_resize_and_center(image, reference_size=cfg.image_size)
                image = np.ascontiguousarray(image.transpose(2, 0, 1))

                batch = {
                    'D': [len(image)],
                    'image': torch.from_numpy(image).byte().to(DEVICE),
                }

                with torch.amp.autocast(enabled=True, device_type=DEVICE):
                    with torch.no_grad():
                        for k in range(num_net):
                            output = net[k](batch)

                og = output['grade'].data.cpu().numpy()[0]
                oxy = output['xy'].data.cpu().numpy()[0]
                oz = output['z'].data.cpu().numpy()[0]
                grade.append(og)
                xy.append(oxy)
                z.append(oz)

            grade = np.stack(grade).mean(0)
            xy = np.stack(xy).mean(0)
            z = np.stack(z).mean(0)

            # convert but to orginal image as in the ground truth label csv
            # s1, dx1, dy1 = resize_param1
            # s2, dx2, dy2 = resize_param2
            # xy2 = (xy*4 - [[dx2, dy2]]) / s2
            # xy1 = (xy2 - [[dx1, dy1]]) / s1
            # xy  = xy1
            z = np.round(z).astype(np.int32)
            z_to_instance_number_map = {
                z: n for (n, z) in dicom_df[['instance_number', 'z']].values
            }
            instance_number = [z_to_instance_number_map.get(s, -1) for s in z]
            # ---

            result[study_id[i]] = dotdict(
                grade=grade.tolist(),
                xy=xy.tolist(),
                z=z.tolist(),
                instance_number=instance_number,
            ) #L1...5, R1...5,

        #if 0:
        except:
            print('UNKNOWN ERROR?????', 'study_id', study_id[i])
            pass

        if (len(result) == 80) & (MODE == 'local'):
            print('')
            print('skipping and break!!!')
            break

        if i<2:
            #print for debug
            print('')
            print('study_id', study_id[i])
            print('series_id', series_id[j])
            print('volume', volume.shape)
            print('image', image.shape)
            print('dicom_df', dicom_df.shape)

            print('grade', grade.shape)
            print(result[study_id[i]])
            overlay = image.mean(0).astype(np.uint8)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
            for l in range(5):
                x,y = xy[l]
                x= round(int(x*4))
                y= round(int(y*4))
                color=level_color[l%5]
                cv2.circle(overlay, (x, y), 10, color, 1,cv2.LINE_AA)

            plt.imshow(overlay, cmap='gray')
            plt.show()
    print('')
    return result
