import sys, os
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

#------------------------------------------------
# add path for local lib

THIRD_PARTY_DIR = os.path.dirname(os.path.realpath(__file__)) + '/third_party'
print('THIRD_PARTY_DIR :', THIRD_PARTY_DIR)
sys.path.append(THIRD_PARTY_DIR)
# mark as root sources in pycharm

### common python lib ########################################
# from my_lib.net.rate import *
from my_lib.other import *
from my_lib.draw import *
from my_lib.file import *

import math
import numpy as np
import random
import time
import cv2
import pandas as pd
import json
import zipfile
from shutil import copyfile
from timeit import default_timer as timer
import itertools
import collections
from collections import OrderedDict
from collections import defaultdict
from glob import glob
from copy import deepcopy


from tqdm import tqdm
from print_dict import format_dict
from  ast import literal_eval


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
#matplotlib.use('WXAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg')
print('matplotlib.get_backend : ', matplotlib.get_backend())



### deep learning framework #################################
if 1: # for pytorch
	import torch
	from torch.utils.data.dataset import Dataset
	from torch.utils.data import DataLoader
	from torch.utils.data.sampler import *
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.optim as optim
	from torch.nn.parallel.data_parallel import data_parallel

	def pytorch_version_to_text():
		text = ''
		text += '\tpytorch\n'
		text += '\t\ttorch.__version__              = %s\n' % torch.__version__
		text += '\t\ttorch.version.cuda             = %s\n' % torch.version.cuda
		text += '\t\ttorch.backends.cudnn.version() = %s\n' % torch.backends.cudnn.version()
		text += '\t\ttorch.cuda.device_count()      = %d\n' % torch.cuda.device_count()
		text += '\t\ttorch.cuda.get_device_properties() = %s\n' % str(torch.cuda.get_device_properties(0))[22:-1]
		text += '\n'
		return text
''' 
print(common_string)
	pytorch
		torch.__version__              = 2.0.1+cu117
		torch.version.cuda             = 11.7
		torch.backends.cudnn.version() = 8500
		torch.cuda.device_count()      = 1
		torch.cuda.get_device_properties() = (name='NVIDIA TITAN X (Pascal)', major=6, minor=1, total_memory=12193MB, multi_processor_count=28)

	pytorch
		torch.__version__              = 2.3.0+cu121
		torch.version.cuda             = 12.1
		torch.backends.cudnn.version() = 8902
		torch.cuda.device_count()      = 2
		torch.cuda.get_device_properties() = (name='NVIDIA RTX 6000 Ada Generation', major=8, minor=9, total_memory=48622MB, multi_processor_count=142)

'''



### other kaggle task ####################################
if 0:  # for CT scan related
	# https://github.com/tsangel/dicomsdl
	import nibabel as nib
	import pydicom




######################################################3

if __name__ == '__main__':
	print(pytorch_version_to_text())
