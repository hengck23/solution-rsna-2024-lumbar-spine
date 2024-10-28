# Kaggle Competition Solution (7th)

# RSNA 2024 Lumbar Spine Degenerative Classification
- Classify lumbar spine degenerative conditions  
https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification 

- For discussion, please refer to:  
https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539439
  
This repo contains the code to train one-stage models used in the team solution for the RSNA 2024 lumbar spine degenerative classification Kaggle competition.

## 1. Hardware  
- GPU: 2x Nvidia Ada A6000 (Ampere), each with VRAM 48 GB
- CPU: Intel® Xeon(R) w7-3455 CPU @ 2.5GHz, 24 cores, 48 threads
- Memory: 256 GB RAM

## 2. OS 
- ubuntu 22.04.4 LTS


## 3. Set Up Environment
- Install Python >=3.10.9
- Install requirements.txt in the python environment
- Set up the directory structure as shown below.
``` 
└── <repo_dir>
    ├── <DATA_KAGGLE_DIR> 
    |         ├── rsna-2024-lumbar-spine-degenerative-classification
    |               ├── test_images
    │               ├── train_images
    │               ├── train.csv
    │               ├── train_label_coordinates.csv
    │               ├── train_series_descriptions.csv
    │               ├── ... other files ...
    ├── <DATA_PROCESSED_DIR>
    |         ├── train_label_coordinates.fix01b.csv
    |         ├── nfn_sag_t1_mean_shape.512.npy
    |         ├── scs_sag_t2_mean.512.npy
    | 
    ├── <RESULT_DIR>
    ├── src 
    ├── LICENSE 
    ├── README.md
    ├── requirements.txt
```
- Modify the path setting by editing  "/src/third_party/\_dir_setting_.py"

```
# please use the full path 
DATA_KAGGLE_DIR     = '... for downloaded and unzipped Kaggle data ... '
DATA_PROCESSED_DIR  = '... for intermediate processed data ...'
RESULT_DIR          = '... for training output like model weights, training logs, etc ...'
```

## 4. Set Up Dataset

- <DATA_KAGGLE_DIR> contains data from "rsna-2024-lumbar-spine-degenerative-classification.zip" at:  
https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data

- <DATA_PROCESSED_DIR> contains the 3 manually created files from the DATA_PROCESSED_DIR folder in this repo.
    - train_label_coordinates.fix01b.csv: correct annotation for spinal canal stenosis point
    - nfn_sag_t1_mean_shape.512.npy, scs_sag_t2_mean.512.npy: mean reference shape created from
https://www.kaggle.com/code/hengck23/shape-alignment
    - Other processed data can be created by running the python script:
  ```
  python "/src/process-data-01/run_make_data.py"  
  ```
    - A backup copy of all processed data can be found in the google-share drive:
  https://drive.google.com/drive/folders/1jPPxAP6DHGQMHJPUGjPO7_Q5Asrj_LL3?usp=sharing 



## 5. Training the model

### Warning !!! training output will be overwritten to the "<RESULT_DIR>" folder

### NFN (neural foraminal narrowing) models
- Bugged Models:
Due to a bug in the flip augmentation (left-right points not reordered) in training, the submitted model weights are 
not correct. You can reproduce the bugged models by running the following script:
```  
cd src/nfn_trainer_bugged
python run_train_nfn_pvtv2_b4_bugged.py

output model:
- <RESULT_DIR>/one-stage-nfn-bugged/pvt_v2_b4-decoder3d-01/
```  
- Fixed Models: The bug was fixed at the last minute, a day before the deadline. We only managed to retrain and use two model weights (fold2,3 out of 5) 
for submission. To train the corrected models, run:

```  
cd src/nfn_trainer
python run_train_nfn_pvtv2_b4_fixed.py

output model:
- <RESULT_DIR>/one-stage-nfn-fixed/pvt_v2_b4-decoder3d-01/
```  
- Optional Models: These are not used in submission. We use various image backbone encoders like convnext and efficientnet for post-submission
experiments.


```  
cd src/nfn_trainer
python run_train_nfn_covnext_small.py
python run_train_nfn_effnet_b5.py

output model:
- <RESULT_DIR>/one-stage-nfn-fixed/convnext_small-decoder3d-01/
- <RESULT_DIR>/one-stage-nfn-fixed/effnet_b5-decoder3d-01/
```


- Lastly, if you want to ensemble and perform local validation, run the script:

```
cd src/nfn_trainer_bugged
python run_ensemble_and_local_validation.py

cd src/nfn_trainer
python run_ensemble_and_local_validation.py
```  

### SCS (spinal canal stenosis) models

- Optional Models: These are not used in submission as one-stage SCS models did not improve the public lb score when added to the team solution. 
```  
cd src/scs_trainer
python run_train_scs_pvtv2_b4_fixed.py
python run_train_scs_covnext_base.py
python run_train_scs_effnet_b3.py

output model:
- <RESULT_DIR>/one-stage-scs/pvt_v2_b4-decoder2d-01/
- <RESULT_DIR>/one-stage-scs/convnext_base-decoder2d-01/
- <RESULT_DIR>/one-stage-scs/effnet_b4-decoder2d-01/
```
- Similarly, if you want to ensemble and perform local validation, run the script:

```
cd src/scs_trainer
python run_ensemble_and_local_validation.py
```  



## 5. Submission csv 
Team submission notebook can be found at:  
https://www.kaggle.com/code/hengck23/lhw-v24-ensemble-add-heng
![Selection_506](https://github.com/user-attachments/assets/97cc87fa-5e4c-4897-8041-c651adea4eb0)

Team post-submission notebook can be found at:  
https://www.kaggle.com/code/hengck23/post-lhw-v24-ensemble-add-heng
![Selection_507](https://github.com/user-attachments/assets/223b40f2-11e9-4321-b231-53cb2a21ce99)

## 6. Demo
heng's part:
https://www.kaggle.com/code/hengck23/clean-final-submit02-scs-nfn-ensemble


## 7. Reference trained models and validation results
- Reference results can also be found in the shared google drive at :  
  <google-drive>/kaggle-submit-rsna-2024-spine/result  
  https://drive.google.com/drive/folders/1jPPxAP6DHGQMHJPUGjPO7_Q5Asrj_LL3?usp=sharing

- It includes the weight files and train/validation logs.
  

## Authors

- https://www.kaggle.com/hengck23

## License

- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement

"We extend our thanks to HP for providing the Z8 Fury-G5 Data Science Workstation, which empowered our deep learning experiments. The high computational power and large GPU memory enabled us to design our models swiftly."
