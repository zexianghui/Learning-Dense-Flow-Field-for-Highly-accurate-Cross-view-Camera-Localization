#Learning Dense Flow Field for Highly-accurate Cross-view Camera Localization(NeurIPS 2023)

### Dataset
KITTI:
please refer to https://github.com/shiyujiao/HighlyAccurate to download the dataset.
Your dataset folder structure should be like: 

KITTI:
  raw_data:
    2011_09_26:
      2011_09_26_drive_0001_sync:
        image_00:
        image_01:
        image_02:
        image_03:
	calib_cam_to_cam.txt
    2011_09_28:
    2011_09_29:
    2011_09_30:
    2011_10_03:
  satmap:
    2011_09_26:
        2011_09_26_drive_0001_sync：
    2011_09_29:
    2011_09_30:
    2011_10_03:


Ford multi-AV:
please refer to https://github.com/shiyujiao/HighlyAccurate to download the dataset.
Your dataset folder structure should be like: 

Ford:
  2017-08-04:
    V2:
      Log1:
        2017-08-04-V2-Log1-FL:
        SatelliteMaps_18:
        grd_sat_quaternion_latlon.txt
        grd_sat_quaternion_latlon_test.txt
  2017-10-26:
  Calibration-V2:

VIGOR:
please refer to https://github.com/Jeff-Zilence/VIGOR and https://github.com/tudelft-iv/SliceMatch to download the dataset.
Your dataset folder structure should be like: 

VIGOR:
    Chicago:
        panorama:
        satellite:
    NewYork:
    SanFrancisco:
    Seattle:
    splits_corrected：

Oxford RobotCar:
For instructions on how to obtain the dataset, please visit https://github.com/tudelft-iv/CrossViewMetricLocalization.


### Codes
1. Training:

python BEV_KITTI_train.py

python BEV_Ford_train.py --train_log_start 0 --train_log_end 1 
python BEV_Ford_train.py --train_log_start 1 --train_log_end 2 

python BEV_VIGOR_train.py --area cross --rotation_range 0
python BEV_VIGOR_train.py --area cross --rotation_range 180
python BEV_VIGOR_train.py --area same --rotation_range 0
python BEV_VIGOR_train.py --area same --rotation_range 180

python BEV_oxford_train.py  


2. Evaluation:

python BEV_KITTI_test.py

python BEV_Ford_test.py -- test_log_ind 0
python BEV_Ford_test.py -- test_log_ind 1

python BEV_VIGOR_test.py --area cross --rotation_range 0
python BEV_VIGOR_test.py --area cross --rotation_range 180
python BEV_VIGOR_test.py --area same --rotation_range 0
python BEV_VIGOR_test.py --area same --rotation_range 180

python BEV_oxford_test.py 

### Files to be downloaded
1. Some files from the Oxford dataset

Some files from the Oxford dataset can be downloaded here and then placed in the dataLoader directory.
https://drive.google.com/drive/folders/1B4RAqGwECydLgj4eeAVtnOxx8ob6sNBn?usp=drive_link

2. Model files

https://drive.google.com/drive/folders/1sqIATdj5U-v21DyW31hTUpX4el76LTeF?usp=drive_link
