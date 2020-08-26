# Head Pose Estimation
## Environment
Python 3.6.5  
Keras 2.2.4  
Tensorflow-gpu 1.12.0

## Preperation   
### Dataset Placement  
First, make the 'dataset' folder in the same directory with this repository. And replace the training and test dataset in the "dataset" directory.
Then, execute the command below.
```
python only_for_mat.py
```
To load the dataset, execute the command `generate_{axis}.py` per each axis(pitch, yaw, roll) in the folder 'data_maker' to split dataset folder.

## How to make the model　　
make the respective model based on each 3 axis. Least square method is used to estimate the amount of rotation.
### Train the model
```
python train_by_vgg.py --num_epochs [number of epochs]\
    --direction [axis(yaw, roll, pitch)]\
    --output_folder [Numbers are preferred.]
```
### Evaluate with the test image
*Preparation*  
```
python prepare_for_mls.py --direction [axis(yaw, roll, pitch) --output_folder [the name of the folder which you save]
```
*Evaluate the model*  
```
python test_mode.py --direction [axis(yaw, roll, pitch)]  --output_folder [the name of the folder which you save]
```
