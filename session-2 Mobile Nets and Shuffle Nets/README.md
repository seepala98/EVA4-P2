
# Training MobileNet V2 against custom dataset of 4 flying objects as classes and then Deploying on AWS


## REST api :

https://xvosj2koa5.execute-api.ap-south-1.amazonaws.com/dev/classify

## Approach 

## Data Preparation :

* here is the raw data that was collected by crowed source Drivelink : https://drive.google.com/drive/folders/1hVfwSoRQmzStNdAl36VUOrqrNXwWkaGm
* performed hashing across the folders to remove duplicate images using hash method which creates the hash for a perticular image and store it in a hash table [https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/hash.py]

* colab notebook where preprocessing is performed : https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/pre_processing_e4p2s2%20(1).ipynb 

* post deletion of duplicate images left with these 19,127 images with following counts : 

  |  objects | count  |
  |---|---|
  | Flying birds  | 7816  |
  |  small quandcopters | 3294  |
  | large quandcopters  | 3948  |
  |winged drones | 4069 |

* The images are catagorized and saved in their respective folders and zipped . here is the location of the drive link that contains the pre-processed data 
https://drive.google.com/file/d/1-B-0TgiOF0lES0iHY94yfcgEUylCHnS_/view?usp=sharing

## Model training

- Images from dataset_final.zip was extracted and split into training and testing datasets based on 70:30 ratio. 
- Following transforms were applied to these images so that more variety of images will be available during training apart from the original ones.
   1) Rotate (-30,30)
   2) Horizontal Flip
   3) Resize (224x224)
   4) RGB Shift
   5) Normalize (Used channel mean & std-dev of imagenet dataset)
   6) Cutout ( num_holes = 1 )
   ![data sample](https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/transformed_datasamples.png)
- Then pretrained MobileNet-V2 model is downloaded. 
- Except layers 16, 17, 18 and final classifier layer, all remaining layers were frozen so that they will continue to use the pre-trained weights. 
- As Mobilenet-V2 was pre-trained using imagenet dataset it didn't know about the classes so final layers were unfrozen by setting requires_grad as True 
- This way model was given chance to learn these classes during training. 
- Also classifier layer was modified to take 4 classes instead of 1000 imagenet classes. Code base listed below.
https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/eva4p2s4_final.ipynb

<!-- RESIZE STRATEGY -->
## Resize Strategy
- images when looked at post processing shows they all are of various resolution so need to resize the images to desired shape that imageNetv2 can handle
- Hence, images were resized to (3, 224, 224) during pre-processing as imagenet uses this 
- cv2.imread(p) then cv2.resize(image, (224, 224))

<!-- MODEL TRAINED -->
## Model Training
- Torch version was downgraded to torch==1.5.1+cu92 and torchvision==0.6.1+cu92 cause of space constraints while uploading to lambda and throws a segmentation fault
- Mobilenet-V2 with 3,504,872 parameters was used for training. 
- Pytorch model has 18 convolution layers. 
- Also final linear classifier layer was customized to accept 4 custom classes (flying objects) instead of 1000 image-net classes. 
- Model was trained for 10 epochs and achieved a test accuracy of 78.9%
- Optimizer used was SGD with learning_rate = 0.03, momentum = 0.9. L2_factor of 0.0001 were used inside train_loss function.
- Model was saved post training on gpu as usual and to get the cpu used model.to('cpu') -> model.eval() -> torch.jit.trace -> torch.save

<!-- MISCLASSIFIED IMAGES -->
## Misclassified Images :

* flying bird
![flying_bird Misclassified Images](https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/flying_misclassified.png)

* large drone 
![large drone Misclassified Images](https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/large_misclassified.png)

* winged drone
![winged drone misclassified Images](https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/small_drone_missclassified.png)

* small drone
![small drone misclassified Images](https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/small_drone_missclassified.png)

## Result : 

* deployed the model to lambda function 
![deployed](https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/deployed.png)

* here are the [handler.py](https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/handler.py), [serverless.yaml](https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/serverless.yaml), [requirements.txt](https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/requirements.txt), [flying_classes.json](https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/flying_classes.json), [packages.json](https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/package.json)

* flying_class.json : 
```
{"0": ["Flying_Birds"], "1": ["Large_Quadcopter"], "2": ["Small_Quadcopter"], "3": ["Winged_Drone"]}
```
## flying bird :

<img src="https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/flying_bird.jpeg" width="425"/><img src="https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/flying_bird_0_insomia.png" width="1000"/>

## large drone  : 

<img src="https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/large_drone.jpeg" width="425"/><img src="https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/large_drone_1_insomia.png" width="1000"/>

## winged drone : 

<img src="https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/winged_drone.jpeg" width="425"/><img src="https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/winged_drone_3_insomia.png" width="1000"/>

## small drone:

<img src="https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/small_drone.jpg" width="425"/><img src="https://github.com/seepala98/EVA4-P2/blob/master/session-2%20Mobile%20Nets%20and%20Shuffle%20Nets/src/Images/small_drone_2_insomia.png" width="1000"/>
