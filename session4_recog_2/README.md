# Face Recognition - 2 (Custom Image dataset + LFW dataset) 

## Assignment

1.  Refered to this [blog (Links to an external site.)](https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79).
2.  Collect 10 facial images of 10 people you know (movie stars). The more the images you collect, the better your experience would be. Add it to this  [LFW (Links to an external site.)](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz)  dataset.
3.  Train as in the blog and upload the FR model to Lambda
4.  Share the link to "that" single page.
5.  Share the link to GitHub repo.

## Datasets and Training

`160x160` Face Aligned dataset:
[https://drive.google.com/file/d/1ECSIlNXxrQx4XE1Qz8mPezuSZEBFmw65/view?usp=sharing](https://drive.google.com/file/d/1ECSIlNXxrQx4XE1Qz8mPezuSZEBFmw65/view?usp=sharing)

[The above dataset was then added to LFW Dataset]

LFW-Plus-Dataset (LFW + 10 Indian People): https://drive.google.com/file/d/1nMJdJ8fFYwCQAeNyuB9SE29hrWdl_Fby/view?usp=sharing

Training: 
(Training on LFW + 10 Indian People)
[https://github.com/seepala98/EVA4-P2/blob/master/session4_recog_2/eva4p2s4_lfwplusdataset_training.ipynb](https://github.com/seepala98/EVA4-P2/blob/master/session4_recog_2/eva4p2s4_lfwplusdataset_training.ipynb)

(Training on 10 faces only)
[https://github.com/seepala98/EVA4-P2/blob/master/session4_recog_2/eva4s4_10faces.ipynb](https://github.com/seepala98/EVA4-P2/blob/master/session4_recog_2/eva4s4_10faces.ipynb)

## link to the site :

https://deeplearnjourney.s3.ap-south-1.amazonaws.com/face_align/index.html

## lambda api link : 

https://l4ylle3qv0.execute-api.ap-south-1.amazonaws.com/dev/recognize
