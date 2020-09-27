
try:
    import unzip_requirements
except ImportError:
    pass
print('here')

import copy
import numpy as np
import re
import os
import io
import boto3
import json
import base64
import onnxruntime
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from requests_toolbelt.multipart import decoder

print('then here')
from PIL import Image, ImageDraw

print('Import END...')

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

datname = r'cars.quantized.onnx'

print("Loading Model")
ort_session = onnxruntime.InferenceSession(datname)
print("Model Loaded...")

def transform_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        np_image = copy.deepcopy(image)
        np_image = np_image.resize((128,128))
        np_image = np.asarray(np_image)
        np_image = np.expand_dims(np_image, axis = 0)
        np_image = np_image.transpose(0, 3, 1, 2)
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(np_image.shape).astype('float32')
        for i in range(np_image.shape[0]):
            for j in range(np_image.shape[1]):
                norm_img_data[i,j,:,:] = (np_image[i,j,:,:]/255 - mean_vec[j]) / stddev_vec[j]

        return norm_img_data

    except Exception as e:
        print( repr(e))
        raise(e)

def get_prediction(image_bytes):
    z = transform_image(image_bytes=image_bytes)

    ort_inputs = {ort_session.get_inputs()[0].name: z}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_out1 = ort_outs[0]
    img = np.transpose(ort_out1,(0,2,3,1))
    
    plt.figure(figsize = (6,6))
    gs1 = gridspec.GridSpec(6,6)
    gs1.update(wspace=0.0, hspace=0.0)
    for i in range(1,2):
        plt.subplot(1,1,i)
        plt.axis('off')
        plt.imshow((img[0]), 'gray')
    
    
    in_mem_file = io.BytesIO()
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    pil_img = Image.open(buf)
    pil_img = pil_img.convert('RGB')
    pil_img.save(in_mem_file, format='jpeg')
    in_mem_file.seek(0)
    print("putting to s3 bucket")
    s3.Object('eva4p2-session1', 'cars.jpg').put(Body=in_mem_file,ContentType='image/JPG', ACL='public-read')
    url = "https://{}.s3.amazonaws.com/{}".format('eva4p2-session1', 'cars.jpg')

    ##url = 'https://gdeotale-session6-cars.s3.ap-south-1.amazonaws.com/cars.jpg'
    print(url)
    buf.close()
    return url


def get_cars(event, context):
    try:
        content_type_header = event['headers']['content-type']
        #print(event['body'])
        body = base64.b64decode(event["body"])
        print('BODY LOADED')
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        print("Picture loaded") 
        prediction = get_prediction(image_bytes=picture.content)
        print("Prediction done")
        
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type':'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'file': "Predicted Cars", 'image URL':prediction})
        }

    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type':'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials':True
            },
            "body": json.dumps({"error": repr(e)})
        }

