try:
    import unzip_requirements
except ImportError:
    pass
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import boto3
import os
import tarfile
import io
import base64
import json

from requests_toolbelt.multipart import decoder
from classes import class_names

print('Import End....')

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'eva4p2-session1'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'lfw_plus_model_traced.pt'

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Main: Creating Bytestream...")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Main: Loading Model...")
        model = torch.jit.load(bytestream)
        print("Main: Model Loaded...")
except Exception as e:
    print('Main:', repr(e))
    raise(e)


def transform_image(image_bytes):
    try:
        print('transform_image: start')
        transformations = transforms.Compose([
            transforms.Resize(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        print('transform_image: Image opened')
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print('transform_image: start',repr(e))
        raise(e)


def get_prediction(image_bytes):
    print('get_prediction: start')
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()


def recognize_image(event, context):
    try:
        print('recognize_image: start')
        content_type_header = event['headers']['content-type']
        #print(event['body'])
        print('recognize_image: content_type_header',content_type_header)
        body = base64.b64decode(event["body"])
        print('recognize_image: Body loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        # decode base64 to bytes
        #image_bytes = base64.b64decode(picture.content)
        #prediction = get_prediction(image_bytes=image_bytes)
        prediction = get_prediction(image_bytes=picture.content)
        
        
        #with open('lfw_plus_politicians_classnames.json') as f:
	#        class_idx = json.load(f)
	
	    #print(prediction)
        className = class_names[prediction]
        print('Id',prediction,' Class:',className)

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'file': filename.replace('"', ''), 'predictedId': prediction,'predicted':className })
        }
    except Exception as e:
        print('recognize_image',repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
