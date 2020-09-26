try:
    import unzip_requirements
except ImportError:
    pass
import torch
from PIL import Image

import boto3
import os
import io
import base64
import json

import numpy as np


from requests_toolbelt.multipart import decoder
print('import ends....')

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3BUCKET' in os.environ else 'eva4p2-session1'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'red_left_700_car_gan_generator.pt'

print('Downloading model...')

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model")
        model = torch.jit.load(bytestream)
        print("Model Loaded...")

except Exception as e:
    print(repr(e))
    raise(e)        

DEVICE = "cpu"

def get_sample_image(G, n_noise=100):
    G.eval()
    z = torch.randn(1, n_noise).to(DEVICE)
    print(z.shape)
    y_hat = G(z).view(1, 3, 96, 96).permute(0, 2, 3, 1) 
    result = (y_hat.detach().cpu().numpy()+1)/2.
    return result


def generate_image(event, context):
    try:
        print('generate_image: start')
        	
        img = get_sample_image(model)
        img = img.squeeze(0)
        # convert numpy image array into PIL image
        pil_img = Image.fromarray(np.uint8(img*255))
        # Generate jpeg Byte array stream
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        base64_pil_img = base64.b64encode(byte_im)
        base64_pil_img = base64_pil_img.decode("utf-8") 
        
        print('generate_image: Returning Image.......')
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"img": base64_pil_img})
          }
    except Exception as e:
        print('generate_image',repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
