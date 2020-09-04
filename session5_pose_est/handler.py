
try:
    import unzip_requirements
except ImportError:
    pass

import PIL
from PIL import Image

import io
import base64
from requests_toolbelt.multipart import decoder
import json
import numpy as np
import onnxruntime    # Using ONNX Runtime for inferencing onnx format model

# custom packages
from utils.skeleton import get_skeleton
from utils.transforms import transform_image

print("Import End...")

HPE_QNNX_MODEL_PATH = 'trained_model/simple_pose_estimation.quantized.onnx'

onnxrt_session = onnxruntime.InferenceSession(HPE_QNNX_MODEL_PATH)
print('onnx runtime session created for the model')

# HPE Pose Prediction using ONNX Runtime
def get_poses_predictions(ort_s, img):
    try:
        tr_img = transform_image(img)
        inp = {ort_s.get_inputs()[0].name: tr_img}
        out = ort_s.run(None, inp)
        out = np.array(out[0][0])
        return out
    except Exception as e:
        print(repr(e))
        raise(e)

def img_to_base64(img):
    img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    img_str = f"data:image/jpeg;base64,{base64.b64encode(myimage).decode()}"
    return img_str

'''
Input image is received as formdata.
NOTE: In API gateway , Binay Media type shall be set to multipart/form-data
'''
def main_handler(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print('content_type header: ' + content_type_header)
        #print('Event Body: ' + event["body"])

        body = base64.b64decode(event["body"])
        print('Image content Loaded')
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        print(f'MultipartDecoder processed')
        img = Image.open(io.BytesIO(picture.content))
        print(f'Input image data loaded')

        output = get_poses_predictions(ort_s=onnxrt_session, img=img)        
        print(f'got image poses, output shape: {output.shape}')

        img = np.array(img)
        hpe_img = get_skeleton(img, output)
        print(f'Image prepared with all joints connected')

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
            "body": json.dumps({'file': filename.replace('"', ''), 'hpeImg': img_to_base64(hpe_img)})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }

'''
Input image is received as base64 encoded format in JSON body
'''
def main_handler_jsondata(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print('content_type header: ' + content_type_header)
        #print('Event Body: ' + event["body"])

        # use this when input image is received as JSON body
        json_body = json.loads(event["body"])
        im = base64.b64decode(json_body["img"])
        print('Image content Loaded')

        img = Image.open(io.BytesIO(im))

        output = get_poses_predictions(ort_s=onnxrt_session, img=img)         
        print(f'Got image poses, output shape: {output.shape}')

        img = np.array(img)
        hpe_img = get_skeleton(img, output)
        print(f'Image prepared with all joints connected')


        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'hpeImg': img_to_base64(hpe_img)})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
