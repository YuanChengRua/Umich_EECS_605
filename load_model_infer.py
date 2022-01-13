import sys
sys.path.append("./packages")
import os
import numpy as np
from datetime import datetime
import boto3
import onnxruntime as rt
from PIL import Image, ImageOps
from scipy.special import softmax


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess(image):
    image = ImageOps.invert(image)
    smaller_image = image.resize((28, 28), Image.BICUBIC)
    numpy_smaller_image = np.asarray(smaller_image)
    numpy_smaller_image = numpy_smaller_image.astype("float32") / 255
    processed_image = np.reshape(numpy_smaller_image, (1,1,28,28))
    return processed_image


def makeInference(sess, input_img):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_onx = sess.run([output_name], {input_name: input_img})[0]
    scores = softmax(pred_onx)
    return scores


def lambda_handler(event, context):

    s3_bucket_name = "<your-unique-s3-bucket-name>"
    lambda_tmp_directory = "/tmp"
    model_file_name = "model.onnx"
    input_file_name = "digit.png"
    output_file_name = "results.txt"

    # Making probability print-out look pretty.
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    try:
        # Download test image and model from S3.
        client = boto3.client('s3')
        client.download_file(s3_bucket_name, input_file_name, os.path.join(lambda_tmp_directory, input_file_name))
        client.download_file(s3_bucket_name, model_file_name, os.path.join(lambda_tmp_directory, model_file_name))
    except:
        pass

    # Import input image in grayscale and preprocess it.
    image = Image.open(os.path.join(lambda_tmp_directory, input_file_name)).convert("L")
    processed_image = preprocess(image)
    
    # Make inference using the ONNX model.
    sess = rt.InferenceSession(os.path.join(lambda_tmp_directory, model_file_name))
    inferences = makeInference(sess, processed_image)
    
    # Output probabilities in an output file.
    f = open(os.path.join(lambda_tmp_directory, output_file_name), "w+")
    f.write("Predicted: %d \n" % np.argmax(inferences))
    for i in range(10):
        f.write("class=%s ; probability=%f \n" % (i, inferences[0][i]))
    f.close()

    # Get today's date and append to the filename.
    current_date_time = str(datetime.now())

    try:
        # Upload the output file to the S3 bucket.
        client.upload_file(os.path.join(lambda_tmp_directory, output_file_name), s3_bucket_name, output_file_name)
    except:
        pass

# Uncomment to run locally.
# lambda_handler(None, None)