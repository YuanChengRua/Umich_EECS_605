import sys
sys.path.append("./packages")
import os
import numpy as np
from datetime import datetime
import boto3
import onnxruntime as rt
from PIL import Image, ImageOps
from scipy.special import softmax


# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()
#
#
# def preprocess(image):
#     image = ImageOps.invert(image)
#     smaller_image = image.resize((28, 28), Image.BICUBIC)
#     numpy_smaller_image = np.asarray(smaller_image)
#     numpy_smaller_image = numpy_smaller_image.astype("float32") / 255
#     processed_image = np.reshape(numpy_smaller_image, (1,1,28,28))
#     return processed_image
#
#
# def makeInference(sess, input_img):
#     input_name = sess.get_inputs()[0].name
#     output_name = sess.get_outputs()[0].name
#     pred_onx = sess.r`un([output_name], {input_name: input_img})[0]
#     scores = softmax(pred_onx)
#     return scores


def preprocess(x):
    intersection = np.ones((1, 100))
    complete_x = np.vstack([intersection, x])

    return complete_x

def calculate_prob(complete_x, theta_ini_t):
    prob = np.exp(np.dot(theta_ini_t, complete_x)) / (1 + np.exp(np.dot(theta_ini_t,complete_x)))
    return prob


def lambda_handler(event, context):

    s3_bucket_name = "assignment-2-yuan-cheng-q3"
    lambda_tmp_directory = "/tmp"
    # model_file_name = "model.onnx"
    parameters_file_name = "theta_ini_t.txt"
    input_file_name = "fashion_mnist_images.txt"
    input_file_name_label = "fashion_mnist_labels.txt"
    output_file_name = "results.txt"

    # Making probability print-out look pretty.
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)


    # Download test image and model from S3.
    client = boto3.client('s3')
    client.download_file(s3_bucket_name, input_file_name, os.path.join(lambda_tmp_directory, input_file_name))
    client.download_file(s3_bucket_name, parameters_file_name, os.path.join(lambda_tmp_directory, parameters_file_name))
    client.download_file(s3_bucket_name, input_file_name_label, os.path.join(lambda_tmp_directory, input_file_name_label))
    # client.download_file(s3_bucket_name, model_file_name, os.path.join(lambda_tmp_directory, model_file_name))


    # # Import input image in grayscale and preprocess it.
    # image = Image.open(os.path.join(lambda_tmp_directory, input_file_name)).convert("L")

    x = np.loadtxt('/tmp/fashion_mnist_images.txt').reshape(784, 100)
    y = np.loadtxt('/tmp/fashion_mnist_labels.txt').reshape(1, 100)
    # train_y = y[:, 0:5000]
    # test_y = y[:, 5000:]
    complete_x = preprocess(x)
    theta_ini_t = np.loadtxt("/tmp/theta_ini_t.txt").reshape(1,785)

    p_list = []  # use the final thetas to generate the probability of each data point of the test set
    for i in range(x.shape[1]):
        prob = calculate_prob(complete_x[:, i], theta_ini_t)
        p_list.append(prob[0])


    label_list = []
    for i, ele in enumerate(p_list):
        if ele <= 0.5:
            label_list.append(-1)
        else:
            label_list.append(1)

    error_counter = 0
    for i in range(len(label_list)):
        if label_list[i] != y[0, i]:
            error_counter += 1

    error_rate = error_counter / len(label_list)

    # error_counter_test = 0
    # for i in range(len(label_list_test)):
    #     if label_list_test[i] != test_y[:, i][0]:
    #         error_counter_test += 1
    #
    # error_count_train = 0
    # for i in range(len(label_list_train)):
    #     if label_list_train[i] != train_y[:, i][0]:
    #         error_count_train += 1
    #
    # error_rate_test = str(error_counter_test / len(p_list_test))
    # error_rate_train = str(error_count_train / len(p_list_train))

    # processed_image = preprocess(image)
    #
    # # Make inference using the ONNX model.
    # sess = rt.InferenceSession(os.path.join(lambda_tmp_directory, model_file_name))
    # inferences = makeInference(sess, processed_image)


    # Output probabilities in an output file.
    f = open(os.path.join(lambda_tmp_directory, output_file_name), "w+")
    # f.write("Predicted: %d \n" % np.argmax(inferences))
    f.write("The error rate is" + "\n" + str(error_rate) + "\n")
    # for i in range(10):
    #     f.write("class=%s ; probability=%f \n" % (i, inferences[0][i]))
    f.close()

    # Get today's date and append to the filename.
    current_date_time = str(datetime.now())

    # Upload the output file to the S3 bucket.
    client.upload_file(os.path.join(lambda_tmp_directory, output_file_name), s3_bucket_name, output_file_name)


# Uncomment to run locally.
# lambda_handler(None, None)
