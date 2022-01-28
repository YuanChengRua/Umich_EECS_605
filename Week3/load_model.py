import sys
sys.path.append("./packages")
import os
import numpy as np
from datetime import datetime
import boto3
import csv



def lsqfit_poly_periodic(t, y, d_poly, d_periodic, T):
    m = t.shape[0]

    # Step 1: Construct the A matrix for the polynomial part
    Apoly = np.zeros((m, d_poly + 1))
    for i in np.arange(d_poly + 1):
        Apoly[:, i] = np.power(t, i)

    # Step 2 : Construct the A matrix for the periodic part
    Aperiodic = np.zeros((m, 2 * d_periodic))

    for i in np.arange(d_periodic):
        Aperiodic[:, 2 * i] = np.cos(((2 * np.pi * (i + 1)) / T) * t)
        Aperiodic[:, (2 * i) + 1] = np.sin(((2 * np.pi * (i + 1)) / T) * t)

    # Step 3: Concatenate them
    A = np.hstack((Apoly, Aperiodic))

    # Step 4: Solve the least squares problem
    c_ls = np.matmul(np.linalg.pinv(A), y)

    return A, c_ls

def poly_periodic(t, c, d_poly, d_periodic, T):
    # complete the function including the return statement
    m = t.shape[0]
    y = np.zeros((m,))

    # Step 1: Extact polynomial and periodic coefficients
    c_poly = c[0:d_poly + 1]
    c_periodic = c[d_poly + 1:]

    # Step 2: Compute the polynomial part
    for i in np.arange(d_poly + 1):
        y += np.power(t, i) * c_poly[i]

    # Step 3: Compute the periodic part
    for i in np.arange(d_periodic):
        y += np.cos(((2 * np.pi * (i + 1)) / T) * t) * c_periodic[2 * i]  # cosine term
        y += np.sin(((2 * np.pi * (i + 1)) / T) * t) * c_periodic[(2 * i) + 1]  # sine term

    return y


def lambda_handler(event, context):

    s3_bucket_name = "assignment3-yuancheng"
    lambda_tmp_directory = "/tmp"
    # model_file_name = "parameters.txt"
    training_set_name = "training_set.csv"
    add_files = "newData.csv"  # later, the GUI could be changed one by one or together?
    testing_set_name = "testing_set.csv"
    model_file_name = "parameters.txt"
    output_file_name_pred_test = "test_mse.txt"

    # Making probability print-out look pretty.
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    '''
    The way I design this way is that whenever we put new data, we may input several csv files in AWS because in GUI, we only upload 1 file pertime. So this file is 
    to combine the original data and the additional data and we can retrain the model when we have 7, 14, 21, etc additional data and the retraining process is 
    also completed here. 
    '''

    # Download test image and model from S3.
    client = boto3.client('s3')
    client.download_file(s3_bucket_name, training_set_name, os.path.join(lambda_tmp_directory, training_set_name))
    client.download_file(s3_bucket_name, model_file_name, os.path.join(lambda_tmp_directory,model_file_name))
    client.download_file(s3_bucket_name, add_files, os.path.join(lambda_tmp_directory, add_files))
    client.download_file(s3_bucket_name, testing_set_name, os.path.join(lambda_tmp_directory, testing_set_name))

    response = client.list_objects(Bucket = s3_bucket_name)
    user_input_name = []
    user_output_name = []
    for name in response['Contents']:
        if name['Key'][:4] == "user":
            user_input_name.append(name['Key'])
            user_output_name.append("result" + name['Key'][9:-4] + ".txt")  # each input has its own output file
            client.download_file(s3_bucket_name, name['Key'], os.path.join(lambda_tmp_directory, name['Key']))

    '''
    Updating the training data set to do the retrain process is necessary
    '''
    csv_file = open('/tmp/training_set.csv', 'r')
    date = []
    sales = []
    for a, b in csv.reader(csv_file, delimiter=','):
        # Append each variable to a separate list
        date.append(a)
        sales.append(b)
    # # 70% for training 10% unused data 20 for testing
    date_int = np.array(date).astype(float)
    sales = np.array(sales).astype(float)

    # need to change the input date to date_int
    # add all additional info together
    additional_date = []
    additional_sales = []
    add_csv_file = open(os.path.join(lambda_tmp_directory, add_files), 'r')
    for a, b in csv.reader(add_csv_file, delimiter=','):
        additional_date.append(a)
        additional_sales.append(b)
        #client.delete_object(Bucket=s3_bucket_name, Key=os.path.join(lambda_tmp_directory, file)) # every time, when we upload one file, we will extract the information and after that, we can remove this file from s3

    additional_date = np.array(additional_date)
    additional_date_int = np.arange(np.max(date_int)+1, 1 + np.max(date_int) + additional_date.shape[0]).astype(float)
    additional_sales = np.array(additional_sales).astype(float)

    date_train = np.hstack((date_int, additional_date_int))
    sales_train = np.hstack((sales, additional_sales))
    total_data = np.hstack((date_train.reshape(-1, 1), sales_train.reshape(-1, 1)))
    # np.savetxt(os.path.join(lambda_tmp_directory, training_set_name), total_data, delimiter=',')  # save the updated training set in S3 and cover the previous version of training set

    f_add = open(os.path.join(lambda_tmp_directory, training_set_name), "w+")
    np.savetxt(f_add, total_data, delimiter=',')
    f_add.close()
    client.upload_file(os.path.join(lambda_tmp_directory, training_set_name), s3_bucket_name, training_set_name)

    if (total_data.shape[0] - 152) % 7 == 0:
        A, c_ls = lsqfit_poly_periodic(date_train, sales_train, 3, 4, 12)
        f = open(os.path.join(lambda_tmp_directory, model_file_name), "w+")  # every time we open the same file and update the same file.
        # The updated parameters will cover the previous parameters. So prediction file can use this parameters file directly
        for row in c_ls.reshape(-1, 1):
            np.savetxt(f, row)
        f.close()
        client.upload_file(os.path.join(lambda_tmp_directory, model_file_name), s3_bucket_name, model_file_name)


    '''
    Use the entire testing data to see our model's improved performance
    '''
    parameters = np.loadtxt("/tmp/parameters.txt")
    csv_file_test = open(os.path.join(lambda_tmp_directory, testing_set_name), 'r')
    date_int_test = []
    sales_test = []
    for a, b in csv.reader(csv_file_test, delimiter=','):
        # Append each variable to a separate list
        date_int_test.append(a)
        sales_test.append(b)
    date_int_test = np.array(date_int_test).astype(float)
    sales_test = np.array(sales_test).astype(float)
    ypred = poly_periodic(date_int_test, parameters, 3, 4, 12)

    mse_test = np.linalg.norm(sales_test - ypred) / sales_test.shape[0]
    f_user = open(os.path.join(lambda_tmp_directory, output_file_name_pred_test), "w+")
    f_user.write("The mse of the testing data is " + '\n' + str(mse_test))
    f_user.close()
    client.upload_file(os.path.join(lambda_tmp_directory, output_file_name_pred_test), s3_bucket_name, output_file_name_pred_test)
    '''
    To deal with the limit of lambda's hander, we can put prediction here so every time when we update the model, we can directly compute the error here 
    '''
    '''
    Use the current parameters to every user's input but how to output different output file? 
    '''
    parameters = np.loadtxt("/tmp/parameters.txt")
    for i, filename in enumerate(user_input_name):
        csv_file_user = open(os.path.join(lambda_tmp_directory, filename), 'r')
        date_org = []
        for a in csv.reader(csv_file_user, delimiter = ','):
            date_org.append(a)

        datetime_user = np.datetime64(date_org[0][0])
        year = datetime_user.astype(object).year
        month = datetime_user.astype(object).month
        date_int_user = float((year - 2004) * 12 + 1 + month)  # can predict any month after 2022-01
        ypred = poly_periodic(np.array(date_int_user).reshape(-1, ), parameters, 3, 4, 12)
        f_user = open(os.path.join(lambda_tmp_directory, user_output_name[i]), "w+")
        f_user.write(str(ypred))
        f_user.close()
        client.upload_file(os.path.join(lambda_tmp_directory, user_output_name[i]), s3_bucket_name, user_output_name[i])


    # Get today's date and append to the filename.
    current_date_time = str(datetime.now())

    # Upload the output file to the S3 bucket.
    # client.upload_file(os.path.join(lambda_tmp_directory, model_file_name), s3_bucket_name, model_file_name)


# Uncomment to run locally.
# lambda_handler(None, None)
