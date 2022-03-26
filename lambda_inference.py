import json
import sys
sys.path.append("./packages")
import os
import base64
import pandas as pd
import numpy as np
import onnxruntime as rt
import matplotlib.pyplot as plt
# from keras.models import load_model



def preprocess(dataframe):
    x_std = (dataframe.iloc[:, 0] - dataframe.iloc[:, 0].min()) / (dataframe.iloc[:, 0].max() - dataframe.iloc[:, 0].min())
    scaled_data = x_std * (x_std.max() - x_std.min()) + x_std.min()
    return scaled_data

def inverse_transform(data_for_predict, scaled_val):
    max_val_sr = data_for_predict.max()
    max_val = max_val_sr.get(key='Adj Close')
    min_val_sr = data_for_predict.min()
    min_val = min_val_sr.get(key='Adj Close')
    og_val = (scaled_val*(max_val - min_val)) + min_val
    return og_val

def lambda_handler(event, context):
    lambda_temp_directory = "/tmp"
    model_file_name = "LSTM_price.onnx"
    model_file_name_vol = "LSTM_vol.onnx"
    input_file_name = "stock.csv"
    output_file_name = "final_res.png"
    output_rmse_file_name = "rmse_res.txt"
    model_file_path = model_file_name
    input_file_path = os.path.join(lambda_temp_directory, input_file_name)
    output_file_path = os.path.join(lambda_temp_directory, output_file_name)
    output_rmse_file_path = os.path.join(lambda_temp_directory, output_rmse_file_name)


    try:
        stockDataB64 = event["csv"]
        stockBytes = base64.b64decode(stockDataB64.encode("utf8"))
        with open(input_file_path, "wb") as f:
            f.write(stockBytes)
            f.close()
    except Exception as error:
        return {
            "statusCode": 400,
            "errorMessage": json.dumps(
                {
                    "outputResultsData": str(error)
                }
            )
        }
    try:
        image = pd.read_csv(input_file_path)
        data_for_predict = image['Adj Close']  # this is my real y

        data_for_predict = pd.DataFrame(data_for_predict)
        processed_image = preprocess(data_for_predict)

        # For demo, we only use last 10% data as test data to see the performance of the model
        train_set_idx = int(np.ceil(len(processed_image) * .90))
        test_data = processed_image[train_set_idx - 60:]

        processed_image_x = []  # we need 60 data to predict the next data and use this prediction to compare with the real y value
        for i in range(60, len(test_data)):
            processed_image_x.append(test_data[i - 60:i])

        processed_image_x = np.array(processed_image_x).astype("float32")
        processed_image_x = np.reshape(processed_image_x, (processed_image_x.shape[0], processed_image_x.shape[1], 1))

    except Exception as error:
        return {
            "statusCode": 400,
            "errorMessage": json.dumps(
                {
                    "outputResultsData": str(error)
                }
            )
        }

    try:
        session = rt.InferenceSession(model_file_path)
        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: processed_image_x})[0]
        predictions = inverse_transform(data_for_predict, predictions)
        predictions_df = pd.DataFrame(predictions, columns=['prediction'])
        temp = data_for_predict[train_set_idx:].to_numpy()
        temp = temp.reshape(-1,)
        predictions_df['real'] = temp.tolist()
        # predictions_df.to_csv(output_file_path_pred, sep='\t', encoding='utf-8', header=['prediction', 'real_data'])  # convert to txt tile and send to the next api

        plt.figure(figsize=(16, 6))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(predictions_df[['prediction', 'real_data']])
        plt.legend(['prediction', 'real'], loc='lower right')
        plt.savefig(output_file_path)
        plt.show()
        real = data_for_predict['real_data']
        real = np.array(real)
        prediction = np.array(data_for_predict['prediction'])
        rmse = np.sqrt(np.mean(((prediction - real) ** 2)))
        with open(output_rmse_file_path, "w+") as f:
            f.write("The rmse of the prediction is : %d \n" % rmse)
            f.close()

    except Exception as error:
        return {
            "statusCode": 400,
            "errorMessage": json.dumps(
                {
                    "outputResultsData": str(error)
                }
            )
        }

    try:
        with open(output_file_path, "rb") as outputFile:
            outputFileBytes = outputFile.read()
        outputFileBytesB64 = base64.b64encode(outputFileBytes).decode("utf8")

        with open(output_rmse_file_path, "rb") as outputFilermse:
            outputFIleBytesrmse = outputFilermse.read()
        outputFIleBytesB65rmse = base64.b64encode(outputFIleBytesrmse).decode("utf8")

    except Exception as error:
        return {
            "statusCode": 400,
            "errorMessage": json.dumps(
                {
                    "outputResultsData": str(error)
                }
            )
        }

    return {
        "statusCode":200,
        "body": json.dumps(
            {
            "outputResultsData": outputFileBytesB64,
            "outputResultsDatarmse": outputFIleBytesB65rmse,
            "fileType": ".png"
            }
        )
    }

