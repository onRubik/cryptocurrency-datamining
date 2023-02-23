# import platform
from data_integrity import dataIntegrity
from pathlib import Path
import math
import numpy as np
from binance import Client
import pandas as pd
import os
import sqlite3
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class controller:
    def __init__(self, symbol: str, interval: str, lookback_string: str, image_output_name: str):
        self.symbol = symbol
        self.interval = interval
        self.lookback_string = lookback_string
        self.image_output_name = image_output_name

    
    def binanceClient(self):
        ## to set a temporal environment variable with powershell:
        ## use $Env:binance_api_key = "" $Env:binance_api_secret_key = ""
        binance_api_key = os.environ.get('binance_api_key')
        binance_api_secret_key =  os.environ.get('binance_api_secret_key')
        client = Client(binance_api_key, binance_api_secret_key)

        return client


    def getKlines(self):
        client = self.binanceClient()
        frame = pd.DataFrame(client.get_historical_klines(self.symbol, self.interval, self.lookback_string))
        frame = frame.iloc[:,:6]
        frame.columns = ['time','open','high','low','close','volume']
        frame = frame.set_index('time')
        frame.index = pd.to_datetime(frame.index, unit = 'ms')
        frame = frame.astype(float)

        return frame


    def sqlUpdate(self):
        db_path, os_type = dataIntegrity.imgFolder()
        db_path = Path(db_path)
        db_path = db_path.parent

        if os_type == 'Windows':
            db_path_fix = str(db_path) + '\\historical_klines.db'
        if os_type == 'Linux':
            db_path_fix = str(db_path) + '/historical_klines.db'

        con = sqlite3.connect(db_path_fix)
        table_name = str(self.symbol).lower() + '_' + str(self.interval).lower() + '_historical'
        print('table name = ' + table_name)

        df_insert = self.getKlines()
        df_insert.to_sql(table_name, con, if_exists='append', index_label='time')
        con.commit()
        con.close()


    def readDb(self):
        # next commented lines is an example to print an sql query if needed
        # cur = con.cursor()
        # for row in cur.execute('''
        # SELECT time, open, high, low, close, volume
        # FROM btcbusd_4h_historical
        # LIMIT 10
        # '''):
        #     print(row)

        db_path, os_type = dataIntegrity.imgFolder()
        db_path = Path(db_path)
        db_path = db_path.parent

        if os_type == 'Windows':
            db_path_fix = str(db_path) + '\\historical_klines.db'
        if os_type == 'Linux':
            db_path_fix = str(db_path) + '/historical_klines.db'

        con = sqlite3.connect(db_path_fix)
        df_read = pd.read_sql_query(
        '''
        SELECT *
        FROM (
            SELECT * FROM btcbusd_4h_historical ORDER BY time DESC LIMIT 2190
        )
        ORDER BY time ASC
        ;''', con)
        con.commit()
        con.close()

        return df_read

    
    def dataset_matrix(self, dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])

        return np.array(dataX), np.array(dataY)


    def lstmModel(self):
        img_output, os_type = dataIntegrity.imgFolder()
        img_output = Path(img_output)
        img_output = img_output.parent

        if os_type == 'Windows':
            img_output_fix = str(img_output) + '\\img_output\\' + self.image_output_name
        if os_type == 'Linux':
            img_output_fix = str(img_output) + '/img_output/' + self.image_output_name
        # considering only a year of data
        # target prediction data is the close value per date

        get_data = self.readDb()
        close_values = get_data[['time', 'close']]
        close_values_copy = close_values.copy()
        
        # normilizing the dataframe
        del close_values['time']
        scaler = MinMaxScaler(feature_range=(0,1))
        close_values = scaler.fit_transform(np.array(close_values).reshape(-1, 1))
        
        # seting the percentage data sets for training 60% and testing 40%
        training_size = int(len(close_values)* 0.60)
        # test_size = len(close_values) - training_size
        training_data = close_values[0:training_size,:]
        test_data = close_values[training_size:len(close_values),:1]
        print('train data = ', training_data.shape)
        print('train data = ', test_data.shape)
        
        # create the dataset matrix
        time_step = 120
        X_training, y_training = self.dataset_matrix(training_data, time_step)
        X_test, y_test = self.dataset_matrix(test_data, time_step)
        print('X_training = ', X_training.shape)
        print('y_training = ', y_training.shape)
        print('X_test = ', X_test.shape)
        print('y_test = ', y_test.shape)
        
        # building the LSTM shape
        X_training =X_training.reshape(X_training.shape[0],X_training.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
        print('X_training = ', X_training.shape)
        print('X_test = ', X_test.shape)        
        
        # building the LSTM model
        model=Sequential()
        model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam")
        
        # next line can be used to plot the training and validation loss
        history = model.fit(X_training, y_training, validation_data=(X_test, y_test), epochs=200, batch_size=32, verbose=1)

        # executing the prediction
        training_predict = model.predict(X_training)
        test_predict = model.predict(X_test)
        print(training_predict.shape)
        print(test_predict.shape)

        # model evaluation:
        # data form rollback
        training_predict = scaler.inverse_transform(training_predict)
        test_predict = scaler.inverse_transform(test_predict)
        original_ytraining = scaler.inverse_transform(y_training.reshape(-1,1)) 
        original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))

        # result of RMSE, MSE and MAE metrics
        print('Training data RMSE: ', math.sqrt(mean_squared_error(original_ytraining,training_predict)))
        print('Training data MSE: ', mean_squared_error(original_ytraining,training_predict))
        print('Training data MAE: ', mean_absolute_error(original_ytraining,training_predict))
        print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
        print("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
        print("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))

        # variance regression score
        print("Train data explained variance regression score:", explained_variance_score(original_ytraining, training_predict))
        print("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))

        # R square score for regression
        print("Train data R2 score:", r2_score(original_ytraining, training_predict))
        print("Test data R2 score:", r2_score(original_ytest, test_predict))

        # regression Loss Mean Gamma deviance regression loss (MGD) and Mean Poisson deviance regression loss (MPD)
        print("Train data MGD: ", mean_gamma_deviance(original_ytraining, training_predict))
        print("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
        print("Train data MPD: ", mean_poisson_deviance(original_ytraining, training_predict))
        print("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))

        
        # print(training_predict)
        # print(test_predict)
        # graph between original values VS predicted
        look_back=time_step
        trainPredictPlot = np.empty_like(close_values)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(training_predict) + look_back, :] = training_predict
        print("Training predicted data: ", trainPredictPlot.shape)

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(close_values)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(training_predict) + (look_back*2) + 1 : len(close_values)-1, :] = test_predict
        print("Test predicted data: ", testPredictPlot.shape)

        names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

        plotdf = pd.DataFrame({'time': close_values_copy['time'],
            'original_close': close_values_copy['close'],
            'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
            'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist()})

        fig = px.line(plotdf, x=plotdf['time'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
            plotdf['test_predicted_close']],
            labels={'value':'close values','time': 'time'})
        fig.update_layout(title_text='Comparision between original close price vs predicted close price',
            plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
        fig.for_each_trace(lambda t:  t.update(name = next(names)))

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        # fig.show()
        fig.write_image(img_output_fix)