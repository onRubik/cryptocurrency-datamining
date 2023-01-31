from sklearn.ensemble import RandomForestClassifier
from binance import Client
import pandas as pd
import os
import sqlite3
# from typing import List

class controller:
    def __init__(self, symbol: str, interval: str, lookback_string: str):
        self.symbol = symbol
        self.interval = interval
        self.lookback_string = lookback_string

    
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
        con = sqlite3.connect('H:\SQLite\SQLiteData\historical_klines.db')
        table_name = str(self.symbol).lower() + '_' + str(self.interval).lower() + '_historical'
        print('table name = ' + table_name)

        # cur = con.cursor()
        # for row in cur.execute('''
        # SELECT time, open, high, low, close, volume
        # FROM btcbusd_4h_historical
        # LIMIT 10
        # '''):
        #     print(row)

        df_insert = self.getKlines()
        df_insert.to_sql(table_name, con, if_exists='append', index_label='time')
        con.commit()

        con.close

    
    def sklearnExample(delf):

        test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

        y = train_data_no_mods["Survived"]

        features = ["Pclass", "Sex", "SibSp", "Parch"]
        X = pd.get_dummies(train_data_no_mods[features])
        X_test = pd.get_dummies(test_data[features])

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        model.fit(X, y)
        predictions = model.predict(X_test)

        output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
        output.to_csv('submission.csv', index=False)
        print('csv file saved')
