from binance import Client
import pandas as pd
import os
# from typing import List

class controller:
    def __init__(self, symbol, interval, lookback):
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback

    
    def binanceClient(self):
        ## to set a temporal environment variable with powershell:
        ## use $Env:binance_api_key = "" $Env:binance_api_secret_key = ""
        binance_api_key = os.environ.get('binance_api_key')
        binance_api_secret_key =  os.environ.get('binance_api_secret_key')
        client = Client(binance_api_key, binance_api_secret_key)

        return client


    def getKlines(self):
        client = self.binanceClient()
        frame = pd.DataFrame(client.get_historical_klines(self.symbol, self.interval, self.lookback))
        frame = frame.iloc[:,:6]
        frame.columns = ['time','open','high','low','close','volume']
        frame = frame.set_index('time')
        frame.index = pd.to_datetime(frame.index, unit = 'ms')
        frame = frame.astype(float)

        return frame