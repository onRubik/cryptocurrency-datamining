from controller import controller


class mainModel:
    def runModel(self):
        symbol = 'BTCBUSD'
        interval = '4h'
        lookback_val = '4'
        lookback_frame = 'years'
        lookback_utc_adjustment = 'UTC-7'
        lookback_string = lookback_val + ' ' + lookback_frame + ' ago ' + lookback_utc_adjustment

        newController = controller(symbol, interval, lookback_string)
        newController.sqlUpdate()
        newController.lstmModel()


if __name__ == "__main__":
    newMainModel = mainModel()
    newMainModel.runModel()