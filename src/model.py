from controller import controller


class mainModel:
    def runModel(self):
        symbol = 'BTCBUSD'
        interval = '4h'
        lookback_val = '1'
        lookback_frame = 'years'
        lookback_utc_adjustment = 'UTC-7'
        lookback_string = lookback_val + ' ' + lookback_frame + ' ago ' + lookback_utc_adjustment
        image_output_name = 'image.png'
        flat_file_name = 'ff_30_days.csv'
        predict_days = 30

        newController = controller(symbol, interval, lookback_string, image_output_name, predict_days, flat_file_name)
        newController.sqlUpdate()
        newController.lstmModel()  


if __name__ == "__main__":
    newMainModel = mainModel()
    newMainModel.runModel()