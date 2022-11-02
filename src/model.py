from controller import controller


class mainModel:
    def runModel(self):
        symbol = 'BTCBUSD'
        interval = '4h'
        lookback = '4 years ago UTC-7'

        newController = controller(symbol, interval, lookback)
        # out = newController.getKlines()
        # df_types = out.dtypes
        # print(df_types)
        # print('\n')
        # print(out)
        newController.sqlUpdate()


if __name__ == "__main__":
    newMainModel = mainModel()
    newMainModel.runModel()