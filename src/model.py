from controller import controller


class mainModel:
    def runModel(self):
        symbol = 'BTCBUSD'
        interval = '5m'
        lookback = '30 min ago UTC-7'

        newController = controller(symbol, interval, lookback)
        out = newController.getKlines()
        print(out)


if __name__ == "__main__":
    newMainModel = mainModel()
    newMainModel.runModel()