import alpaca_backtrader_api as Alpaca
import backtrader as bt
from datetime import datetime

alpaca_paper = {
    'api_key': 'PKZCKAV0KMBFPEYMQQ50',
    'api_secret': 'ySRx97SlZ2vkHcBG0AxJEk6vk9VipFtZPpRrbeFT',
}

ALPACA_KEY_ID = alpaca_paper['api_key']
ALPACA_SECRET_KEY = alpaca_paper['api_secret']
ALPACA_PAPER = True
fromdate = datetime(2020,8,5)
todate = datetime(2020,8,10)

tickers = ['AAPL']
timeframes = {
    '15Min':15,
    '30Min':30,
    '1H':60,
}

class Mystrategy(bt.strategy):
	def next(self):
		for i in range(0,len(self.datas)):
			print(f'{self.datas[i].datetime.datetime(ago=0)} \
        	{self.datas[i].p.dataname}: OHLC: \
              	o:{self.datas[i].open[0]} \
              	h:{self.datas[i].high[0]} \
              	l:{self.datas[i].low[0]} \
              	c:{self.datas[i].close[0]} \
              	v:{self.datas[i].volume[0]}' )

cerebro = bt.Cerebro()
cerebro.addstrategy(Mystrategy)
cerebro.broker.setcash(100000)
cerebro.broker.setcommission(0)

store = Alpaca.AlpacaStore(
    key_id=ALPACA_KEY_ID,
    secret_key=ALPACA_SECRET_KEY,
    paper=ALPACA_PAPER
)

DataFactory = store.getdata

for ticker in tickers:
    for timeframe, minutes in timeframes.items():
        print(f'Adding ticker {ticker} using {timeframe} timeframe at {minutes} minutes.')

        d = DataFactory(
            dataname=ticker,
            timeframe=bt.TimeFrame.Minutes,
            compression=minutes,
            fromdate=fromdate,
            todate=todate,
            historical=True)

        cerebro.adddata(d)

cerebro.run()
cerebro.plot(style='candlestick', barup='green', bardown='red')