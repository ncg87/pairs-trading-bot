#relate current position to the threshold
#relate the threshold to trade size
import requests, json
import alpaca_trade_api as tradeapi
import threading
import time
import websocket
from alpaca_trade_api import StreamConn
from alpaca_trade_api.common import URL
import asyncio
import matplotlib.pyplot as plt
data_url = 'wss://data.alpaca.markets'
BASE_URL = 'https://paper-api.alpaca.markets'
HISTORICBASE_URL = 'https://data.alpaca.markets'
REALTIMEDATA_URL = 'wss://stream.data.alpaca.markets/v2/iex'
ACCOUNT_URL = '{}/v2/account'.format(BASE_URL)
ORDERS_URL = '{}/v2/orders'.format(BASE_URL)

class Stock:
	def __init__(self,slowlda,fastlda,positionlimit,minimumtradethreshold,ordersizes,thresholdsizes,symbol):
		self.slowlda = slowlda
		self.fastlda = fastlda
		self.positionlimit = positionlimit
		self.minimumtradethreshold = minimumtradethreshold
		self.ordersizes = ordersizes
		self.thresholdsizes = thresholdsizes
		self.symbol = symbol
		self.lastfivebids = []
		self.lastfiveasks = []
		self.position = 0
		self.midprices = []
		self.fastemalist = []
		self.slowemalist = []
	def addbid(self,abid):
		self.lastfivebids.append(abid)
	def addask(self,aask):
		self.lastfiveasks.append(aask)
	def getlfa(self):
		return self.lastfiveasks
	def getlfb(self):
		return self.lastfivebids
	def removefirstbidandask(self):
		self.lastfivebids.pop(0)
		self.lastfiveasks.pop(0)
	def getlastmid(self):
		return self.midprices[-1]

HEADERS = {"APCA-API-KEY-ID": 'PKDJLFNT2XL9XIAGOVAV', "APCA-API-SECRET-KEY": 'CJR4G3XkmiWyyv67SkHXbiNeB4kI6vnB67lwDHXO'}
def get_account():
	r = requests.get(ACCOUNT_URL, headers=HEADERS)
	return json.loads(r.content)
def get_pandl():
	r = requests.get(ACCOUNT_URL, headers=HEADERS)
	return json.loads(r.content)['portfolio_value']
def send_order(symbol, qty, side, type, time_in_force):
	data ={
	'symbol': symbol,
	'qty': qty,
	'side': side,
	'type': type,
	'time_in_force': time_in_force,
	}
	r = requests.post(ORDERS_URL, json=data, headers = HEADERS)
	print(r)
	return json.loads(r.content)
def gethistoricaldata(symbol, start, end, limit):
	data ={
	'start': start,
	'end': end,
	'limit': limit,
	'page_token': None,
	}
	historicaldata_url = '{url}/v2/stocks/{symbol}/trades'.format(url = HISTORICBASE_URL, symbol = symbol)
	print (historicaldata_url)
	r = requests.get(historicaldata_url, json=data, headers = HEADERS)
	return r

def getlatestonestock(symbol,requesttype):
	historicaldata_url = '{url}/v2/stocks/{symbol}/{requesttype}/latest'.format(url = HISTORICBASE_URL, symbol = symbol, requesttype = requesttype)
	r = requests.get(historicaldata_url, headers = HEADERS)
	return json.loads(r.content)

def getlatestsnapshot(symbol):
	historicaldata_url = '{url}/v2/stocks/{symbol}/snapshot'.format(url = HISTORICBASE_URL, symbol = symbol)
	r = requests.get(historicaldata_url, headers = HEADERS)
	return json.loads(r.content)
def realtimeauthenticate():
	data ={
	'action': "auth",
	'key': HEADERS['APCA-API-KEY-ID'],
	'secret': HEADERS['APCA-API-SECRET-KEY'],
	}
	r = requests.post(REALTIMEDATA_URL, json=data, headers = HEADERS)
	return r
def consumer_thread():
	try:
		loop = asyncio.get_event_loop()
		loop.set_debug(True)
	except RuntimeError:
		asyncio.set_event_loop(asyncio.new_event_loop())
	global conn
	conn = StreamConn(
    	HEADERS['APCA-API-KEY-ID'],
        HEADERS['APCA-API-SECRET-KEY'],
        base_url=URL('https://paper-api.alpaca.markets'),
        data_url=URL('https://data.alpaca.markets'),
        data_stream='alpacadatav1'
    )

	@conn.on(r'T\..+')
	async def on_quotes(conn, channel, quote):
	    print('quote', quote)

	@conn.on(r'^AM\..+$')
	async def on_minute_bars(conn, channel, bar):
	    print('bars', bar)

	@conn.on(r'Q\..+')
	async def on_quotes(conn, channel, quote, ):
	    askprice = quote.askprice
	    bidprice = quote.bidprice
	    asksize = quote.asksize
	    bidsize = quote.bidsize
	    symbol = quote.symbol
	    for stock in astocklist:
	    	if symbol == stock.symbol:
	    		stock.addbid(bidprice)
	    		stock.addask(askprice)
	    		lfb = stock.getlfb()
	    		lfa = stock.getlfa()
	    		if len(lfb) > 5:
	    			stock.removefirstbidandask()
	    		if  len(lfb) == 5:
	    			averagebid = sum(lfb) / 5
	    			averageask = sum(lfa) / 5
	    			mid_price = (averagebid + averageask) / 2
	    			stock.midprices.append(mid_price)
		    		if stock.fastemalist:
		    			f = stock.fastemalist[-1] * stock.fastlda + mid_price * (1-stock.fastlda)
		    			stock.fastemalist.append(f)
		    		else:
		    			stock.fastemalist.append(mid_price)
		    		if stock.slowemalist:
		    			f = stock.slowemalist[-1] * stock.slowlda + mid_price * (1-stock.slowlda)
		    			stock.slowemalist.append(f)
		    		else:
		    			stock.slowemalist.append(mid_price)
		    
	    #print(askprice,bidprice)
	conn.run(['alpacadatav1/Q.AAPL','alpacadatav1/Q.TSLA'])	  

if __name__ == '__main__':
	AAPLordersizes = [100,150,200,250,300,350,400,450,500,550]
	AAPLthresholdsizes = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
	TSLAordersizes = [10,15,20,25,30,35,40,45,50,55]
	TSLAthresholdsizes = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
	totalriskthreshold = 50000
	overallposition = 0
	totalvolumetraded = 0
	#make it  higher if it works well
	AAPL = Stock(slowlda = 0.95,fastlda = .2,positionlimit = 1000,minimumtradethreshold = .5,ordersizes = AAPLordersizes,thresholdsizes = AAPLthresholdsizes,symbol = 'AAPL') 
	TSLA = Stock(slowlda = 0.95,fastlda = .2,positionlimit = 50,minimumtradethreshold = 7.5,ordersizes = TSLAordersizes,thresholdsizes = TSLAthresholdsizes,symbol = 'TSLA') 
	#to add a new stock define new stock above and add tolist below and add connection alpacadatv1/Q.stockname above and below
	astocklist = [AAPL,TSLA]
	start_time = time.time()
	threading.Thread(target=consumer_thread).start()
	loop = asyncio.get_event_loop()
	time.sleep(5)
	subscriptions = [['alpacadatav1/Q.AAPL', 'alpacadatav1/Q.TSLA']]
	while 1:
		if time.time() - start_time > 900:
			print("**************************************************")
			print('total volume '+str(totalvolumetraded)+' total p and l ' +str(get_pandl()))
			print("**************************************************")
	# 		plt.plot(midprices, label="realprice")
	# 		plt.plot(fastemalist,label="fastema")
	# 		plt.plot(slowemalist,label="slowema")
	# 		plt.legend(loc='upper left')
	# 		plt.show()
		if time.time() - start_time > 20:
			for stock in astocklist:
				print('slow-fast '+str(stock.slowemalist[-1]-stock.fastemalist[-1])+" "+stock.symbol)
				print('total volume '+str(totalvolumetraded)+' total p and l ' +str(get_pandl()))
				if stock.slowemalist[-1] - stock.fastemalist[-1] > max(stock.minimumtradethreshold, stock.minimumtradethreshold * stock.position / 100) and stock.position < stock.positionlimit:
					thr = stock.slowemalist[-1] - stock.fastemalist[-1]
					index = 0
					for t in stock.thresholdsizes:
						if thr > t:
							break
						if index < len(stock.thresholdsizes)-2:
							index += 1
					tradesize = stock.ordersizes[index]
					if overallposition + tradesize * stock.getlastmid() < totalriskthreshold:
						send_order(stock.symbol, tradesize, "buy", "market", "gtc")
						stock.position += tradesize
						overallposition += tradesize*stock.getlastmid()
						totalvolumetraded += tradesize * stock.getlastmid()
						print('buy '+stock.symbol+ ' order sent')
				if stock.slowemalist[-1] - stock.fastemalist[-1] < -max(stock.minimumtradethreshold, stock.minimumtradethreshold * stock.position / -100) and stock.position > -stock.positionlimit:
					thr = -(stock.slowemalist[-1] - stock.fastemalist[-1])
					index = 0
					for t in stock.thresholdsizes:
						if thr > t:
							break
						if index < len(stock.thresholdsizes)-2:
							index += 1
					tradesize = stock.ordersizes[index]
					if overallposition - tradesize * stock.getlastmid() > -totalriskthreshold:
						send_order(stock.symbol, tradesize, "sell", "market", "gtc")
						stock.position -= tradesize
						overallposition -= tradesize * stock.getlastmid()
						totalvolumetraded += tradesize * stock.getlastmid()
						print('sell '+stock.symbol+ ' order sent')
		for channels in subscriptions:
			loop.run_until_complete(conn.subscribe(channels))
			time.sleep(1)

	#ordersizes = [100,150,200,250,300,350,400,450,500,550]
	#thresholdsizes = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

