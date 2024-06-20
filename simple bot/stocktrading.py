
import yfinance as yf
import pandas as pd
import statistics
import matplotlib.pyplot as plt
lamslow = .6
lamfast = .4
msft = yf.Ticker("TSLA")
hist = msft.history(period="2y")
print (hist)
prices = hist['Close'].tolist()
days=[]
for i in range(len(prices)):
	days.append(i)
#plt.plot(days, prices)
emaslow=[prices[0]]
for i in range(1,len(prices)):
	emaslow.append(prices[i]*(1-lamslow)+lamslow*emaslow[i-1])
emafast=[prices[0]]
for i in range(1,len(prices)):
	emafast.append(prices[i]*(1-lamfast)+lamfast*emafast[i-1])
#plt.plot(days,emaslow)
#plt.plot(days,emafast)

position=[]
tradesize = 100
averageprice=[]
pnl=[]
closedpnl=[]
difference=[]
for i in range(len(emaslow)):
	difference.append(emaslow[i]-emafast[i])
#plt.plot(days,difference)
tradethreshold = 1
for i in range(len(emaslow)):
	if difference[i]>tradethreshold:
		if (position[i-1]+tradesize)==0:
			averageprice.append(0)
			closedpnl.append(pnl[i-1]+closedpnl[i-1])
		else:
			#buy
			averageprice.append((averageprice[i-1]*position[i-1]+prices[i]*tradesize)/(position[i-1]+tradesize))
			if len(closedpnl)==0:
				closedpnl.append(0)
			else:
				closedpnl.append(closedpnl[i-1])

		position.append(position[i-1]+tradesize)
	elif difference[i]<-tradethreshold:
		if (position[i-1]-tradesize)==0:
			averageprice.append(0)
			closedpnl.append(pnl[i-1]+closedpnl[i-1])
		else:
			#sell
			averageprice.append((averageprice[i-1]*position[i-1]-prices[i]*tradesize)/(position[i-1]-tradesize))
			if len(closedpnl)==0:
				closedpnl.append(0)
			else:
				closedpnl.append(closedpnl[i-1])
		position.append(position[i-1]-tradesize)
	else:
		if len(position)==0:
			position.append(0)
			averageprice.append(0)
		else:
			position.append(position[i-1])
			averageprice.append(averageprice[i-1])
		if len(closedpnl)==0:
			closedpnl.append(0)
		else:
			closedpnl.append(closedpnl[i-1])
	pnl.append(position[i]*(prices[i]-averageprice[i]))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax3 = ax1.twinx()
totalpnl=[]
for i in range(len(pnl)):
	totalpnl.append(pnl[i]+closedpnl[i])

ax1.plot(days,totalpnl, color='tab:red')
ax3.plot(days,position,color='tab:blue')
ax2.plot(days,prices,color='tab:green')
# print (prices)
# dailyreturns=[]
# for i in range(1,len(prices)):
# 	dailyreturns.append((prices[i] - prices[i-1]) / prices[i-1])
# print (dailyreturns)
# print (statistics.mean(dailyreturns))
# print (statistics.stdev(dailyreturns))

# plt.hist(dailyreturns, bins=100)  # density=False would make counts
# plt.ylabel('count')
# plt.xlabel('daily returns');
# plt.show()
plt.show()