import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from sklearn.metrics import mean_squared_error
np.random.seed(7)




df=pd.read_csv('sales data-set.csv', index_col=[2])
#df=pd.DataFrame(df)
#print(df)


#df1=df.loc[df['Dept']==10]
#df1=df1.loc[df1['Store']<=3]
#print(df1)
revenue={2:0.25 ,4:0.15,8:0.15,10:0.17,9:0.13,7:0.19,5:0.14,1:0.14,3:0.11,6:0.13}

def create_dataset(dataset, look_back,dept):
	dataX, dataY = [],[]
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def model1(df1,store,dept,look_back=4):
	# print(np.sum(df1.loc[df1['Store']==1]['Weekly_Sales']))

	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset= scaler.fit_transform(df1['Weekly_Sales'].values.reshape(-1,1))
	#print(df1)
	train_size = int(len(dataset) * 0.8)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

	X_train, Y_train = create_dataset(train, look_back,dept)
	X_test, Y_test = create_dataset(test, look_back, dept)

	print(X_train.shape)
	print(Y_train.shape)

	X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
	X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


	model = Sequential()
	model.add(CuDNNLSTM(4, input_shape=(1,look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(X_train, Y_train, epochs=10, batch_size=16)


	train_preds = model.predict(X_train)
	testPredict = model.predict(X_test)

	train_preds = scaler.inverse_transform(train_preds)
	Y_train = scaler.inverse_transform([Y_train])
	testPredict = scaler.inverse_transform(testPredict)
	Y_test = scaler.inverse_transform([Y_test])

	trainScore = math.sqrt(mean_squared_error(Y_train[0], train_preds[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(Y_test[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))


	train_preds_plt = np.empty_like(dataset)
	train_preds_plt[:, :] = np.nan
	train_preds_plt[look_back:len(train_preds)+look_back, :] = train_preds


	test_preds_plt = np.empty_like(dataset)
	test_preds_plt[:, :] = np.nan
	test_preds_plt[len(train_preds)+(look_back*2)+1:len(dataset)-1, :] = testPredict

	test_plt= np.empty_like(dataset)
	test_plt[:,:]=np.nan
	test_plt[train_size:len(dataset),:] =test

	plt.plot(scaler.inverse_transform(dataset[0:train_size,:])*revenue[dept],label='Training_data')
	holidays=df1['IsHoliday']
	list1=[]
	for ind,i in enumerate(holidays):
		if i==False:
			list1.append(1)
		else:
			list1.append(0)
	#print(list1)
	list1=np.asarray(list1,dtype=np.float32)
	for ind,i in enumerate(list1):
		if i==1:
			list1[ind]=np.nan
	#print(list1)

	#print(train_preds_plt[list1])


	#plt.plot(scaler.inverse_transform(test_plt),color='red')
	plt.plot(train_preds_plt*revenue[dept],label='Train_predictions')
	plt.plot(test_preds_plt*revenue[dept],label='Future Predictions')
	#plt.plot(list1.reshape(-1,1),'ro',label="Holidays")
	#plt.show()
	plt.legend(loc='upper left')
	plt.xlabel("Weeks")
	plt.ylabel('Earnings generated')
	#plt.show()
	str1='./image_rev1/image'+str(store)+str(dept)+".png"
	plt.savefig(str1)
	plt.close()

for i in range(1,3):
	for j in range(1,11):
		df1=df.loc[df['Dept']==j]
		df1=df1.loc[df1['Store']==i]
		model1(df1,store=i,dept=j)
for i in range(1,11):
	df1=df.loc[df['Dept']==i]
	df1=df1.loc[df1['Store']<=4]
	model1(df1,store=3,dept=i)