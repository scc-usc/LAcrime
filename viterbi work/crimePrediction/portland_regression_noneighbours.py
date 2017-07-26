"""
Ajitesh's idea
Train regression model for training examples that are formed from T weeks history + t-1 time neighbour data
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
import scipy.io as sio
import numpy as np
import pickle
import sys
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from keras.regularizers import l2
from sklearn.cross_validation import KFold

#Create data points by using a sliding window shifting across a timeseries
def create_dataset(dataset, dataset_norm, timeslots, monthVec, look_back=1,  flag=0):
	dataX, dataY = [], []
	for i in timeslots:
		
		a = dataset_norm[i:(i+look_back)].flatten()
		checksum = dataset[i:(i+look_back)].flatten
		
		a = np.concatenate((a,monthVec[i,:]))
		
		if(np.sum(checksum) > 0 or flag == 1):
			dataX.append(a)
			dataY.append(dataset[4,i + look_back])
	return np.array(dataX), np.array(dataY)

def pad_grid(grid):
	padded_data = np.zeros((grid.shape[0]+2,grid.shape[1]+2,grid.shape[2],grid.shape[3]))
	padded_data[1:-1,1:-1,:] = grid[:,:,:,:]
	padded_data[0,1:-1,:,:] = padded_data[1,1:-1,:,:]
	padded_data[-1,1:-1,:,:] = padded_data[-2,1:-1,:,:]
	padded_data[:,0,:,:] = padded_data[:,1,:,:]
	padded_data[:,-1,:,:] = padded_data[:,-2,:,:]
	return padded_data

#Parameter for regularization
alpha = float(sys.argv[1])

week_no = 251

grid_dim_x = 15
grid_dim_y = 20
mat = sio.loadmat('data_grid_'+str(grid_dim_x)+'_'+str(grid_dim_y)+'_Downtown.mat')
cellarray = mat['data_grid_downtown'] #data

mat2 = sio.loadmat('weekly_MonthVector.mat')
monthVec = mat2['monthVec']
padded_data = pad_grid(cellarray[:,:,:,:week_no])

run_no = 1
filename = 'regression_Downtown_avgneighbours'



predict_week_no = 12
predicted_grid = np.zeros((grid_dim_x,grid_dim_y,week_no,predict_week_no))
		
best_error = np.inf
best_param = []
for look_back in [1]:#range(1,5):
	timesteps = look_back*5
		
	#Generate test set days
	indices = np.arange(week_no-timesteps)
	np.random.shuffle(indices)
	kf = KFold(len(indices), 5, shuffle=False)
	for train_index, test_index in kf:
		
		train_weeks = indices[train_index]
		test_weeks = indices[test_index]
		data = np.sum(cellarray[:,:, :, np.add(train_weeks,timesteps)],axis=2)
		data = np.reshape(data,(-1))
		# scaler = StandardScaler().fit(data.T)
		
		batch_size = 100
		data_dim = 9
		
		x_train = []
		y_train = []
		x_test = np.zeros((81, timesteps, data_dim))
		y_test = []
		initialized = 0
		traininitialized = 0
		testinitialized = 0
		
		#For each grid
		for _x in range(grid_dim_x):
			for _y in range(grid_dim_y):
				x = _x+1
				y = _y+1
				# data = np.sum(padded_data[x-1:x+2,y-1:y+2, :, :week_no],axis=2)
				# 											 
				# data = np.reshape(data,(9,-1))
				dat = data[x,y,:,:week_no].astype('float32')
				
				
				#For random train/test weeks subset
				x_train_temp, y_train_temp = create_dataset(dat,dat,train_weeks, monthVec, timesteps)				
				
				if(x_train_temp.shape[0] == 0):
					continue
				
				
				#Perform Regression
				clf = Ridge(alpha=alpha)
				clf.fit(x_train_temp, y_train_temp)

				#Create test set
				x_test_temp, y_test_temp = create_dataset(dat,dat,test_weeks, monthVec, timesteps,flag=1) #allow zero vectors			
				#Get predictions on test set
				testPredict = clf.predict(x_test_temp)
					
				
				count = 0	
				for dat_point in testPredict:
					predicted_grid[_x,_y,test_weeks[count],0] = dat_point
					count = count+1
				
				#Predict next data point by appending the latest predicted value at the end of the vector, repeat for length of period to predict
				for _gridno in range(predict_week_no-1):
					gridno = _gridno+1
					padded_predicted_grid = pad_grid(predicted_grid[:,:,:,:])	
					for k in range(len(test_weeks)):
						#with neighbours
						x_test_temp[k,0:timesteps-1] = x_test_temp[k,1:timesteps]
						x_test_temp[k,timesteps-1] = predicted_grid[_x,_y,test_weeks[k],_gridno]
						
					testPredict = clf.predict(x_test_temp)
						
					count = 0
					for dat_point in testPredict:
						predicted_grid[_x,_y,test_weeks[count],gridno] = dat_point
						count = count+1

				

		
		
		pickle.dump(predicted_grid, open('results/'+filename+'_run'+str(run_no)+'_alpha'+str(alpha)+'_timestep'+str(timesteps)+'_predictions.pkl','w'))
		
		
		#Compute training and testing error
		trainerror = sum(np.power(np.reshape(trainPredict,(-1))-y_train,2))/float(len(y_train))
		testerror = sum(np.power(np.reshape(testPredict,(-1))-y_test,2))/float(len(y_test))
		print trainerror, testerror
		

