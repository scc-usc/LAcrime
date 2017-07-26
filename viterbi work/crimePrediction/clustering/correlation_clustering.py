"""
Author : Omkar Damle
Date : 28th June 2017

This code prepares a correlation cell matrix.
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import scipy.stats as sts

from pylab import *
from scipy import integrate
#import pandas as pd

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


"""
data = [[2,3,4],[4,6,8],[6,9,12]]
score = [9,18,27]

lr.fit(data,score)

print lr.coef_
"""

row = int(sys.argv[1])
col = int(sys.argv[2])
time_period = int(sys.argv[3])
lookback = int(sys.argv[4])

matrix = np.load('matrix_' + str(time_period) + 'h_' + str(row) + 'x' + str(col)+ '_DTLA.npy')
#print matrix[0][0]

#row = len(matrix)
#col = len(matrix[0])

correlated_cell_matrix = [[[] for x in range(col)] for y in range(row)]


"""
Lets find the cells with highest correlation
"""
for r in range(row):
	for c in range(col):
		x1 = matrix[r][c]


		for r1 in range(row):
			for c1 in range(col):
				if r1==r and c1==c:
					continue

				#pair selected, find the correlation

				x2 = matrix[r1][c1]
				
				pcc, pVal = sts.pearsonr(x1[1:],x2[:-1])	
				#print pcc,pVal
				#raw_input('')
				if pcc > 0.4:
					correlated_cell_matrix[r][c].append((r1,c1))

np.save('correlation_cell_matrix' + str(row) + 'x' + str(col) + '_' +str(time_period) + '_DTLA' , correlated_cell_matrix)


for i in range(row):
	for j in range(col):
		print i,j
		if len(correlated_cell_matrix[i][j])!=0:
			print correlated_cell_matrix[i][j]
		print(':::,')	
	print('')			


print correlated_cell_matrix
#print(len(matrix[2][9][:-1]))
#plt.scatter(matrix[2][9][:-1],matrix[0][0][1:])
#plt.show()


#x = matrix[7][2][:-1]
#y = matrix[0][0][1:]
"""

nbins = 20
fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)

axes[0, 0].set_title('Scatterplot')
axes[0, 0].plot(x, y, 'ko')

axes[0, 1].set_title('Hexbin plot')
axes[0, 1].hexbin(x, y, gridsize=nbins)

axes[1, 0].set_title('2D Histogram')
axes[1, 0].hist2d(x, y, bins=nbins)


# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
k = kde.gaussian_kde(data.T)
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

axes[1, 1].set_title('Gaussian KDE')
axes[1, 1].pcolormesh(xi, yi, zi.reshape(xi.shape))


fig.tight_layout()
plt.show()
"""



