"""
Calculating mutual information between crime and street lights in DTLA 
Author : Omkar Damle
Date : 21st June 2017
"""

import pandas as pd
import math
import numpy 
import matplotlib.pyplot as plt
import sys
from astral import Astral
import pytz

type = sys.argv[1]
print("did you select 'lights'?")



#streetData = pd.read_csv('LA_STLIGHT.csv')
dtLights = numpy.load('downTownLights.npy')

#dtLights = numpy.load('downTownMeters.npy')

#dtLights = numpy.load('60AtmLocationsDTLA.npy')

#Downtown LA
# Lower Left -> 34.038811, -118.273534
# Upper Right -> 34.053781, -118.237727

#horizontal length = 3.27km
#vertical length = 1.74km

ll_lat = 34.038811
ll_lon = -118.273534
ur_lat = 34.053781
ur_lon = -118.237727

count = 0
downTownLights = []

rows = 10
cols = 10

lightFreqMat = [[0 for j in range(cols)] for i in range(rows)]

height = math.fabs(ur_lat - ll_lat)/rows
width = math.fabs(ur_lon - ll_lon)/cols


#fig, ax = plt.subplots()

#lon1 =[]
#lat1 =[]
x = 0
count1 = 0
for lightLoc in dtLights:
	lon,lat = lightLoc
	row_index = int(math.floor((ur_lat - lat)/height))
	col_index = int(math.floor((lon - ll_lon)/width))

	if row_index < 0 or row_index >= rows:
		continue

	if col_index < 0 or col_index >= cols:
		continue
		

	#lon1[x] = lon
	#lat1[x] = lat
	lightFreqMat[row_index][col_index] += 1
	count1+=1

print count1
#ax.scatter(lon, lat)

#plot_url = py.plot_mpl(fig, filename="mpl-scatter")

freqValues = []

for row in range(rows):
	for col in range(cols):
		freqValues.append(lightFreqMat[row][col])

freqValues.sort()

oneThird = freqValues[33]
twoThird = freqValues[66]



#0-32, 33-65, 66-99

print freqValues


crimeMat = numpy.load('matrix_10x10_LAPD_area_lightCrime.npy')
crimeFreqMat = [[0 for j in range(cols)] for i in range(rows)]
crimeFreqValues = []

for row in range(rows):
	for col in range(cols):
		myList = crimeMat[row][col]
		
		if type == 'lights':
			city_name = 'Los Angeles'
			
			a = Astral()
			a.solar_depression = 'civil'
			city = a[city_name]
			nightCrimeCount = 0
			for crimeTime in myList:
				sun = city.sun(date=crimeTime.date(), local=True)
				
				tz = sun['dawn'].tzinfo
				crimeTimeAware = tz.localize(crimeTime)

				if (sun['dawn'] < crimeTimeAware < sun['dusk']):
					#print 'pass'
					pass
				else:
					nightCrimeCount += 1
					#print crimeTime
			crimeFreqMat[row][col] = nightCrimeCount
			crimeFreqValues.append(nightCrimeCount)
		else:	
			crimeFreqMat[row][col] = len(myList)
			crimeFreqValues.append(len(myList))

crimeFreqValues.sort()
print crimeFreqValues

oneThird1 = crimeFreqValues[33]
twoThird1 = crimeFreqValues[66]

"""
x - crime
y - street light
"""
xlevels = 3
ylevels = 3


px = [0 for i in range(xlevels)]
py = [0 for i in range(ylevels)]
pxy = [[0 for j in range(xlevels)] for i in range(ylevels)]
nxy = [[0 for j in range(xlevels)] for i in range(ylevels)]

lowLight = 0
mediumLight = 0
highLight = 0

lowCrime = 0
mediumCrime = 0
highCrime = 0

"""
specially for atms!!!!!!!!
"""
#oneThird = 1
#twoThird = 2


for row in range(rows):
	for col in range(cols):
		l = -1
		c = -1
		if lightFreqMat[row][col]<oneThird:
			lowLight+=1
			l = 0
		elif lightFreqMat[row][col]<twoThird:	
			mediumLight += 1
			l = 1
		else:
			highLight += 1
			l = 2

		if crimeFreqMat[row][col]<oneThird1:
			lowCrime+=1
			c = 0
		elif crimeFreqMat[row][col]<twoThird1:	
			mediumCrime += 1
			c = 1
		else:
			highCrime += 1
			c = 2


		nxy[c][l] += 1

totalLights = lowLight + mediumLight + highLight
print totalLights


py[0] = float(lowLight)/totalLights
py[1] = float(mediumLight)/totalLights
py[2] = float(highLight)/totalLights


totalCrimes = lowCrime + mediumCrime + highCrime
print totalCrimes

px[0] = float(lowCrime)/totalCrimes
px[1] = float(mediumCrime)/totalCrimes
px[2] = float(highCrime)/totalCrimes

print py
print px

MI = 0
for i in range(xlevels):
	for j in range(ylevels):
		pxy[i][j] = float(nxy[i][j])/totalCrimes

		MI += pxy[i][j]*math.log((pxy[i][j]/(px[i]*py[j])),2)


Hx = 0
Hy = 0
for x in range(xlevels):
	Hx += -px[x]*math.log(px[x],2)

for y in range(ylevels):
	Hy += -py[y]*math.log(py[y],2)

print pxy
print MI

normalizedMI = MI/math.sqrt(Hx*Hy)
print Hx
print Hy

print normalizedMI
