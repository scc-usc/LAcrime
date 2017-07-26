"""
Author : Omkar Damle
Date : 13th june 2017
"""

from geopy.distance import vincenty
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from scipy import stats

#dtLights = numpy.load('downTownLights.npy')
#dtCrimes = numpy.load('downTownCrimesLocations.npy')

n=0
#print len(dtCrimes)
#print len(dtLights)

crimeDistances = numpy.load('distanceCrimeAndLightsInDTLA.npy')
#plt.hist(crimeDistances, range = (0,30),bins = 30)
#plt.show()

print stats.ttest_1samp(crimeDistances, 15)


raw_input('')

"""
distanceList = []

for each in dtCrimes:
	n += 1 

	if n%100 == 0:
		print n
	
	crimeLon,crimeLat = each	

	minDist = 10000
	for lightLoc in dtLights:
		 lightLon,lightLat = lightLoc
		 dist = vincenty((crimeLat,crimeLon),(lightLat,lightLon)).meters
		 #print dist
		 #raw_input('')
		 if dist < minDist:
		 	minDist = dist

	#print minDist
	distanceList.append(minDist)
numpy.save('distanceCrimeAndLightsInDTLA',distanceList)
"""
#no of street lights within 50m
noST50 = []
maxRecords = 300
for each in dtCrimes[0:maxRecords]:
	n += 1 

	if n%100 == 0:
		print n
	
	crimeLon,crimeLat = each	

	count = 0
	for lightLoc in dtLights:
		 lightLon,lightLat = lightLoc
		 dist = vincenty((crimeLat,crimeLon),(lightLat,lightLon)).meters
		 #print dist
		 #raw_input('')
		 if dist < 50:
		 	count += 1

	#print minDist
	noST50.append(count)
numpy.save('within50mCrimeAndLightsDTLA',noST50)
plt.hist(noST50,bins = 10, range = (0,30))
plt.show()

