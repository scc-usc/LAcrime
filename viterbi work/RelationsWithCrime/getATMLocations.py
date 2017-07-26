"""
Author : Omkar Damle
Date : 22nd July 2017

ATM locations
"""

import json
import requests
import time
import numpy

atmLocations = []
calls = 0
pagetoken = ''

while True:
	
	url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=34.046296,-118.2556305&radius=2000&keyword=ATM&key=AIzaSyBYyx8e5Cv1MK9z8onZGrhQMRM7CX9SAIk'
	if calls != 0:
		url += '&pagetoken=' + pagetoken

	response = requests.get(url)
	json_data = json.loads(response.text)
	#print json_data

	try:
		pagetoken = json_data['next_page_token']
		print pagetoken
	except KeyError:
		pagetoken = ''	

	#print type(json_data)

	results = json_data['results']

	for each in results:
		loc = each['geometry']['location']
		lon,lat = loc['lng'], loc['lat']
		atmLocations.append((lon,lat))
	calls += 1	

	time.sleep(5)

	if pagetoken == None or pagetoken == '':
		break

print calls
print atmLocations

numpy.save('60AtmLocationsDTLA',atmLocations)