"""
Visualizing crime data
Author : Omkar Damle
Date : 1st June 2017
reference source:https://blog.dominodatalab.com/creating-interactive-crime-maps-with-folium/

data used : LA county department data
"""

import folium
import pandas as pd
import math
import privateInfo
from pyproj import Proj, transform
import datetime

LA_COORDINATES = (34.05, -118.24)
crimedata = pd.read_csv('2016-PART_I_AND_II_CRIMES.csv')

crimedata['INCIDENT_DATE'] = pd.to_datetime(crimedata['INCIDENT_DATE'],format='%m/%d/%Y %I:%M:%S %p')
crimedata.sort_values(by='INCIDENT_DATE',ascending = True, inplace = True)
# for speed purposes
MAX_RECORDS = 14827

# create empty map zoomed in on Los Angeles
map = folium.Map(location=LA_COORDINATES, zoom_start=12)

#make a marker cluster
marker_cluster = folium.MarkerCluster().add_to(map)

"""
# add a marker for every record in the filtered data
for each in crimedata[0:MAX_RECORDS].iterrows():
    folium.Marker([each[1]['Y'],each[1]['X']]).add_to(marker_cluster)
"""

#print crimedata.info()

inProj = Proj(init = 'epsg:2229',preserve_units = True)
outProj = Proj(init = 'epsg:4326')

count = 0

for each in crimedata[4827:MAX_RECORDS].iterrows():
	t = each[1]['INCIDENT_DATE']
	#print each[1]['INCIDENT_DATE']
	count += 1
	if count == 1 or count == 9999:
		print t.date()
	if  t.date() > datetime.date(2015,12,31):
		#print count
		#break

		#print each[1]['INCIDENT_DATE']
		#x1,y1 = 6492697.18845464,1784046.04811607
		x1 = each[1]['X_COORDINATE']
		y1 = each[1]['Y_COORDINATE']
		#x1,y1 = 6483070.70197929,1801382.31348969
		lon,lat = transform(inProj,outProj,x1,y1)
		#print lon,lat
		
		if math.isnan(lat) or math.isnan(lon):
			pass
		else:
			folium.Marker([lat,lon]).add_to(marker_cluster)

#display(map)

map.save('LA_crime.html')
"""
# add a marker for every record in the filtered data, use a clustered view

lons=[]
lats=[]
for each in crimedata[0:MAX_RECORDS].iterrows():
	lon = each[1]['Y']
	lons.append(lon) 
	lat = each[1]['X'] 
	lats.append(lat) 
# build and display(map) 
# correct the markers adding clustered markers
locations = list(zip(lons, lats))
popups = ['{}'.format(loc) for loc in locations]
map.add_children(MarkerCluster(locations=locations, popups=popups))

#display(map)
map.create_map(path='map.html')
"""
"""
# add a marker for every record in the filtered data, use a clustered view
for each in crimedata[0:MAX_RECORDS].iterrows():
    map.simple_marker(
        location = [each[1]['Y'],each[1]['X']], 
        clustered_marker = True)
 
display(map)
"""

"""
import folium
map_osm = folium.Map(location=[45.5236, -122.6750])
map_osm.save('osm.html')
"""