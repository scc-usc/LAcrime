"""
Visualizing tweet data
Author : Omkar Damle
Date : 2nd June 2017
reference source:https://blog.dominodatalab.com/creating-interactive-crime-maps-with-folium/

if chloropleth map required reference: http://andrewgaidus.com/leaflet_webmaps_python/
"""

import folium
import pandas as pd
import math
import string
import privateInfo
from pyproj import Proj, transform
import datetime
#abstract syntax trees - ast
import ast

LA_COORDINATES = (34.05, -118.24)
MAX_RECORDS = 10000

# create empty map zoomed in on Los Angeles
map = folium.Map(location=LA_COORDINATES, zoom_start=12)

#make a marker cluster
marker_cluster = folium.MarkerCluster().add_to(map)

tweetData = pd.read_csv('tweets_LA_geo_28may.csv')

count = 0


for each in tweetData[0:MAX_RECORDS].iterrows():
	

	#t = each[1]['Date Occurred']
	
	if count ==0 or count == MAX_RECORDS-1:
		print each[1]['created']
	count += 1
	
	locString = each[1]['coordinates']
	
	if pd.isnull(locString):
		continue

	dic = ast.literal_eval(locString)
	#print dic

	lon,lat = 	dic['coordinates'][0],dic['coordinates'][1]
	#print lon,lat
	
	if math.isnan(lat) or math.isnan(lon):
		pass
	else:
		folium.Marker([lat,lon], popup = each[1]['text'].decode('utf-8')).add_to(marker_cluster)

#display(map)


"""
We calculate the densities of crime areas
"""

#Read tracts shapefile into GeoDataFrame
#tracts = gpd.read_file('ct2010.shp')
#tracts = gpd.read_file('CENSUS_TRACTS_2010.shp')
#tracts = gpd.read_file('tl_2016_06_tract.shp')

#tracts.insert(0,'id',range(len(tracts)))

#Generate Counts of Assaults per Census Tract
#Spatially join census tracts to assaults (after projecting) and then group by Tract FIPS while counting the number of crimes
#tract_counts = gpd.tools.sjoin(assaults.to_crs(tracts.crs), tracts.reset_index()).groupby('CTFIPS10').size()





#map.save('tweets_keywords.html')
map.save('tweets.html')

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