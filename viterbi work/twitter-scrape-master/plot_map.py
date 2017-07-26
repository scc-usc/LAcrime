"""
Autjor : Omkar Damle
Date : 26th May 2017

This code plots points where crime has occured on a map 
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('GTK')

"""
map = Basemap(projection='merc', lat_0 = 34,lon_0 = -118.25,
	resolution = 'h', area_thresh = 0.1,
	llcrnlon = -118.5, llcrnlat = 33.8,
	urcrnlon = -118, urcrnlat = 34.2)
"""
map = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)

map.drawcoastlines()

plt.show()
