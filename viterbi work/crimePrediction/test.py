"""
test

"04/08/2016 07:30:00 AM","04/08/2016","MISDEMEANORS MISCELLANEOUS","399","MISDEMEANORS, MISCELLANEOUS: All Other Misdemeanors","300 W ALMOND ST, COMPTON, CA  90220","300 W ALMOND ST","COMPTON","90220",6492697.18845464,1784046.04811607,"916-04186-2835","2835","04186","NO","CA0190042","COMPTON","N"


"01/06/2016 10:00:00 AM","01/06/2016","MISCELLANEOUS NON-CRIMINAL","733","VEHICLE/BOAT, OTHER NON-CRIMINAL: Veh/Boat Recovered (Other Juris)","800 E 104TH ST, LOS ANGELES, CA","800 E 104TH ST","LOS ANGELES",,6483070.70197929,1801382.31348969,"916-00007-3803","3803","00007","NO","CA01900V1","TRAP","N"

"""

#import stateplane

#print stateplane.tolatlon(6492697.18845464,1784046.04811607,fips = 06037)

# los angeles county epsg :26945
#source : https://gist.github.com/fitnr/10795511

from pyproj import Proj, transform

inProj = Proj(init = 'epsg:2229',preserve_units = True)
outProj = Proj(init = 'epsg:4326')
x1,y1 = 6492697.18845464,1784046.04811607
#x1,y1 = 6483070.70197929,1801382.31348969
x2,y2 = transform(inProj,outProj,x1,y1)
print x2,y2