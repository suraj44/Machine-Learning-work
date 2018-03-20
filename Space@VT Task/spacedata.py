'''
==================================================================================
Task 4
Brief explanation: 

The data structure I'm using to create the SQL database is a ditionary myDict wherein
each key in the dictionary is a date2num converted number repreneting a particular 
DateTime instance and the corresponding value to that key is a dictionary with each key as 
a feature and the corresponding value a list containing the values of the particular feature
corresponding to that DateTime instance.

Building the data structure would take O(N) time complexity
but access time for a given feature would be O(N) this way



Note: Due to technical issues, I was without a working machine for a weak and hence 
did not have a resource to work on the task.


Based on the research paper conerning the SuperDARN group of radars, I identified 
some features that would be suitable to feed to the KMeans clustering algorithm.


I have normalized the data by dividing all values of a given feature fo a particular 
DateTime instance by the maximum value of the feature corresponding to that DateTime
instance. 

==================================================================================


Yet to complete:
1. Using the jsons module to dump the structure into a SQL database
2. Running the KMeans algorithm on the data
'''

from sklearn.cluster import KMeans
from davitpy import pydarn
import davitpy.pydarn.sdio
import datetime as dt
from matplotlib.dates import *
import json

sTime = dt.datetime(2011,1,1,1,0)
print sTime

#rad is the 3-letter radar code for the radar we want (reqd input)
rad='bks'

#NOTE:the rest of the inputs are optional
#eTime is the end time we want to read until
eTime = dt.datetime(2011,1,1,2,0)
print eTime

#channel is the radar channel we want data from
#By default this is set to None.
#Note: For certain radars, like the UAF radars, the channel must
#be explicitly identified such as 'a'.
channel=None

#bmnum is the beam number we want data from.  by default this is
#None, which will read data from all beams
bmnum=7

#cp is the control program id number which we want data from
#by default, this is set to None which reads data from all cpids
cp=None

#fileType specifies the type of data we want.  valid inputs are
#'fitex','fitacf','lmfit','rawacf'.  by default this is 'fitex'
#if a fit type is requested but not found, the code will automatically
#look for other fit types
fileType='fitacf'

#filter is a boolean indicating whether to boxcar filter the data.
#this is onyl valid for fit types, and wont work on mongo data
filtered=False

#src is a string indicating the desired data source.  valid
#inputs are 'mongo','local','sftp'.  by default this is set to
#None which will sequentially try all sources
src=None

myPtr = pydarn.sdio.radDataOpen(sTime,rad,eTime=eTime,channel=channel,bmnum=bmnum,cp=cp,fileType=fileType,filtered=filtered, src=src)


myBeam = pydarn.sdio.radDataReadRec(myPtr)

print myBeam

vel,t ,pwr, spec =[], [], [], []
final = []

while(myBeam!= None):
	vel.append(myBeam.fit.v)
	t.append(myBeam.time)
	pwr.append(myBeam.fit.pwr0)
	spec.append(myBeam.fit.w_l)

	myBeam = pydarn.sdio.radDataReadRec(myPtr)

#print vel[0]


v_final = []
pwr_final = []
spec_final = []
#ax = gca()
myDict = {}
maxi = -(float("inf"))


for i in range(len(t)):
	myDict[date2num(t[i])] = {'vel': [], 'pwr': [], 'spec' :[]}


for i in range(len(t)):
    if not vel[i]: continue
  
    for j in vel[i]:
    	if j > maxi:
    		maxi = j
    	myDict[date2num(t[i])]['vel'].append(j)
    	v_final.append([date2num(t[i]), j, 'vel'])
for j in v_final:
	j[1] /= float(maxi)


temp_list = []
for i in v_final:
	if i[0] not in temp_list:
		temp_list.append(i[01])

maxi = -(float("inf"))

for i in range(len(t)):
    if not pwr[i]: continue
    for j in pwr[i]:
    	if j > maxi:
    		maxi = j
    	myDict[date2num(t[i])]['pwr'].append(j)
    	pwr_final.append([date2num(t[i]), j, 'pwr'])
for j in pwr_final:
    j[1] /=maxi
maxi = -(float("inf"))

for i in range(len(t)):
    if not spec[i]: continue
    for j in spec[i]:
    	if j > maxi:
    		maxi = j
    	myDict[date2num(t[i])]['spec'].append(j)
    	spec_final.append([date2num(t[i]), j, 'spec'])
for j in spec_final:
	j[1] /=maxi

dataset = v_final + pwr_final + spec_final

X = []
y = []
for i in dataset:
	X.append(i[:2])
	y.append(i[2])
#print X[:2]

import random

#print myDict[734138.0419535764]


import json
with open('radData.json', 'w') as fp:
	json.dump(myDict, fp)

# km = KMeans(n_clusters = 3)

# km.fit(X)


# def ClusterIndices(clustNum, labels_array):
# 	return np.where(labels_array = clustNum)[0]

# print ClusterIndicesNumpy(2, km.labels_)