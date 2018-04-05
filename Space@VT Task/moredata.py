from sklearn.cluster import KMeans
from davitpy import pydarn
import davitpy.pydarn.sdio
import datetime as dt
from matplotlib.dates import *
import json
import numpy as np

'''
==================================================================================
Loading the data as shown in the documentation of DaVitPy
==================================================================================
'''
sTime = dt.datetime(2011,1,1,1,0)
print sTime


#rad is the 3-letter radar code for the radar we want (reqd input)
rad='bks'

#NOTE:the rest of the inputs are optional
#eTime is the end time we want to read until
eTime = dt.datetime(2011,1,2,2,0)
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

#print myBeam


vel,t ,pwr, spec, elv =[], [], [], [], []
final = []

while(myBeam!= None):
    vel.append(myBeam.fit.v)
    t.append(myBeam.time)
    pwr.append(myBeam.fit.pwr0)
    spec.append(myBeam.fit.w_l)
    elv.append(myBeam.fit.elv)

    myBeam = pydarn.sdio.radDataReadRec(myPtr)
print len(t)

v_final = []
pwr_final = []
spec_final = []
#ax = gca()
myDict = {}
maxi = -(float("inf"))


for i in range(len(t)):
	myDict[t[i]] = {'vel': [], 'pwr': [], 'spec' :[], 'elv':[]}


for i in range(len(t)):
    if not vel[i]: 
        continue
  
    for j in vel[i]:
    	if j > maxi:
    		maxi = j
    	myDict[t[i]]['vel'].append(j)
    	v_final.append([t[i], j, 'vel'])
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
    	myDict[t[i]]['pwr'].append(j)
    	pwr_final.append([t[i], j, 'pwr'])
for j in pwr_final:
    j[1] /=maxi
maxi = -(float("inf"))

for i in range(len(t)):
    if not spec[i]: continue
    for j in spec[i]:
    	if j > maxi:
    		maxi = j
    	myDict[t[i]]['spec'].append(j)
    	spec_final.append([t[i], j, 'spec'])
        
for i in range(len(t)):
    if not spec[i]: continue
    for j in elv[i]:
    	if j > maxi:
    		maxi = j
    	myDict[t[i]]['elv'].append(j)
for j in spec_final:
	j[1] /=maxi

#dataset = v_final + pwr_final + spec_final

# X = []
# y = []
# for i in dataset:
# 	X.append(i[:2])
# 	y.append(i[2])
#print X[:2]

import random


import pandas as pd
df = pd.DataFrame(myDict).T.reset_index()
df.to_pickle('data')