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
import numpy as np
import pickle

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


#print myBeam

vel,t ,pwr, spec, gflg, slist =[], [], [], [], [], []
final = []

while(myBeam!= None):
	vel.append(myBeam.fit.v)
	t.append(myBeam.time)
	pwr.append(myBeam.fit.pwr0)
	spec.append(myBeam.fit.w_l)
	gflg.append(myBeam.fit.gflg)
	slist.append(myBeam.fit.slist)

	myBeam = pydarn.sdio.radDataReadRec(myPtr)

for i in slist[:10]:
	print i

print '\n'
for i in gflg[:10]:
	print i