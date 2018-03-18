from davitpy import pydarn
import davitpy.pydarn.sdio
import datetime as dt
from matplotlib.pyplot import *
from matplotlib.dates import *
import peewee
import math
import random
from sklearn.cluster import KMeans
import numpy as np
import MySQLdb
import json

database = peewee.SqliteDatabase("space_vt.db")


class SuperDARN(peewee.Model):
	""" ORM Model of of the SuperDARN radars table"""

	name = peewee.CharField()

	class Meta:
		database = database



class Radar(peewee.Model):
	""" ORM Model of of the SuperDARN radars table"""

	name = peewee.ForeignKeyField(SuperDARN)
	beam_record = peewee.DateTimeField()
	bmnum  = peewee.IntegerField() 
	fPtr  =  peewee.CharField()
	fitex  = peewee.CharField(default = None) 
	fit  =  peewee.CharField()
	prm  =  peewee.CharField()
	recordDict  =  peewee.CharField()
	stid  = peewee.IntegerField()
	lmfit  = peewee.CharField(default = None)  
	exflg  =  peewee.CharField(default = None) 
	iqflg  =  peewee.CharField(default = None) 
	offset  =  peewee.IntegerField()
	rawacf  =  peewee.CharField()
	lmflg  = peewee.CharField(default = None)  
	rawflg  =  peewee.CharField(default = None) 
	fType  =  peewee.CharField()
	time  = peewee.DateTimeField()
	acflg  =  peewee.CharField(default = None) 
	cp  =  peewee.IntegerField()
	iqdat  = peewee.CharField()
	fitacf  =  peewee.CharField(default = None) 
	channel  = peewee.IntegerField() 



	class Meta:
		database = database



class prm(peewee.Model):
	""" ORM Model of of the Radar table"""
	name = peewee.ForeignKeyField(Radar)
	ptab = peewee.BlobField()
	mplgs = peewee.IntegerField()
	nave = peewee.IntegerField()
	noisesearch = peewee.FloatField()
	scan = peewee.IntegerField()
	smsep = peewee.IntegerField()
	mplgexs = peewee.IntegerField()
	xcf = peewee.IntegerField()
	noisesky = peewee.FloatField()
	rsep = peewee.IntegerField()
	mppul = peewee.IntegerField()
	txpl = peewee.IntegerField()
	inttsc = peewee.IntegerField()
	frang = peewee.IntegerField()
	bmazm = peewee.FloatField()
	lagfr = peewee.IntegerField()
	ifmode = peewee.IntegerField()
	noisemean = peewee.FloatField()
	tfreq = peewee.IntegerField()
	inttus = peewee.IntegerField()
	rxrise = peewee.IntegerField()
	ltab = peewee.BlobField()
	mpinc = peewee.IntegerField()
	nrang = peewee.IntegerField()

	class Meta:
		database = database


class fit(peewee.Model):
	name = peewee.ForeignKeyField(Radar)
	pwr0 = peewee.IntegerField()
	slist= peewee.IntegerField()
	w_l=peewee.IntegerField()
	elv=peewee.IntegerField()
	npnts=peewee.IntegerField()
	w_l_e=peewee.IntegerField()
	p_l=peewee.IntegerField()
	phi0_e=peewee.IntegerField()
	p_s=peewee.IntegerField()
	v_e=peewee.IntegerField()
	p_l_e=peewee.IntegerField()
	phi0=peewee.IntegerField()
	v=peewee.IntegerField()
	w_s_e=peewee.IntegerField()
	qflg=peewee.IntegerField()
	p_s_e=peewee.IntegerField()
	gflg=peewee.IntegerField()
	nlag=peewee.IntegerField()
	w_s=peewee.IntegerField()

	class Meta:
		database = database

class rawacf(peewee.Model):
	""" ORM Model of of the SuperDARN radars table"""

	name = peewee.ForeignKeyField(Radar)
	pwr0 = peewee.BlobField()
	pwr0 =peewee.BlobField()
	pwr0 =peewee.BlobField()
	pwr0 =peewee.BlobField()
	class Meta:
		database = database

class iqdat(peewee.Model):

	chnnum = peewee.FloatField()
	badtr=peewee.FloatField()
	tsze=peewee.FloatField()
	skpnum=peewee.FloatField()
	seqnum=peewee.FloatField()
	smpnum=peewee.FloatField()
	tus=peewee.FloatField()
	tsc=peewee.FloatField()
	intData=peewee.FloatField()
	tbadtr=peewee.FloatField()
	mainData=peewee.FloatField()
	toff= peewee.FloatField()
	tnoise=peewee.FloatField()
	tatten=peewee.FloatField()
	btnum=peewee.FloatField()


	class Meta:
		database = database

try:
    SuperDARN.create_table()
except peewee.OperationalError:
    print "SuperDARN table already exists!"

try:
    Radar.create_table()
except peewee.OperationalError:
    print "Radar table already exists!"

try:
    prm.create_table()
except peewee.OperationalError:
    print "prm table already exists!"

try:
    rawacf.create_table()
except peewee.OperationalError:
    print "rawacf table already exists!"

try:
    iqdat.create_table()
except peewee.OperationalError:
    print "iqdat table already exists!"

#the first routine we will call is radDataOpen, which
#establishes a data piepeline.  we will now set up the args.

#the first routine we will call is radDataOpen, which
#establishes a data piepeline.  we will now set up the args.

#sTime is the time we want to start reading (reqd input)
sTime = dt.datetime(2011,1,1,1,0)
print sTime

#rad is the 3-letter radar code for the radar we want (reqd input)
rad='bks'

#NOTE:the rest of the inputs are optional
#eTime is the end time we want to read until
eTime = dt.datetime(2011,1,2,1,0)
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

myPtr = pydarn.sdio.radDataOpen(sTime,rad,eTime=eTime,channel=channel,bmnum=bmnum,cp=cp,fileType='rawacf',filtered=filtered, src=src)
myBeam = pydarn.sdio.radDataReadRec(myPtr)
for key in myBeam.fit.__dict__.keys():
    print 'myBeam.fit.'+key

vel,t ,pwr, spec =[], [], [], []
final = []

while(myBeam!= None):
	vel.append(myBeam.fit.v)
	t.append(myBeam.time)
	pwr.append(myBeam.fit.pwr0)
	spec.append(myBeam.fit.w_l)

	myBeam = pydarn.sdio.radDataReadRec(myPtr)

v_final = []
pwr_final = []
spec_final = []
#ax = gca()
myDict = {}
maxi = -(float("inf"))


for i in len(t):
	myDict[date2num(t[i])] = {'vel': [], 'pwr'= [], 'spec' = []}


for i in range(len(t)):
    if not vel[i]: continue
  
    for j in vel[i]:
    	if j > maxi:
    		maxi = j
    	myDict[date2num(t[i])]['vel'].append(j)
    	v_final.append([date2num(t[i]), j, 'vel'])
    for j in v_final:
    	j[1] /=maxi


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
    for j in specl[i]:
    	if j > maxi:
    		maxi = j
    	myDict[date2num(t[i])]['spec'].append(j)
    	spec_final.append([date2num(t[i]), j, 'spec'])
    for j in spec_final:
    	j[1] /=maxi

dataset = v_final + pwr_final + spec_final
random.shuffle(dataset)
print dataset
X = []
y = []
for i in dataset:
	X.append(i[:2])
	y.append(i[2])
print X
# km = KMeans(n_clusters = 3)

# km.fit(X)


# def ClusterIndices(clustNum, labels_array):
# 	return np.where(labels_array = clustNum)[0]

# print ClusterIndicesNumpy(2, km.labels_)
    #scatter([date2num(t[i])]*len(vel[i]), vel[i], s=1)
# ylim([-400, 400])
# xlim([date2num(sTime), date2num(eTime)])
# tloc = MinuteLocator(interval=10)
# tfmt = DateFormatter('%H:%M')
# ax.xaxis.set_major_locator(tloc)
# ax.xaxis.set_major_formatter(tfmt)
# ylabel('Velocity [m/s]')
# grid()