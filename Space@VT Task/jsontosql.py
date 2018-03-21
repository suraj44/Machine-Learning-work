'''
================================================================
Task 4

This is an attempt to convert the JSON file created in 
spacedata.py to an SQL table

if we used SQLite, we could simple used the 'to_sql()' method
offered by Pandas, however, it does not support MySQL



The problem is that the JSON data is currently has a uniform and 
well defined hierarchy. I feel keeping it as a JSON would be more 
fruitful rather than flattening the structure out and converting
it to a SQL table as the access time in the case of the JSON  is
O(1)
================================================================
'''
import json
import pandas as pd
import sqlite3


db = sqlite3.connect('SuperDARN.db')
cursor = db.cursor()

with open("radData.json") as f:
	data = json.load(f)
	#data = list(data) #Dictionary needs to be flattened out. Haven't figured out the logic for this yet.
	for r in data:
		# data[r]['vel'] = [data[r]['vel']]
		# data[r]['pwr'] = [data[r]['pwr']]
		# data[r]['spec'] = [data[r]['spec']]
	 	data[r] = [data[r]]

	# data = [data]
print data
#print data
#print data
t = pd.DataFrame(data)



t.to_sql(name =  'date' ,con = db, if_exists='replace', index=False)
