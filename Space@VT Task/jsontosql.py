'''
=============================================================
Task 4

This is an attempt to convert the JSON file created in 
spacedata.py to an SQL table

if we used SQLite, we could simple used the 'to_sql()' method
offered by Pandas, however, it does not support MySQL.
=============================================================
'''
import json
import pandas as pd
import sqlite3


db = sqlite3.connect('SuperDARN.db')
cursor = db.cursor()

with open("radData.json") as f:
	data = json.load(f)
	data = list(data) #Dictionary needs to be flattened out. Haven't figured out the logic for this yet.
print data
t = pd.DataFrame(data)



t.to_sql(name =  'date' ,con = db, if_exists='replace', index=False)
