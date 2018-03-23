# Space@VT GSoC Task - 4 

This is my *(yet to be completed)* submission of task 4.


# Files

```spacedata.py``` was a primitive version of how I planned to go about the task and ```jsontosql.py``` was supposed to be the script for converting the JSON data structure to a SQL table. 

```space_clustering.ipynb``` is the latest iteration of my work. 

 

## Idea For Organizing the data

Initially, my plan was to explicitly define the fields in each table of the database but I kept running into problems because each DateTime value had variable number of features associated with it.  
  
After going through the readRadar.ipynb in the DaVitPy tutorial, I got an idea from the first scatter plot shown. The idea was that I would traverse all DateTime elements in myBeam, convert them to numbers using the date2num function in matplotlib and initialize them as keys to a dictionary. The value corresponding to each key would be of the following type:  
  
myDict\[date2num(DateTimeObject)\] = {feature1: \[\], feature2: \[\], ….., featureN: \[\]}

  
Then the idea was to repeatedly traverse through the database again and append values of the features to the corresponding DateTimeObject’s feature list.  
  
The time complexity of this entire operation would be big-O(N) and given that you know which feature of which DateTimeObject you’re looking for, access time complexity would be big-O(1).

## Data Preprocessing

For each DateTime objects, there are a list of values associated with it and it wouldn't make sense to retain all of these values. The listof values has been reduced down to two features:
1. feature_mean (mean)
2. feature_stddev (standard deviation)

### Corner Cases
Some of the values in the original list were ```NaN``,  they were converted to 0.

### Problems Yet to be Addressed
Some of the values of the spectral width list are ```inf```, so they need to be represented in some other form. Some sort of function that maps them to smaller values yet retaining the range must be applied. 

NOTE: As of now only 3 of the 6 features listed in the report are being considered. 


### Analysing the Clusters
(currently working on a smaller section of the data, as loading a week's worth of data from DaVitPy takes extremely long)
I did not specify the number of clusters to be created. Scikit-learn grouped the data into 8 clusters and based on maybe the elevation angle, multiple clusters can be grouped into categories to classify backscatter as ground scatter or ionospheric scatter.
For example consider the following clusters:

![Cluster 2](https://i.imgur.com/j1V1HO1.png)

![Cluster 5](https://i.imgur.com/fF4dWti.png)

Clearly the elevation angle can be used to categorize each cluster as either ground scatter or ionosphere scatter. In the future, after reading more about the features available from the Blackstone radar, there could be more parameters included to categorize the clusters.