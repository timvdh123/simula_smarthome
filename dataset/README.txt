Thank you for downloading the semiDataset. This document explains how the dataset is structured. 
If you have any questions or suggestions please email me at: tim0306@gmail.com

These datasets were used for the experiments as described in: 
	Activity Recognition Using Semi-Markov Models on Real World Smart Home Datasets
	T.L.M. van Kasteren, G. Englebienne and B.J.A. Kröse
	In Journal of Ambient Intelligence and Smart Environments thematic issue on Smart Homes, IOS Press, 2010
Please refer to this paper when you use this dataset in your publications.

Other datasets and publications can be found at: http://sites.google.com/site/tim0306/

Contents of dataset package: 
- Actual data in matlab form (semiDatasets.mat)
- actstruct and sensorstruct are structures used for storing the data in matlab


Using the dataset in matlab:
----------------------------
Make sure the actstruct and sensorstruct directories are copied to a directory which is included in your matlab path. After that just load the semiDatasets.mat
into matlab. This should give you fourvariables bathroom1, bathroom2, kitchen1 and kitchen2, corresponding to the datasets as described in the paper. The as variable contains the annotation information, the ss variable contains the sensor information.

Some commands that are of use:
bathroom1.as.getIDs				= List of activities ids
bathroom1.activity_labels(houseA.as.getIDs) 	= List of activity labels
bathroom1.ss.getIDs				= List of sensor ids
bathroom1.sensor_labels				= List of sensor labels

Code that might be of use:
--------------------------
For a visualization tool and scripts for discretizing the data please download my ubicomp dataset from http://sites.google.com/site/tim0306/
For code of models for activity recognition and additional datasets please download my benchmark dataset from http://sites.google.com/site/tim0306/