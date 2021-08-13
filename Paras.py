import numpy as np
import matplotlib.pyplot as plt
# Import data
dataInput = open('earthquake.txt','r', encoding='utf-8-sig')
data = dataInput.read().split('\n')
# Initialize earthquake data
listData = []
# Read data
for dataPoints in data:
    if dataPoints == '': 
        continue
    
    elif np.array([dataPoints]).dtype != 'float' or 'int':
        listData.append(float(dataPoints))
eqData = np.zeros((len(listData),1))
for i in range(len(listData)):
    eqData[i] = listData[i]
eqData = (eqData-np.mean(eqData)) * 9.81
# Define legnth of signal series
for i in range(len(eqData)):

    if i**3 < len(eqData): continue
    else:
        N = (i**3) * 2
        break
print('# data points are: {}'.format(N))
Pad_Begin = np.zeros((100,1))
Pad_End = np.zeros((N-len(eqData)-100, 1))
Data = np.concatenate((Pad_Begin, eqData,Pad_End), axis=0)

