#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import root_pandas
import pandas as pd
from keras.models import load_model
import ROOT as R
#sns.set(color_codes=True)



#import data
pd.set_option('display.float_format', lambda x: '%.10f' % x)
df = root_pandas.read_root('/srv/data/hosein47/Analysis/Analysis_BKGx1_etau_signal_all_pi.root',key='pi')
X = df.iloc[:,[1,3]]
lh=df.iloc[:,6]
y = df.iloc[:,11]

y=pd.DataFrame(y)
data=X.join(y)
data.head()




#Find out the number of sgnals in each bin
plt.figure(figsize=[15,8])

H_signal, xedges_1, yedges_1,im = plt.hist2d(x=data['cosTheta'], y=data['pt'], bins=(8,12), range=[[-1, 1], [0, 3]], weights=data['isSignal']==1, normed=False)
extent = [0,3, 1, -1]
plt.imshow(H_signal, extent=extent,interpolation='nearest')

for i in range(len(xedges_1)-1):
    for j in range(len(yedges_1)-1):
        plt.text(xedges_1[i]+0.125,yedges_1[j]+0.125, int(H_signal[i,j]), 
                color="w", ha="center", va="center", fontweight="bold")

plt.colorbar()
plt.show()




#Find out the number of all points in each bin including both signal and background
plt.figure(figsize=[15,8])

H_all, xedges_2, yedges_2 ,im = plt.hist2d(x=data['cosTheta'],y=data['pt'], bins=(8,12),range=[[-1, 1], [0, 3]])
extent = [0,3, 1, -1]
plt.imshow(H_all, extent=extent,interpolation='nearest')

for i in range(len(xedges_2)-1):
    for j in range(len(yedges_2)-1):
        plt.text(xedges_2[i]+0.125,yedges_2[j]+0.125, int(H_all[i,j]), 
                color="w", ha="center", va="center", fontweight="bold")

plt.colorbar()
plt.show()




#The number of signals over the number of background which is the prior
plt.figure(figsize=[15,8])

H_signal, xedges, yedges, im = plt.hist2d(x=data['cosTheta'], y=data['pt'], bins=(8,12), range=[[-1, 1], [0, 3]], weights=data['isSignal']==1,normed=False)
H_all, xedges, yedges, im = plt.hist2d(x=data['cosTheta'],y=data['pt'], bins=(8,12),range=[[-1, 1], [0, 3]])
extent = [0,3, 1, -1]
plt.imshow(H_signal/H_all, extent=extent,interpolation='nearest')

for i in range(len(xedges)-1):
    for j in range(len(yedges)-1):
        plt.text(xedges[i]+0.125,yedges[j]+0.125, np.around((np.nan_to_num(H_signal/H_all))[i,j]*100,decimals=0), 
                 color="w", ha="center", va="center", fontweight="bold")
plt.colorbar()
plt.show()



#transfering the info to a dataframe
plt.figure(figsize=[15,8])
H1=pd.DataFrame(H_signal)
H2=pd.DataFrame(H_all)
prior=(H1/H2)
prior.replace(np.nan, 0, inplace=True)
prior




#reorganizing each bin into a 1-d dataframe
ls=[]
for i in range(0,8):
    for j in range(0,12):
        ls.append(prior[j][i])

new_prior=pd.DataFrame(ls).rename(columns={0:"prior"})



#biining the data and adding the new column which is the prior
bins_pt = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
bins_ct=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
com= pd.DataFrame.copy(X)
com=com.join(lh)
com['binned_pt'] = pd.cut(com['pt'], bins_pt)
com['binned_ct']= pd.cut(com['cosTheta'], bins_ct)
com=com.groupby(['binned_ct','binned_pt']).mean().reset_index()
#com2=com1.mean().reset_index()
com=com.join(new_prior)
com.replace(np.nan, 0, inplace=True)



#Defining the new posterior as "PID"
com['PID']=((com['prior']*com['pionID'])/((1-com['prior'])+(com['prior'] * com['pionID'])))
analysis= pd.DataFrame.copy(com)
analysis = analysis[['cosTheta',"pt","prior","pionID","PID","binned_ct","binned_pt"]]
analysis.head(12)




plt.figure(figsize=[10,5])
sns.heatmap(analysis.corr(), annot=True)



print(__doc__)
import rootpy
from rootpy.plotting import root2matplotlib as rplt
from rootpy.plotting import Hist2D

a = Hist2D(50, -1, 1, 50, 0, 3)
a.fill_array(data.iloc[:,[0,1]].values)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

ax1.set_title('hist2d')
rplt.hist2d(a, axes=ax1)

ax2.set_title('imshow')
im = rplt.imshow(a, axes=ax2)

ax3.set_title('contour')
rplt.contour(a, axes=ax3)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

if not R.gROOT.IsBatch():
    plt.show()




