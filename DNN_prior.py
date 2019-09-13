#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import root_pandas
import pandas as pd
import ROOT as R
sns.set(color_codes=True)


# Importing the dataset
#pd.set_option('display.float_format', lambda x: '%.8f' % x)
df = root_pandas.read_root('/srv/data/hosein47/Analysis/Analysis_BKGx1_etau_signal_all_pi.root',key='pi')
X = df.iloc[:,[1,3]]
y = df.iloc[:,11]
lh=df.iloc[:,6]

#Load the DNN model
from keras.models import load_model
model = load_model('DNN_model.h5')



#adding the output of the last layer of DNN as the prior probability (value between 0 and 1)
from keras import backend as K
inp = model.input
output=model.layers[6].output
functor=K.function([inp],[output])
out=np.array(functor([X]))
arr=np.reshape(out,(85289,-1))
prior=pd.DataFrame(arr)
data=pd.DataFrame(X)
data.insert(2, "prior", prior)
data.insert(3, "PionID", lh)
#data.insert(4, "isSignal", y)
data.head()



sns.scatterplot(data['pt'], data['prior'])


#binnig the cosTheta and transverse momentum
bins_ct = np.linspace(-1,1,num=11)
bins_pt = np.linspace(0,3,num=11)
bins_prior = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
data['binned_ct'] = pd.cut(data['cosTheta'], bins_ct, include_lowest=False)     #Bin values into discrete intervals
data['binned_pt'] = pd.cut(data['pt'], bins_pt, include_lowest=False)
data['binned_prior'] = pd.cut(data['prior'], bins_prior, include_lowest=False)
data=pd.DataFrame(data)
data.head()


#This categorisation helps finding out the prior for each desired bin
gr=data.groupby(['binned_pt', 'binned_ct'])
nw= gr.mean().reset_index()
nw.head(10)


plt.figure(figsize=[15,7])
sns.boxplot(nw['binned_pt'], nw['prior'])
sns.scatterplot(nw['pt'],nw['prior'])


# Adding the new Posterior to the dataset as PID, and examine its performance with respect to the old Posterior
data['PID']=((data['prior']*data['PionID'])/((1-data['prior'])+(data['prior'] * data['PionID'])))
analysis= pd.DataFrame.copy(data)
analysis.insert(8, "isSignal", y)
analysis = analysis[['cosTheta',"pt","prior","PionID","PID","binned_ct","binned_pt","binned_prior","isSignal"]]
analysis.head()



#Defing a function to show distplot with hue (Should add it to the Seaborn repo)
#The prior is actually correct as the output of my DNN
def distplot_with_hue(data=None, x=None, hue=None, row=None, col=None, legend=True, **kwargs):
    _, bins = np.histogram(data[x].dropna())
    g = sns.FacetGrid(data, hue=hue, row=row, col=col, height=8, aspect=2)
    g.map(sns.distplot, x, **kwargs)
    if legend and (hue is not None) and (hue not in [x, row, col]):
        g.add_legend(title=hue)
        
distplot_with_hue(data=analysis, x='prior', hue='isSignal', hist=True)




#Checking the correlation between PID and isSignal
plt.figure(figsize=[10,5])
sns.heatmap(analysis.corr(), annot=True)



#Comparing the new posterior, prior, and old PionID (The new PID is doing better!)
from sklearn.metrics import roc_curve, auc
u = analysis['isSignal'].values
scores_prior= analysis['prior'].values
fpr_prior, tpr_prior, thresholds_prior = roc_curve(u, scores_prior)
roc_auc_prior = auc(fpr_prior, tpr_prior)

scores_PionID= analysis['PionID'].values
fpr_PionID, tpr_PionID, thresholds_PionID =roc_curve(u,scores_PionID)
roc_auc_PionID = auc(fpr_PionID, tpr_PionID)

scores_PID= analysis['PID'].values
fpr_PID, tpr_PID, thresholds_PID =roc_curve(u,scores_PID)
roc_auc_PID = auc(fpr_PID, tpr_PID)

plt.figure()
lw = 2
plt.figure(figsize=[15,7])
plt.plot(fpr_prior, tpr_prior, color= 'red', lw=lw, label='ROC curve prior (area = %0.3f)' % roc_auc_prior)
plt.plot(fpr_PionID, tpr_PionID, color= 'black', lw=lw, label='ROC curve PionID (area = %0.3f)' % roc_auc_PionID)
plt.plot(fpr_PID, tpr_PID, color= 'green', lw=lw, label='ROC curve PID (area = %0.3f)' % roc_auc_PID)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([-0.01, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('true Positive Rate')
plt.title('ROC_AUC')
plt.legend(loc="lower right")
plt.show()

