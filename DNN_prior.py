#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import root_pandas
import pandas as pd
import ROOT as R
sns.set(color_codes=True)


# In[42]:


# Importing the dataset
#pd.set_option('display.float_format', lambda x: '%.8f' % x)
df = root_pandas.read_root('/srv/data/hosein47/Analysis/Analysis_BKGx1_etau_signal_all_pi.root',key='pi')
X = df.iloc[:,[1,3]]
y = df.iloc[:,11]
lh=df.iloc[:,6]


# In[43]:


from keras.models import load_model
model = load_model('DNN_model.h5')


# In[5]:


from keras import backend as K

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp], [out]) for out in outputs]
layer_outs = [func([X_train]) for func in functors]
#print(layer_outs[6])


# In[6]:


#adding the output of the last layer of DNN as the prior probability (value between 0 and 1)
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


# In[7]:


sns.scatterplot(data['pt'], data['prior'])


# In[8]:


plt.figure(figsize=[10,4])
sns.heatmap(data.corr(),annot=True)


# In[9]:


#binnig the cosTheta and transverse momentum
bins_ct = np.linspace(-1,1,num=11)
bins_pt = np.linspace(0,3,num=11)
bins_prior = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
data['binned_ct'] = pd.cut(data['cosTheta'], bins_ct, include_lowest=False)     #Bin values into discrete intervals
data['binned_pt'] = pd.cut(data['pt'], bins_pt, include_lowest=False)
data['binned_prior'] = pd.cut(data['prior'], bins_prior, include_lowest=False)
data=pd.DataFrame(data)
data.head()


# In[10]:


#This categorisation helps finding out the prior for eachdesired bin
gr=data.groupby(['binned_pt', 'binned_ct'])
nw= gr.mean().reset_index()
nw.head(10)


# In[11]:


sns.scatterplot(nw['pt'],nw['prior'])


# In[12]:


plt.figure(figsize=[15,7])
sns.boxplot(nw['binned_pt'], nw['prior'])


# In[13]:


# Adding the new Posterior to the dataset as PID, and examine its performance with respect to the old Posterior
data['PID']=((data['prior']*data['PionID'])/((1-data['prior'])+(data['prior'] * data['PionID'])))
analysis= pd.DataFrame.copy(data)
analysis.insert(8, "isSignal", y)
analysis = analysis[['cosTheta',"pt","prior","PionID","PID","binned_ct","binned_pt","binned_prior","isSignal"]]


# In[14]:


analysis.head()


# In[15]:


plt.figure(figsize=[10,5])
sns.heatmap(analysis.corr(), annot=True)


# In[37]:


a=analysis.iloc[:,:4]
b=analysis.iloc[:,8]
from sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.2, random_state = 0)


# In[41]:


# Applying PCA

from sklearn.decomposition import PCA
pca = PCA(n_components = None)
a = pca.fit_transform(a)
a=pd.DataFrame(a)
explained_variance = pca.explained_variance_ratio_
explained_variance


# In[ ]:





# In[ ]:





# In[ ]:




