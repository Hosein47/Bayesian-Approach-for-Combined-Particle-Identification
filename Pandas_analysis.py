#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import root_pandas
import pandas as pd
import ROOT as R
sns.set_style('whitegrid')
sns.set(color_codes=True)


# In[4]:


pd.set_option('display.float_format', lambda x: '%.8f' % x)
#df = root_pandas.read_root('/srv/data/hosein47/Analysis/Analysis_BKGx1_etau_signal_all_mu.root',key='mu')
df = root_pandas.read_root('/srv/data/hosein47/Analysis/Analysis_BKGx1_etau_signal_all_pi.root',key='pi')
#df = root_pandas.read_root('/srv/data/hosein47/Analysis/Analysis_BKGx1_etau_signal_all_e.root',key='e')
#df = root_pandas.read_root('/srv/data/hosein47/Analysis/Analysis_BKGx1_etau_signal_all_k.root',key='k')
#df.describe()
#mu.info()


# In[5]:


df.describe()


# In[6]:


#unique_particles = np.unique(mu['M'].values)
#unique_particles
signal= df[df['isSignal']==1]['pionID']
background= df[df['isSignal']==0]['pionID']


# In[7]:


#cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
plt.figure(figsize=[15,7])
sns.scatterplot(x= 'cosTheta', y= 'pt', data=df, hue='isSignal')
#plt.legend(loc='upper right')


# In[10]:


sns.catplot(x="isSignal", y="pt", data=df)


# In[11]:


sns.distplot(df['cosTheta'], bins=20, kde=False)


# In[10]:


df['pt'].hist(bins=30, by=df['isSignal'])


# In[32]:


sns.scatterplot(x= 'pionID', y= 'muonID', data=df, hue='isSignal')


# In[24]:


from sklearn.metrics import roc_curve, auc
y = df['isSignal'].values
scores = df['pionID'].values
fpr, tpr, thresholds = roc_curve(y, scores)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.figure(figsize=[15,7])
plt.plot(fpr, tpr, color= 'red', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([-0.01, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('true Positive Rate')
plt.title('ROC_AUC')
plt.legend(loc="lower right")
plt.show()


# In[36]:


from sklearn.metrics import roc_curve, auc
plt.figure(figsize=[15,7])
lw = 2
y = df['isSignal'].values
for column in df[['muonID', 'pionID', 'kaonID','electronID']]:
    scores=df[column]
    fpr, tpr, thresholds = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr,lw=lw, label='ROC curve  (area = %0.3f)' % roc_auc)
    print(column)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([-0.01, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('true Positive Rate')
plt.title('ROC_AUC')
plt.legend(loc="lower right")
plt.show()


# In[38]:


sns.boxplot(y='pt', x='isSignal',data=df)


# In[40]:


sns.boxplot(y='cosTheta', x='isSignal',data=df)


# In[7]:


df['pt'].hist(bins=30)


# In[8]:


sns.jointplot(x='pt',y='cosTheta', data=df)


# In[25]:


plt.figure(figsize=[15,8])
sns.heatmap(df.corr(), vmin=0, vmax=1, annot=True)


# In[37]:


plt.figure(figsize=[15,7])
sns.distplot(df['pionID'],bins=30, kde=False, rug=False)


# In[79]:


sns.pairplot(df)


# In[ ]:


plt.figure(figsize=[15,8])
pv=df.pivot_table(values='isSignal',index='pt',columns='cosTheta')
sns.heatmap(pv)


# In[8]:


from sklearn.ensemble import ExtraTreesClassifier

X = df.iloc[:,1:5]
y = df.iloc[:,11]

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(4).plot(kind='barh')
plt.show()


# In[15]:


#PCA analysis
from sklearn.decomposition import PCA
X = df.iloc[:,1:5].values
y = df.iloc[:,11].values
# feature extraction
pca = PCA(n_components=4)
fit = pca.fit(X)
# summarize components
print(("Explained Variance: %s") % fit.explained_variance_ratio_)
print(fit.components_)


# In[20]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = df.iloc[:,2:5]
y = df.iloc[:,11]

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(3,'Score'))  #print 10 best features


# In[33]:


X = df.iloc[:,1:5]
y = df.iloc[:,11]
from sklearn.feature_selection import mutual_info_classif
info=mutual_info_classif(X,y)
print(info)


# In[ ]:




