#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import root_pandas
import pandas as pd
import ROOT as R
sns.set(color_codes=True)


# In[6]:


# Importing the dataset
#pd.set_option('display.float_format', lambda x: '%.8f' % x)
df = root_pandas.read_root('/srv/data/hosein47/Analysis/Analysis_BKGx1_etau_signal_all_pi.root',key='pi')
X = df.iloc[:,[1,3]].values
y = df.iloc[:,11].values


# In[12]:


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[13]:


from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers.normalization import BatchNormalization
from keras import regularizers


# In[14]:


# create model 
model = Sequential()
model.add(Dense(12, input_dim=2, kernel_initializer= 'he_uniform', activation= 'relu', kernel_regularizer= regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dense(10, kernel_initializer= 'he_uniform' , activation= 'relu', kernel_regularizer= regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dense(4, kernel_initializer= 'he_uniform' , activation= 'relu', kernel_regularizer= regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dense(1, kernel_initializer= 'normal' , activation= 'sigmoid'))


# In[15]:


# Compile model
model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])


# In[16]:


# Fit the model
history=model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20, batch_size=512)


# In[19]:


model.save("DNN_model.h5")


# In[17]:


# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = y_pred.round().astype(int)
y_test = y_test.astype(int)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
train_acc = model.evaluate(X_train, y_train) 
test_acc = model.evaluate(X_test, y_test)
report= classification_report(y_test,y_pred)
print("%s: %.2f%%" % (model.metrics_names[1], train_acc[1]*100))
print("%s: %.2f%%" % (model.metrics_names[1], test_acc[1]*100))
print(cm)
print(report)


# In[24]:


# summarize history for accuracy
plt.figure(figsize=[14,7])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('model accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


# In[25]:


# summarize history for loss
plt.figure(figsize=[14,7])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('model loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


# In[ ]:




