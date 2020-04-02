#!/usr/bin/env python
# coding: utf-8

#Keras
import tensorflow as tf
from tensorflow import keras
import keras_metrics as km
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras_adabound import AdaBound
import keras.backend as K

#Others
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["MODIN_ENGINE"] = "dask"
import modin.pandas as pd
import root_pandas
import ROOT as R
#from keras_radam import RAdam 
#not be able to increase acc  after 77%
sns.set_style('darkgrid')
%matplotlib inline


# Importing the dataset
#pd.set_option('display.float_format', lambda x: '%.8f' % x)
d = root_pandas.read_root('/srv/data/hosein47/generic/generic_analysis_k.root',key='k')
#df = vx.from_pandas(df, copy_index=False)
df=d.iloc[:,6:]

X_all = df.iloc[:,[0,2]].values
y_all = df.iloc[:,10].values

# undersample and plot imbalanced dataset with One-Sided Selection
from collections import Counter
from imblearn.under_sampling import OneSidedSelection
from numpy import where
# summarize class distribution
counter = Counter(y_all)
print(counter)
# define the undersampling method
undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=600000, n_jobs=-1, random_state=42)
# transform the dataset
X, y = undersample.fit_resample(X_all, y_all)
# summarize the new class distribution
counter = Counter(y)
print(counter)
#Old: Counter({0.0: 7319966, 1.0: 972038})
#New: Counter({0.0: 3857552, 1.0: 972038})

# scatter plot of examples by class label
plt.figure(figsize=[13,8])
plt.ylim([0,7])
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label), s=16, marker='.')
plt.legend()
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
#X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size = 0.1, random_state = 42)


# create model 
model = Sequential()
model.add(Dense(256, input_dim=2, kernel_initializer= 'he_normal', activation= 'relu', kernel_regularizer= regularizers.l2(0.000001)))
model.add(BatchNormalization())
model.add(Dense(16, kernel_initializer= 'he_normal', activation= 'relu', kernel_regularizer= regularizers.l2(0.000001)))
model.add(BatchNormalization())
model.add(Dense(1, kernel_initializer= 'glorot_normal', activation= 'sigmoid'))

#set the optimizer
optm = AdaBound()
from keras.optimizers import adam
opt=adam(lr=0.00001)

#Define AUC score as a metric
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# Compile model
model.compile(loss= 'binary_crossentropy' , optimizer= opt , metrics=[auc, km.precision(), 'acc'])


# Early stoping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

# Fit the model
weights = {0:1, 1:4}
history=model.fit(X_train_all, y_train_all, class_weight=weights, 
                  validation_data=(X_test_all,y_test_all), callbacks=[early_stopping], epochs=128, batch_size=1024)



#Saving the model weights
model.save("new3.h5")


# Predicting the train set trained on resampled data
y_pred = model.predict(X_test).round().astype(int)
y_test = y_test.astype(int)

# Making the Confusion Matrix, and report
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
train_acc = model.evaluate(X_train, y_train) 
test_acc = model.evaluate(X_test, y_test)
report= classification_report(y_test,y_pred)
print("%s: %.2f%%" % (model.metrics_names[1], train_acc[1]*100))
print("%s: %.2f%%" % (model.metrics_names[1], test_acc[1]*100))
print(cm)
print(report)
labels = ['Class 1', 'Class 0']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()


# Visualising the Training set results

plt.figure(figsize=[15,7])
from matplotlib.colors import ListedColormap
sns.set_style('darkgrid')
X_set, y_set = X_train_all, y_train_all
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.5, stop = X_set[:, 0].max() + 0.5, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 0.5, stop = X_set[:, 1].max() + 0.5, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('skyblue','tomato')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue','red'))(i), label = j, s=1)
plt.title('DNN (Training set)')
plt.xlabel('cosTheta')
plt.ylim([0,7])
plt.ylabel('Pt')
plt.legend()
plt.show()

# Visualising the Test set results
plt.figure(figsize=[15,7])
from matplotlib.colors import ListedColormap
sns.set_style('darkgrid')
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.5, stop = X_set[:, 0].max() + 0.5, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 0.5, stop = X_set[:, 1].max() + 0.5, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('skyblue','tomato')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue','red'))(i), label = j, s=1)
plt.title('DNN (Test set)')
plt.xlabel('cosTheta')
plt.ylim([0,7])
plt.ylabel('Pt')
plt.legend()
plt.show()

# summarize history for accuracy
plt.figure(figsize=[14,7])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('model accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# summarize history for loss
plt.figure(figsize=[14,7])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('model loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

