#!/usr/bin/env python
# coding: utf-8


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
    

NUM_PARALLEL_EXEC_UNITS=float(10)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=10, 
                              inter_op_parallelism_threads=2,
                              log_device_placement=True,
                              allow_soft_placement=True,
                              device_count = {'CPU': 10, 'GPU': 0 })
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"


#Keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import CategoricalAccuracy
#from keras_adabound import AdaBound
from hamming import HammingLoss


#sklearn
from sklearn.model_selection import cross_val_score 
from sklearn.pipeline import Pipeline

#Others
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#os.environ["MODIN_ENGINE"] = "dask"
import pandas as pd
import root_pandas
import ROOT as R
#from keras_radam import RAdam 
#not be able to increase acc  after 77%
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')



# Importing the dataset
#pd.set_option('display.float_format', lambda x: '%.8f' % x)
#d = root_pandas.read_root('/srv/data/hosein47/generic/generic_analysis_pi.root',key='pi')
d = root_pandas.read_root('/srv/data/hosein47/generic/generic_analysis_k.root',key='k')
df=d.iloc[:,6:]

#identifing anti/particle charges
df['mcPDG'] = df['mcPDG'].abs()

#Droping the background particles without PIDs
df = df.drop(df[df.mcPDG ==2212].index).drop(df[df.mcPDG ==0].index).drop(df[df.mcPDG ==1000010020].index).drop(df[df.mcPDG ==2205].index).drop(df[df.mcPDG ==3112].index).drop(df[df.mcPDG ==3312].index).reset_index(drop=True)

#X_all = df.iloc[:,[0,2]].values
X_all = df.iloc[:,0:3].values
y_all = df.iloc[:,4].values
#lh=df.iloc[:,8]

#Encoding different categories
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder() 
encoder.fit(y_all) 
encoded_y = encoder.transform(y_all)

#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder(handle_unknown='ignore')
#y=pd.DataFrame(enc.fit_transform(df[['mcPDG']]).toarray())
#df=df.merge(enc_df, how='outer', left_index=True, right_index=True)

from collections import Counter
counter = Counter(y_all)
print(counter)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, encoded_y, test_size = 0.1, random_state = 42)

y_train= to_categorical(y_train)
y_test= to_categorical(y_test)


# # create model (old one) 
# #kernel_regularizer= regularizers.l2(0.00001)
# model = Sequential()
# model.add(Dense(256, input_dim=2, kernel_initializer= 'he_normal', kernel_constraint=max_norm(3), activation= 'relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(128, kernel_initializer= 'he_normal', kernel_constraint=max_norm(3), activation= 'relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(32, kernel_initializer= 'he_normal', kernel_constraint=max_norm(3), activation= 'relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(4, kernel_initializer= 'glorot_normal', activation= 'softmax'))


# create model(new)
model = Sequential()
model.add(Dense(256, input_dim=3, kernel_initializer= 'he_normal', kernel_regularizer= regularizers.l2(0.000001), activation= 'relu'))
model.add(BatchNormalization())
model.add(Dense(8, kernel_initializer= 'he_normal', kernel_regularizer= regularizers.l2(0.000001), activation= 'relu'))
model.add(BatchNormalization())
model.add(Dense(4, kernel_initializer= 'glorot_normal', activation= 'softmax'))


#optm = AdaBound(lr=1e-5, final_lr=0.0001)
from tensorflow.keras.optimizers import Adam
opt=Adam(lr=0.00001)


#import keras_metrics as km
#km.precision()
#def auc(y_true, y_pred):
    #auc = tf.metrics.auc(y_true, y_pred)[1]
    #K.get_session().run(tf.local_variables_initializer())
    #return auc

#Hamming loss defined as the Hamming distance between two binary strings as an XOR
hl = HammingLoss(mode='multiclass')


# Compile model
model.compile(loss= 'categorical_crossentropy' , optimizer= opt , metrics=['CategoricalAccuracy', hl])

es= EarlyStopping(monitor='val_hamming_loss', verbose=1, patience=4, mode='min', restore_best_weights=True)


#match class weights
from sklearn.utils.class_weight import compute_class_weight
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))
#Tunned class_weights
dcw={0: 4, 1: 5, 2: 0.3, 3: 6}


# Fit the model
history=model.fit(X_train, y_train, validation_data=(X_test,y_test), 
                  callbacks=[es], class_weight=d_class_weights, epochs=64, batch_size=1024, verbose=1)



#Check out the results
y_pred = model.predict(X_test).round().astype(int)
y_test = y_test.astype(int)

# Making the Confusion Matrix, report and Hamming Score
from sklearn.metrics import hamming_loss
from sklearn.metrics import multilabel_confusion_matrix,classification_report
cm=multilabel_confusion_matrix(y_test, y_pred)
train_acc = model.evaluate(X_train, y_train) 
test_acc = model.evaluate(X_test, y_test)
#target_names = ['class 0', 'class 1', 'class 2']
report= classification_report(y_test, y_pred)
hm= hamming_loss(y_test, y_pred)
print("%s: %.2f%%" % (model.metrics_names[1], train_acc[1]*100))
print("%s: %.2f%%" % (model.metrics_names[1], test_acc[1]*100))
print("Hamming_loss:",hm)
print(cm)
print(report)


#Hamming Score, opposite to the Hamming loss
def hs(y_true, y_pred, normalize=True, sample_weight=None):
    
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)



#Getting ready to plot ROC_AUC to see the performance of the Posterior
X_new = df.iloc[:,0:3].reset_index(drop=True)
y_new = model.predict_proba(X_new)
prior=pd.DataFrame(y_new)

df['pr_e']=prior[0]
df['pr_m']=prior[1]
df['pr_p']=prior[2]
df['pr_k']=prior[3]

# Adding the new Posterior to the dataset as PID, and examine its performance with respect to the old Posterior
df['PID_e']=((df['pr_e']*df['electronID'])/((-df['pr_e']+1)+(df['pr_e'] * df['electronID'])))
df['PID_m']=((df['pr_m']*df['muonID'])/((-df['pr_m']+1)+(df['pr_m'] * df['muonID'])))
df['PID_p']=((df['pr_p']*df['pionID'])/((-df['pr_p']+1)+(df['pr_p'] * df['pionID'])))
df['PID_k']=((df['pr_k']*df['kaonID'])/((-df['pr_k']+1)+(df['pr_k'] * df['kaonID'])))

#defined exclusively for Roc curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

y_score = df.iloc[:,15:19].values
y_roc= to_categorical(encoded_y)
y_pid = df.iloc[:,[6,7,5,8]].values

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_roc[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_roc.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Compute ROC curve and ROC area for PID
fpr_id = dict()
tpr_id = dict()
roc_auc_id = dict()
for i in range(4):
    fpr_id[i], tpr_id[i], _ = roc_curve(y_roc[:, i], y_pid[:, i])
    roc_auc_id[i] = auc(fpr_id[i], tpr_id[i])

# Compute micro-average ROC curve and ROC area
fpr_id["micro"], tpr_id["micro"], _ = roc_curve(y_roc.ravel(), y_pid.ravel())
roc_auc_id["micro"] = auc(fpr_id["micro"], tpr_id["micro"])



plt.figure(figsize=[15,7])
lw = 2

plt.plot(fpr[0], tpr[0], color='red',
         lw=lw, label='electronPos (area = %0.3f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], color='purple',
         lw=lw, label='muonPos (area = %0.3f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], color='black',
         lw=lw, label='pionPos (area = %0.3f)' % roc_auc[2])
plt.plot(fpr[3], tpr[3], color='olive',
         lw=lw, label='kaonPos (area = %0.3f)' % roc_auc[3])

plt.plot(fpr_id[0], tpr_id[0], color='darkorange',
         lw=1, label='electronID (area = %0.3f)' % roc_auc_id[0])
plt.plot(fpr_id[1], tpr_id[1], color='pink',
         lw=1, label='muonID (area = %0.3f)' % roc_auc_id[1])
plt.plot(fpr_id[2], tpr_id[2], color='gray',
         lw=1, label='pionID (area = %0.3f)' % roc_auc_id[2])
plt.plot(fpr_id[3], tpr_id[3], color='yellow',
         lw=1, label='kaonID (area = %0.3f)' % roc_auc_id[3])

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC_AUC')
plt.legend(loc="lower right")
plt.show()
