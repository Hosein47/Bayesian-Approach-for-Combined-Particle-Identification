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
sns.set(color_codes=True)



# Load the DNN model
model = load_model('DNN_model.h5')

# A naive function to estimate the prior on the whole data
def prior(X,y):
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X)
    y_pred = y_pred.round().astype(int)
    y_test = y.astype(int)
    cm = confusion_matrix(y, y_pred)
    prior_prob = (cm[1][1]/len(y))*100
    print("The prior probibility is: %.2f%%" % (prior_prob) )


#Loading data from .root files
pd.set_option('display.float_format', lambda x: '%.10f' % x)
df = root_pandas.read_root('/srv/data/hosein47/Analysis/Analysis_BKGx1_etau_signal_all_pi.root',key='pi')
X = df.iloc[:,[1,3]].values
y = df.iloc[:,11].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Predicting the Test set results with DNN
y_pred = model.predict(X_test)
y_pred = y_pred.round().astype(int)
y_test = y_test.astype(int)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
train_acc = model.evaluate(X_train, y_train) 
test_acc = model.evaluate(X_test, y_test)
report_model= classification_report(y_test,y_pred)
print("%s: %.2f%%" % (model.metrics_names[1], train_acc[1]*100))
print("%s: %.2f%%" % (model.metrics_names[1], test_acc[1]*100))
print(cm)
print(report_model)


#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()


# Visualising the Training set results
plt.figure(figsize=[15,7])
from matplotlib.colors import ListedColormap
sns.set_style('darkgrid')
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.5, stop = X_set[:, 0].max() + 0.5, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 0.5, stop = X_set[:, 1].max() + 0.5, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('tomato', 'skyblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label = j, s=1)
plt.title('DNN (Training set)')
plt.xlabel('cosTheta')
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
             alpha = 0.75, cmap = ListedColormap(('tomato', 'skyblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label = j, s=1)
plt.title('DNN (Test set)')
plt.xlabel('cosTheta')
plt.ylabel('Pt')
plt.legend()
plt.show()

#Using Other methods:



# Using Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
Classifier_RF=RandomForestClassifier(n_estimators=50).fit(X_train,y_train)
y_pred_RF = Classifier_RF.predict(X_test) 
accuracy_RF=accuracy_score(y_test,y_pred_RF)
print(accuracy_RF)


#Using kernel SVM

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
classifier_SVM = SVC(C=100,kernel = 'rbf', random_state = 0)
classifier_SVM.fit(X_train, y_train)
y_pred_SVM = classifier_SVM.predict(X_test)
cm_SVM = confusion_matrix(y_test, y_pred_SVM)
accuracy_SVM=accuracy_score(y_test,y_pred_SVM)
print(cm_SVM)
print(accuracy_SVM)

#Save SVM
import pickle
filename = 'classifier_SVM.sav'
pickle.dump(classifier_SVM, open(filename, 'wb'))

#load SVM

import pickle
classifier_SVM= pickle.load(open('classifier_SVM.sav', 'rb'))

y_pred_SVM = classifier_SVM.predict(X_test)
report_SVM= classification_report(y_test,y_pred_SVM)
print(report_SVM)


#Using xgboost

import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, reg_lambda=0.01)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb=accuracy_score(y_test,y_pred_xgb)
print(accuracy_xgb)
print(confusion_matrix(y_test, y_pred_xgb))





#Comparing the models using ROC_AUC
from sklearn.metrics import roc_curve, auc
y = y_test
scores_DNN= model.predict(X_test)
fpr_DNN, tpr_DNN, thresholds_DNN = roc_curve(y, scores_DNN)
roc_auc_DNN = auc(fpr_DNN, tpr_DNN)

scores_SVM= classifier_SVM.predict(X_test)
fpr_SVM, tpr_SVM, thresholds_SVM =roc_curve(y,scores_SVM)
roc_auc_SVM = auc(fpr_SVM, tpr_SVM)


scores_RF= Classifier_RF.predict(X_test)
fpr_RF, tpr_RF, thresholds_RF =roc_curve(y,scores_RF)
roc_auc_RF = auc(fpr_RF, tpr_RF)

scores_xgb= xgb_model.predict(X_test)
fpr_xgb, tpr_xgb, thresholds_xgb =roc_curve(y,scores_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)


plt.figure()
lw = 2
plt.figure(figsize=[15,7])
plt.plot(fpr_DNN, tpr_DNN, color= 'red', lw=lw, label='ROC curve DNN (area = %0.3f)' % roc_auc_DNN)
plt.plot(fpr_xgb, tpr_xgb, color= 'black', lw=lw, label='ROC curve xgb (area = %0.3f)' % roc_auc_xgb)
plt.plot(fpr_SVM, tpr_SVM, color= 'green', lw=lw, label='ROC curve SVM (area = %0.3f)' % roc_auc_SVM)
plt.plot(fpr_RF, tpr_RF, color= 'yellow', lw=lw, label='ROC curve RF (area = %0.3f)' % roc_auc_RF)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([-0.01, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('true Positive Rate')
plt.title('ROC_AUC')
plt.legend(loc="lower right")
plt.show()




