import multiprocessing
multiprocessing.set_start_method('forkserver')

#Others
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import root_pandas
import ROOT as R
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Importing the dataset

#d = root_pandas.read_root('/srv/data/hosein47/generic/generic_analysis_pi.root',key='pi')
d = root_pandas.read_root('/srv/data/hosein47/generic/generic_analysis_k.root',key='k')

df=d.iloc[:,6:]
df['mcPDG'] = df['mcPDG'].abs()
df = df.drop(df[df.mcPDG ==2212].index).drop(df[df.mcPDG ==0].index).drop(df[df.mcPDG ==1000010020].index).drop(df[df.mcPDG ==2205].index).drop(df[df.mcPDG ==3112].index).drop(df[df.mcPDG ==3312].index).reset_index(drop=True)

#X_all = df.iloc[:,[0,2]].values
X_all = df.iloc[:,0:3].values
y_all = df.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y_all)
encoded_y = encoder.transform(y_all)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, encoded_y, test_size = 0.1, random_state = 42)

#Using xgboost
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
w_array = np.ones(y_train.shape[0], dtype = 'float')
for i, val in enumerate(y_train):
    w_array[i] = class_weights[val-1]

#xgboost model
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=4, learning_rate=0.1, n_estimators=1000,
                              random_state=42, subsample=0.5, reg_lambda=0.01, n_jobs=32)
eval_set = [(X_test, y_test)]
xgb_model.fit(X_train, y_train, eval_metric=["mlogloss"],
              early_stopping_rounds=8, eval_set=eval_set, sample_weight=w_array, verbose=True)




from sklearn.metrics import hamming_loss
from sklearn.metrics import multilabel_confusion_matrix,classification_report
y_pred= xgb_model.predict(X_test)
cm=multilabel_confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test,y_pred)
report= classification_report(y_test, y_pred)
hm= hamming_loss(y_test, y_pred)
print("Hamming_loss:",hm)
print("Accuracy:",accuracy)
print(cm)
print(report)


# In[ ]:
