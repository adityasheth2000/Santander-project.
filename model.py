import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import gc
from sklearn import preprocessing

prod_cols=['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

train2=pd.read_csv('../input/santander2/original_data_first_half.csv') #Read from the output done in output_code_2.py
train1=pd.read_csv('../input/santander/original_data_second_half.csv') #Read from the output done in output_code_1.py

test_data=pd.read_csv('../input/santander/test_data.csv')			   #Read from the output done in output_code_1.py

mycols=['ncodpers']
for c in prod_cols:
    mycols.append(c+'_sum00')
    mycols.append(c+'_sum01')
    mycols.append(c+'_sum10')
    mycols.append(c+'_sum11')

#Add toggle features from train2 to train1, so that we can take total sum of x_sum00 and x_sum00_ex into x_sum00 for every product x,
#and same way for every other toggle features.

train1=train1.merge(train2[mycols],how='left',on='ncodpers',suffixes=(None,'_ex')) 
del train2

for c in prod_cols:
    train1[c+'_sum00']+=train1[c+'_sum00_ex']
    train1[c+'_sum01']+=train1[c+'_sum01_ex']
    train1[c+'_sum10']+=train1[c+'_sum10_ex']
    train1[c+'_sum11']+=train1[c+'_sum11_ex']
    train1.drop([c+'_sum00_ex',c+'_sum01_ex',c+'_sum10_ex',c+'_sum11_ex'],axis='columns',inplace=True)


#Make toggle features same in test_data and train_data for a particular customer.
test_data=test_data.merge(train1[mycols],how='left',on='ncodpers',suffixes=(None,'_ex'))
for c in prod_cols:
    test_data[c+'_sum00']=test_data[c+'_sum00_ex']
    test_data[c+'_sum01']=test_data[c+'_sum01_ex']
    test_data[c+'_sum10']=test_data[c+'_sum10_ex']
    test_data[c+'_sum11']=test_data[c+'_sum11_ex']
    test_data.drop([c+'_sum00_ex',c+'_sum01_ex',c+'_sum10_ex',c+'_sum11_ex'],axis='columns',inplace=True)


gc.collect()
train_data=train1

# <-----------------------__________MODEL TRAINING STARTED_________------------------->
print('Model Training started :)')
# %% [code]
from sklearn import ensemble
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.neural_network import MLPClassifier
models={}
model_preds={}
id_preds=defaultdict(list)
training_features =list(train_data.columns)

for i in prod_cols:
    training_features.remove(i)
#Now training features include everything except the product names itself.
ids=test_data['ncodpers'].values

prod_df=pd.DataFrame()
for c in prod_cols:
    prod_df[c]=train_data[c]
    train_data.drop(c,axis='columns',inplace=True)
    
for c in prod_cols:
    print(c)    
    x_train=train_data 
    y_train=prod_df[c]    
    x_test=test_data    
    clf = lgb.LGBMClassifier(n_estimators=400,num_leaves=80,learning_rate=0.05)
    clf.fit(x_train,y_train)
    p_train=clf.predict_proba(x_test)[:,0]    #we predict the probability of product value for next month being 1.
    
    # Now, we predict probability of toggling for each customer in the order they appear in x_test. This is achieved by the following line.
    p_train=(abs(1-np.array(p_train)-np.array(x_test[c+'_lag1']))) 
    p_train=list(p_train)
    
    for id, p in zip(ids,p_train):
        id_preds[id].append([p,c]) #Store the list of pairs( [probability of toggling, product_name] ) for every customer.
# <-----------------------__________MODEL TRAINING and prediction done_________------------------->


# For every customer, sort the id_preds list in decreasing order of probability of toggling and take the first 5 corresponding
# products as the final products to output.
for id in ids:
    id_preds[id].sort(reverse=True) 
    while(len(id_preds[id])>5):
        id_preds[id].pop()
    products=[]    
    for [i,j] in id_preds[id]:
        products.append(j)
    id_preds[id]=products # id_preds[id] now stores the top 5 product names that have highest probability of toggling
    assert(products==id_preds[id])


# <------------------OUTPUT for submission--------------->
customer_order=test_data.ncodpers.values
tests_pred=[]
for id in customer_order:
    tests_pred.append(' '.join(id_preds[id]))

final_output=pd.DataFrame()
final_output=test_data[['ncodpers']]
final_output['changed']=tests_pred
 
final_output.to_csv('output.csv',index=False)

gc.collect()
