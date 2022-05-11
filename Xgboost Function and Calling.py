#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
from xgboost import plot_importance
import xgboost as xgb
from numpy import mean
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
data=pd.read_csv('C:*******.csv')
data = data.dropna()
first_data_forchart=cp.deepcopy(data)

target = data.pop('******')

#splitting into train and test set
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25,random_state=2)

##hyperparameter tuning
xgb_model = XGBClassifier(use_label_encoder=False)

#listing parameters for searching grid 
params = {
    "colsample_bytree": uniform(0.3, 0.7),
    "gamma": uniform(0.001,.5),
    "learning_rate": uniform(.003,1),
    "max_depth": [1,2,3,4,5,6,7], 
    "n_estimators": randint(100, 150), 
    "subsample": uniform(0.001,0.6),
    "min_child_weight":uniform(0,10),
    "max_delta_step":uniform(0,10),
    "reg_alpha":uniform(0,.9),
    "reg_lambda":uniform(1,4),
    "scale_pos_weight":uniform(1,5),
    "tree_method":['auto', 'exact', 'approx', 'hist', 'gpu_hist']
         }

#searching best parameters
search = RandomizedSearchCV(xgb_model, param_distributions=params, 
                            n_iter=200, random_state=42,cv=3, verbose=1, 
                            n_jobs=1, return_train_score=True)

#training the model to find the optimal parameters
search.fit(X_train, y_train)

#set model with optimal parameters
def tuned_hyper(search_results):
    
    #best params for shorter length later on
    #in case of multiples
    whole_array=np.where(search_results['rank_test_score']==1)
    if len(whole_array[0])>1:
        whole_array=np.where(search_results['rank_test_score']==1)[0]
        
    first_spot=int(whole_array[0])
    winning_combo=search_results['params'][first_spot]
    col_tree=winning_combo['colsample_bytree']
    gams=winning_combo['gamma']
    lr=winning_combo['learning_rate']
    m_dep=winning_combo['max_depth']
    nest=winning_combo['n_estimators']
    s_samp=winning_combo['subsample']
    min_ch=winning_combo['min_child_weight']
    max_del=winning_combo['max_delta_step']
    r_alpha=winning_combo['reg_alpha']
    r_lambda=winning_combo['reg_lambda']
    sc_posw=winning_combo['scale_pos_weight']
    treem=winning_combo['tree_method']
    
    #setting the model up with best parameters
    best_model= XGBClassifier(colsample_bytree=col_tree,
                              gamma=gams,
                              learning_rate=lr,
                              max_depth=m_dep,
                              n_estimators=nest, 
                              use_label_encoder=False,
                              subsample=s_samp,
                              min_child_weight=min_ch,
                              max_delta_step=max_del,
                              reg_alpha=r_alpha,
                              reg_lambda=r_lambda,
                              scale_pos_weight=sc_posw,
                              tree_method=treem 
                             )
    return best_model


#function run
a=tuned_hyper(search.cv_results_)

#fitting best model
a.fit(X_train, y_train)

preds=a.predict(X_test)

accuracy_score(y_test,preds)

# predict probabilities on Test and take probability for class 1([:1])
y_pred_prob_test = a.predict_proba(X_test)[:, 1]
#predict labels on test dataset
y_pred_test = preds
# create onfusion matrix
cm = metrics.confusion_matrix(y_test, y_pred_test)
print("confusion Matrix is :nn",cm)
print("n")
# Accuracy 
print("Accuracy  test dataset:  t", accuracy_score(y_test,y_pred_test))
# ROC- AUC score
print("ROC-AUC score  test dataset:  t", metrics.roc_auc_score(y_test,y_pred_prob_test))
#Precision score
print("precision score  test dataset:  t", metrics.precision_score(y_test,y_pred_test))#,average='micro'))
#Recall Score
print("Recall score  test dataset:  t", metrics.recall_score(y_test,y_pred_test))
#f1 score
print("f1 score  test dataset :  t", f1_score(y_test,y_pred_test))#,average='binary'))

cm = metrics.confusion_matrix(y_test,preds)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#to visualize the importance of each feature/column
plot_importance(a)

#measures several accuracy indicators
print(classification_report(y_test, preds))

