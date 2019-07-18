# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:42:17 2019

@author: Joash
"""

import pandas as pd
import numpy as np

missing_values = ["n/a", "na", "--"]
loan = pd.read_csv(r'XYZCorp_LendingData.txt',

                   na_values = missing_values, sep = '\t')

###df = pd.read_csv("property data.csv", na_values = missing_values)

print(loan)
print(loan.isnull().sum())
loan.shape

#Dropping columns with missing values & unwanted data
loan.drop(["inq_last_12m", "total_cu_tl", "inq_fi", "all_util", "max_bal_bc", "open_rv_24m", "open_rv_12m",
           "il_util", "total_bal_il", "mths_since_rcnt_il", "open_il_24m", "open_il_12m", "open_il_6m",
           "open_acc_6m", "verification_status_joint", "dti_joint", "annual_inc_joint", "mths_since_last_major_derog",
           "desc"], axis=1, inplace=True)

loan.drop(["mths_since_last_record", "mths_since_last_delinq", "id", "member_id",
           "zip_code", "title", "emp_title", "addr_state", "next_pymnt_d"], axis=1, inplace=True)

#,           "recoveries","policy_code","last_credit_pull_d","total_rev_hi_lim"

"""
loan.drop(['initial_list_status','out_prncp_inv'], axis=1, inplace = True)
loan.drop(['term','home_ownership','pymnt_plan','delinq_2yrs','collection_recovery_fee',
           'collections_12_mths_ex_med','policy_code','application_type','acc_now_delinq'], axis=1, inplace = True)
loan.drop(['verification_status','sub_grade','grade','tot_coll_amt'],axis=1, inplace=True)
"""
#Checking the datatypes
loan.dtypes
loan.info

#Filling missing values
for value in ['revol_util', 'tot_coll_amt','total_rev_hi_lim',
              'tot_cur_bal','collections_12_mths_ex_med']:
    loan[value].fillna(loan[value].mean(),inplace=True)

for value in ['last_pymnt_d','emp_length','last_credit_pull_d']:
    loan[value].fillna(loan[value].mode()[0],inplace=True)
    
##Converting issue_d to data type=datetime
loan['issue_d'] =  pd.to_datetime(loan['issue_d'], format='%b-%Y')
print(loan.issue_d)
    
##Creating a copy    
loan_copy = pd.DataFrame.copy(loan)    

loan.dtypes  
##Label Encoder
from sklearn import preprocessing

colobj=[]
for x in loan.columns[:]:
    if loan[x].dtype=='object':
        colobj.append(x)

le={}

le=preprocessing.LabelEncoder()

for x in colobj:
     loan[x]=le.fit_transform(loan[x])


##Splitting the data
loan_train = loan.loc[loan['issue_d'] < '2015-06-01']
loan_test = loan.loc[loan['issue_d'] >= '2015-06-01']

##Removing issue_d since we do not need it
loan_train.drop("issue_d", axis=1, inplace=True)
loan_test.drop("issue_d", axis=1, inplace=True)

##Creating X,Y variables
X_train = loan_train.values[:,:-1]
Y_train = loan_train.values[:,-1]
X_test = loan_test.values[:,:-1]
Y_test = loan_test.values[:,-1]


##Logistic
from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))

print(classifier.coef_)
print(classifier.intercept_)

##Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: \n",classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)

##
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)

y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.46:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

for a in np.arange(0,1,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
          cfm[1,0]," , type 1 error:", cfm[0,1])


##Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_Decision_tree =  DecisionTreeClassifier(criterion = "entropy", random_state = 10)
model_Decision_tree.fit(X_train,Y_train)
Y_pred = model_Decision_tree.predict(X_test)

print(dict(zip(loan_train.columns, model_Decision_tree.feature_importances_)))
#application_type,pub_rec,pymnt_plan,acc_now_delinq
#term,home_ownership,pymnt_plan,delinq_2yrs,collection_recovery_fee,collections_12_mths_ex_med,policy_code,application_type,acc_now_delinq
#initial_list_status,out_prncp_inv
#verification_status,sub_grade,grade,tot_coll_amt

print(Y_pred)
print(list(zip(Y_test,Y_pred)))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: \n",classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)



#predicting using the Gradient_Boosting_Classifier
from sklearn.ensemble import GradientBoostingClassifier
Model_GradientBoosting=GradientBoostingClassifier(random_state=10)
#fit the model on the data and predict the values
#Default estimator = 100
Model_GradientBoosting.fit(X_train,Y_train)
Y_Pred=Model_GradientBoosting.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_Pred))
print(accuracy_score(Y_test,Y_Pred))
print(classification_report(Y_test,Y_Pred))


##RFE
from sklearn.feature_selection import RFE
rfe = RFE(classifier, 30)
model_rfe = rfe.fit(X_train, Y_train)
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ") 
print(list(zip(loan_train.columns, model_rfe.support_)))
print("Feature Ranking: ", model_rfe.ranking_) 

Y_pred=model_rfe.predict(X_test)

#predicting using the Random_Forest_Classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(500)

###
#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)

Y_pred=model_RandomForest.predict(X_test)

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = model_RandomForest.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


### ROC Curve
from sklearn import metrics
y_pred_proba = model_Decision_tree.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()