#importing in libraries

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sn

#importing dataset to base dataframe in pandas
df = pd.read_csv('CandidateSummaryAction1.csv')

cont = df['net_con']
cash = df['cas_on_han_beg_of_per']
nu_con = cont.isnull().sum()
nu_cash = cash.isnull().sum()

#quick overview
df.head(2)

df.info()

df.columns

#Dropping name and mailing address information.
col = ['can_nam', 'can_str1', 'can_str2', 'can_cit', 'can_sta', 'can_zip']

df.drop(col, axis=1, inplace=True)

df.columns

#clean up - converts money values to 'float' as a string to be converted in dtype later.
df['tot_dis'] = df['net_ope_exp'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")
df['net_ope_exp'] = df['net_ope_exp'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")
df['net_con'] = df['net_con'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")
df['cas_on_han_beg_of_per']= df['cas_on_han_beg_of_per'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")
df['cas_on_han_clo_of_per']= df['cas_on_han_clo_of_per'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")
df['tot_loa']= df['tot_loa'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")
df['can_loa'] = df['can_loa'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")
df['ind_ite_con']=df['ind_ite_con'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")
df['ind_uni_con']=df['ind_uni_con'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")
df['ind_con']=df['ind_con'].str.replace("$","").str.replace(",","").str.replace("(","-").str.replace(")","")

#checking - still same null value sum
nu_con_2 = cont.isnull().sum()
nu_cash_2 = cash.isnull().sum()

#drop all rows where contributions or cash on hand are null
df = df.dropna(subset = ['net_con'])
df = df.dropna(subset = ['cas_on_han_beg_of_per'])


#filling in NaN of non winners with N and then turning  winner and net_con into numerical value
df['winner'] = df['winner'].fillna('N')
df['winner'] = df['winner'].eq('Y').mul(1)
df['net_con'] = pd.to_numeric(df['net_con'])
df['cas_on_han_beg_of_per'] = pd.to_numeric(df['cas_on_han_beg_of_per'])

df.head(2)
df.info()

#separating out winners for viz
allwin = df[df.winner == 1]
swin = allwin[allwin.can_off == "S"]
hwin = allwin[allwin.can_off == "H"]

#net contributions of winning campaigns fir vuz
allwcon = allwin['net_con']
swcon = swin['net_con']
hwcon = hwin['net_con']

#cash on hand for winning campaigns
allwcash = allwin['cas_on_han_beg_of_per']
swcash = swin['cas_on_han_beg_of_per']
hw = hwin['cas_on_han_beg_of_per'] 



#plots and EDA 


#Logistic regression modeling 
X = df.iloc[:,[35, 37]].values
y = df.iloc[:, 43].values

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Viz matrix
sn.set(font_scale=1.4)#for label size
sn.heatmap(cm, annot=True,annot_kws={"size": 16})# font size


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 3)
acc = accuracies.mean() 
accuracies.std() 

#viz train and test results

#Random forest classification modeling

# Splitting the dataset into the Training set and Test set
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
#sc = StandardScaler()
X_train_rf = sc.fit_transform(X_train_rf)
X_test_rf = sc.transform(X_test_rf)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier_rf.fit(X_train_rf, y_train_rf)

# Predicting the Test set results
y_pred_rf = classifier_rf.predict(X_test_rf)

# Making the Confusion Matrix
cm_rf = confusion_matrix(y_test_rf, y_pred_rf)

#Viz matrix
sn.set(font_scale=1.4)#for label size
sn.heatmap(cm_rf, annot=True,annot_kws={"size": 16})# font size


# Applying k-Fold Cross Validation
accuracies_rf = cross_val_score(estimator = classifier_rf, X = X_train_rf, y = y_train_rf, cv = 3)
acc_rf = accuracies_rf.mean() 
accuracies_rf.std() 

#viz train and test results


#XGboost modeling

# Splitting the dataset into the Training set and Test set
X_train_xg, X_test_xg, y_train_xg, y_test_xg = train_test_split(X, y, test_size = 0.2)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier_xg = XGBClassifier()
classifier_xg.fit(X_train_xg, y_train_xg)

# Predicting the Test set results
y_pred_xg = classifier.predict(X_test_xg)

# Making the Confusion Matrix
cm_xg = confusion_matrix(y_test_xg, y_pred_xg)

#Viz matrix
sn.set(font_scale=1.4)#for label size
sn.heatmap(cm_xg, annot=True,annot_kws={"size": 16})# font size


# Applying k-Fold Cross Validation
accuracies_xg = cross_val_score(estimator = classifier_xg, X = X_train_xg, y = y_train_xg, cv = 3)
acc_xg = accuracies_xg.mean()
accuracies_xg.std()

#viz train and test results

#Viz all the acc off k-folds

obj = ['Logistic', 'Random Forest', 'XGBoost']
y_pos = np.arange(len(obj))
means = [acc, acc_rf, acc_xg]

 
plt.bar(y_pos, means, align='center', alpha=0.5)
plt.xticks(y_pos, obj)
plt.ylabel('Mean of accuracies')
plt.title('K-fold comparison across models')
 
plt.show()

#Tuning XGBoost as most accurate model's hyperparams 

# Applying Grid Search for hyperparams 
from sklearn.model_selection import GridSearchCV
parameters = [{'booster': ['gbtree','dart'], 'gamma':['0','0.1','0.5','1','5']}]
grid_search = GridSearchCV(estimator = classifier_xg,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 3,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Gamma seems to vary based on sample from 0-5