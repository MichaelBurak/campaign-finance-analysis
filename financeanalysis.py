#importing in libraries

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#importing dataset to base dataframe in pandas
df = pd.read_csv('CandidateSummaryAction1.csv')

cont = df['net_con']
exp = df['net_ope_exp']
nu_con = cont.isnull().sum() #171 null values of net contributions
nu_exp = exp.isnull().sum() #149 null values

#quick overview
df.head(2)

df.info()

df.columns

#Dropping name and mailing address information.
col = ['can_nam', 'can_str1', 'can_str2', 'can_cit', 'can_sta', 'can_zip']

df.drop(col, axis=1, inplace=True)

#clean up - add more comments on replacing $ + , + converting negatives to be readable
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

#checking - still 171 and 149 null values
nu_con_2 = df['net_con'].isnull().sum()
nu_exp_2 = df['net_ope_exp'].isnull().sum()

#drop all rows where contributions or expenses are null
df = df.dropna(subset = ['net_con'])
df = df.dropna(subset = ['cas_on_han_beg_of_per'])


#filling in NaN of non winners with N and then turning  winner and net_con into numerical value
df['winner'] = df['winner'].fillna('N')
df['winner'] = df['winner'].eq('Y').mul(1)
df['net_con'] = pd.to_numeric(df['net_con'])
df['cas_on_han_beg_of_per'] = pd.to_numeric(df['cas_on_han_beg_of_per'])


#separating out winners for viz
#allwin = df[df.winner == 'Y']
#swin = allwin[allwin.can_off == "S"]
#hwin = allwin[allwin.can_off == "H"]

#net contributions of winning campaigns fir vuz
#allwcon = allwin['net_con']
#swcon = swin['net_con']
#hwcon = hwin['net_con']

#cash on hand for winning campaigns
#allwexpense = allwin['cas_on_han_beg_of_per']
#swexpense = swin['cas_on_han_beg_of_per']
#hwexpense = hwin['cas_on_han_beg_of_per'] 

#plots and EDA 


#Logistic regression modeling 
X = df.iloc[:,[35, 37]].values
y = df.iloc[:, 43].values

#train test split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)



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

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean() #.68
accuracies.std() #.067


#viz train and test results

#Random forest classification modeling

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean() #.924
accuracies.std() # .032

#viz train and test results


#Gradient boosting modeling (Xgboost?)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Grid-search

#Clustering and cluster visualiation