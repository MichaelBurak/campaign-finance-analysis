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
df = df.dropna(subset = ['net_ope_exp'])

#filling in NaN of non winners with N and then turning  winner and net_con into numerical value
df['winner'] = df['winner'].fillna('N')
df['winner'] = df['winner'].eq('Y').mul(1)
df['net_con'] = pd.to_numeric(df['net_con'])

#separating out winners for viz
#allwin = df[df.winner == 'Y']
#swin = allwin[allwin.can_off == "S"]
#hwin = allwin[allwin.can_off == "H"]

#expenses net of winning campaigns fir vuz
#allwexpense = allwin['net_ope_exp']
#swexpense = swin['net_ope_exp']
#hwexpense = hwin['net_ope_exp']

#Logistic regression modeling 
X = df.iloc[:,[37]].values
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

cm

'''viz train and test results
Random forest modeling 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300)
regressor.fit(X, y)

Predicting a new result with random forest regressor
y_pred = regressor.predict(6.5)

Gradient boosting modeling 

Grid-search --> k folds cross validation model comparison 

Clustering and cluster visualiation
'''