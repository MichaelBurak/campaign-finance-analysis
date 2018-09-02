#importing in libraries

import pandas as pd 
import matplotlib.pyplot as plt

#importing dataset to base dataframe in pandas
df = pd.read_csv('CandidateSummaryAction1.csv')

contributions = df['net_con']
nu_con = contributions.isnull().sum() #171 null values of net contributions

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

#still 171 null values
nu_con_2 = df['net_con'].isnull().sum()

#drop all rows where contributions are null
df = df.dropna(subset = ['net_con'])

#filling in NaN of non winners with N
df['winner'] = df['winner'].fillna('N')

#separating out winners for viz
#allwin = df[df.winner == 'Y']
#swin = allwin[allwin.can_off == "S"]
#hwin = allwin[allwin.can_off == "H"]

#expenses net of winning campaigns fir vuz
#allwexpense = allwin['net_ope_exp']
#swexpense = swin['net_ope_exp']
#hwexpense = hwin['net_ope_exp']


#Logistic regression modeling 
X = df['net_con']
Y = df['winner']

'''train test split, #feature scaling, fit regression to training set, 
predict test set results, viz train and test results

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