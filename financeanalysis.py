#importing in libraries

import pandas as pd 
import matplotlib.pyplot as plt

#importing dataset to base dataframe in pandas
df = pd.read_csv('CandidateSummaryAction1.csv')

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

df.head(2)

#separating out winners
allwin = df[df.winner == 'Y']
swin = allwin[allwin.can_off == "S"]
hwin = allwin[allwin.can_off == "H"]

#expenses net of winning campaigns
allwexpense = allwin['net_ope_exp']
swexpense = swin['net_ope_exp']
hwexpense = hwin['net_ope_exp']

#Linear regression modeling 

#Random forest modeling 

#Gradient boosting modeling 

#Grid-search --> k folds cross validation model comparison 

#Clustering and visualiation