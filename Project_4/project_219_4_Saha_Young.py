#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import numpy as np
from pandas_profiling import ProfileReport
import seaborn
import pycountry_convert as pc
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from statsmodels.regression.linear_model import OLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
from statsmodels.api import add_constant
import itertools
from IPython.display import Image
from skopt import BayesSearchCV
from catboost import CatBoostRegressor
import lightgbm as lgb


# ## Question 1, 2 and 6 

# In[5]:


dataset_folders = ['Bike-Sharing-Dataset/', 'Suicide_Rates/','online_video_dataset/']


# ### Bike Sharing Dataset

# In[6]:


bike = pd.read_csv(dataset_folders[0]+"day.csv")
bike = bike.drop(['instant'],axis=1)
print(bike)
bike_prof = ProfileReport(bike, title="Profile Report for Bike Sharing Dataset")
bike_prof.to_widgets()
bike_prof.to_file('bike_prof.html')


# ### Suicide Rates Dataset

# In[7]:


cols_to_use = ['country','year','sex','age','population',' gdp_for_year ($) ','gdp_per_capita ($)','generation',
                                                               'suicides_no','suicides/100k pop']
suicide = pd.read_csv(dataset_folders[1]+"master.csv",thousands=',',usecols=cols_to_use)[cols_to_use]
print(suicide)
suicide_prof = ProfileReport(suicide, title="Profile Report for Suicide Rates Dataset")
suicide_prof.to_widgets()
suicide_prof.to_file('suicide_prof.html')


# ### Video Transcoding Dataset

# In[8]:


transcode_meas = pd.read_csv(dataset_folders[2]+"transcoding_mesurment.tsv", sep='\t')
transcode_meas = transcode_meas.drop(['id'], axis = 1)
print(transcode_meas)
transcode_meas_prof = ProfileReport(transcode_meas, title="Profile Report for Video Transcoding Dataset")
transcode_meas_prof.to_widgets()
transcode_meas_prof.to_file('transcode_meas_prof.html')


# In[10]:


import matplotlib.pyplot as plt
x = [15.0, 12.0, 29.97,25.0,24.0]
y = [13772, 13764, 13759, 13751, 13738]
plt.hist(x,6, weights=y)
plt.xlabel("Histogram (no. of bins = 6)")
plt.ylabel("Frequency")
plt.savefig("Q2c_o_framerate",dpi=300,bbox_inches='tight')
plt.show()


# ## Question 3

# ### Bike Sharing Dataset

# In[625]:


cat_list = ['season','yr','mnth','holiday','weekday','workingday', 'weathersit']
target_var = ['casual','registered','cnt']
for item in cat_list:
    for target in target_var:
        seaborn.boxplot(x = bike[item],y = bike[target], order=list(set(bike[item])))
        plt.savefig('Q3a'+target+'_'+item+'.png',dpi=300,bbox_inches='tight')
        plt.show()


# ### Suicide Rates Dataset

# In[626]:


cat_list = ['sex','age','generation']
target_var = ['suicides_no','suicides/100k pop']
for item in cat_list:
    for target in target_var:
        ax = seaborn.boxplot(x = suicide[item],y = suicide[target], order=list(set(suicide[item])))
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.savefig('Q3b'+target.replace('/','_')+'_'+item+'.png',dpi=300,bbox_inches='tight')
        plt.show()


# ### Video Transcoding Dataset

# In[9]:


cat_list = ['codec','o_codec']
target_var = ['utime','umem']
for item in cat_list:
    for target in target_var:
        seaborn.boxplot(x = transcode_meas[item],y = transcode_meas[target], order=list(set(transcode_meas[item])))
        plt.savefig('Q3c'+target+'_'+item+'.png',dpi=300,bbox_inches='tight')
        plt.show()


# ## Question 4

# In[628]:


plt.plot(np.arange(1,32,1),bike['cnt'][0:31])
plt.plot(np.arange(1,29,1),bike['cnt'][31:31+28])
plt.plot(np.arange(1,32,1),bike['cnt'][31+28:31+28+31])
plt.plot(np.arange(1,31,1),bike['cnt'][31+28+31:31+28+31+30])
plt.plot(np.arange(1,32,1),bike['cnt'][31+28+31+30:31+28+31+30+31])
plt.plot(np.arange(1,31,1),bike['cnt'][31+28+31+30+31:31+28+31+30+31+30])
plt.legend(['January','February','March','April','May','June'],loc='best')
plt.grid(linestyle=':')
plt.xlabel('Day of Month')
plt.ylabel('Count per day')
plt.title('Bike Sharing (2011)')
plt.savefig('Q4.png',dpi=300,bbox_inches='tight')
plt.show()


# ## Question 5

# In[629]:


country_list = suicide.country.unique()
time_span = []
for item in country_list:
    idx = suicide.country[suicide.country == item].index.tolist()
    idx_range = idx[::len(idx)-1]
    time_span.append(suicide.year[idx_range[1]]-suicide.year[idx_range[0]])
target_country = country_list[sorted(range(len(time_span)), key=lambda i: time_span[i], reverse=True)[:10]]
new_df = suicide[suicide['country'].isin(target_country)]


# In[634]:


seaborn.relplot(data=new_df, x="year", y="suicides/100k pop",hue="sex",kind="line",row='age',col='sex')
plt.savefig('Q5a.png',dpi=300,bbox_inches='tight')
plt.show()


# In[635]:


seaborn.relplot(data=new_df, x="year", y="suicides/100k pop",hue="age",kind="line")
plt.savefig('Q5b.png',dpi=300,bbox_inches='tight')
plt.show()


# In[636]:


seaborn.relplot(data=new_df, x="year", y="suicides/100k pop", hue="sex",kind="line")
plt.savefig('Q5c.png',dpi=300,bbox_inches='tight')
plt.show()


# ## Question 7

# ### Add continent column to Suicides Rates Dataset

# In[13]:


continent_name = []

for i in range(len(suicide)):
    if("Korea" not in suicide.country[i] and "Grenadines" not in suicide.country[i]):
        country_code = pc.country_name_to_country_alpha2(suicide.country[i], cn_name_format="default")
        continent_name.append(pc.country_alpha2_to_continent_code(country_code))
    elif("Korea" in suicide.country[i]):
        continent_name.append('AS')
    elif("Grenadines" in suicide.country[i]):
        continent_name.append('NA')
        
suicide['Continent'] = continent_name
print("List of continents:")
print(set(continent_name))
print(suicide)


# ### Drop unused target variables and convert categorical variables to one-hot encoding

# ### Bike Sharing Dataset

# In[14]:


bike_LR = bike.drop(['dteday','casual','registered'], axis=1)
bike_LR = pd.get_dummies(bike_LR, columns=['season','mnth','weekday','weathersit'], drop_first=False)
print('Dataframe after one-hot encoding categorical variables:')
print(bike_LR)


# ### Suicide Rates Dataset

# In[15]:


suicide_LR = suicide.drop(['country','suicides_no'], axis=1)
suicide_LR = pd.get_dummies(suicide_LR, columns=['sex','age','generation','Continent'], drop_first=False)
print('Dataframe after one-hot encoding categorical variables:')
print(suicide_LR)


# ### Video Transcoding Dataset

# In[16]:


transcode_LR = transcode_meas.drop(['b_size','umem'], axis=1)
transcode_LR = pd.get_dummies(transcode_LR, columns=['codec','o_codec'], drop_first=False)
print('Dataframe after one-hot encoding categorical variables:')
print(transcode_LR)


# ## Question 8

# ### Extract training variables and target values

# In[17]:


XBike = bike_LR.loc[:, bike_LR.columns != 'cnt'].to_numpy()
YBike = bike_LR.cnt
XSuicide = suicide_LR.loc[:, suicide_LR.columns != 'suicides/100k pop'].to_numpy()
YSuicide = suicide_LR["suicides/100k pop"]
XTranscode = transcode_LR.loc[:,  transcode_LR.columns != 'utime'].to_numpy()
YTranscode = transcode_LR["utime"]


# ### Scaling

# In[18]:


bike_scale = StandardScaler()
XBike_S = bike_scale.fit_transform(XBike)
suicide_scale = StandardScaler()
XSuicide_S = suicide_scale.fit_transform(XSuicide)
transcode_scale = StandardScaler()
XTranscode_S = transcode_scale.fit_transform(XTranscode)


# ## Question 9

# ### Testing Feature Selection with Default Hyperparameters for Linear, Ridge and Lasso Regression

# In[19]:


bike_RMSE_MIR = []
bike_RMSE_FR = []
Suicide_RMSE_FR = []
Transcode_RMSE_FR = []

bike_RMSE_MIR_RR = []
bike_RMSE_FR_RR = []
Suicide_RMSE_FR_RR = []
Transcode_RMSE_FR_RR = []

bike_RMSE_MIR_LR = []
bike_RMSE_FR_LR = []
Suicide_RMSE_FR_LR = []
Transcode_RMSE_FR_LR = []


for i in range(1,XBike.shape[1]):
    print('Testing LR, bike dataset for k = ', i)
    XBikeCur_M = SelectKBest(score_func=mutual_info_regression, k=i).fit_transform(XBike, YBike)
    XBikeCur_F = SelectKBest(score_func=f_regression, k=i).fit_transform(XBike, YBike)
    
    BikeOut = cross_validate(LinearRegression(), XBikeCur_M, YBike, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1)
    bike_RMSE_MIR.append(BikeOut['test_neg_root_mean_squared_error'].mean())
    BikeOut = cross_validate(LinearRegression(), XBikeCur_F, YBike, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1)
    bike_RMSE_FR.append(BikeOut['test_neg_root_mean_squared_error'].mean())
    
    print('Testing RR, bike dataset for k = ', i)
    BikeOut = cross_validate(Ridge(), XBikeCur_M, YBike, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1)
    bike_RMSE_MIR_RR.append(BikeOut['test_neg_root_mean_squared_error'].mean())
    BikeOut = cross_validate(Ridge(), XBikeCur_F, YBike, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1)
    bike_RMSE_FR_RR.append(BikeOut['test_neg_root_mean_squared_error'].mean())
    
    print('Testing LaR, bike dataset for k = ', i)
    BikeOut = cross_validate(Lasso(), XBikeCur_M, YBike, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1)
    bike_RMSE_MIR_LR.append(BikeOut['test_neg_root_mean_squared_error'].mean())
    BikeOut = cross_validate(Lasso(), XBikeCur_F, YBike, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1)
    bike_RMSE_FR_LR.append(BikeOut['test_neg_root_mean_squared_error'].mean())


# In[20]:


for i in range(1,XSuicide.shape[1]):
    print('Testing LR, Suicide dataset for k = ', i)
    XSuicideCur_F = SelectKBest(score_func=f_regression, k=i).fit_transform(XSuicide, YSuicide)
    
    SuicideOut = cross_validate(LinearRegression(), XSuicideCur_F, YSuicide, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1)
    Suicide_RMSE_FR.append(SuicideOut['test_neg_root_mean_squared_error'].mean())
    
    print('Testing RR, Suicide dataset for k = ', i)
    SuicideOut = cross_validate(Ridge(), XSuicideCur_F, YSuicide, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1)
    Suicide_RMSE_FR_RR.append(SuicideOut['test_neg_root_mean_squared_error'].mean())
    
    print('Testing LaR, Suicide dataset for k = ', i)
    SuicideOut = cross_validate(Lasso(), XSuicideCur_F, YSuicide, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1)
    Suicide_RMSE_FR_LR.append(SuicideOut['test_neg_root_mean_squared_error'].mean())


# In[21]:


for i in range(1,XTranscode.shape[1]):
    print('Testing LR, Transcode dataset for k = ', i)
    XTranscodeCur_F = SelectKBest(score_func=f_regression, k=i).fit_transform(XTranscode, YTranscode)
    
    TranscodeOut = cross_validate(LinearRegression(), XTranscodeCur_F, YTranscode, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1)
    Transcode_RMSE_FR.append(TranscodeOut['test_neg_root_mean_squared_error'].mean())
    
    print('Testing RR, Transcode dataset for k = ', i)
    TranscodeOut = cross_validate(Ridge(), XTranscodeCur_F, YTranscode, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1)
    Transcode_RMSE_FR_RR.append(TranscodeOut['test_neg_root_mean_squared_error'].mean())
    
    print('Testing LaR, Transcode dataset for k = ', i)
    TranscodeOut = cross_validate(Lasso(), XTranscodeCur_F, YTranscode, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1)
    Transcode_RMSE_FR_LR.append(TranscodeOut['test_neg_root_mean_squared_error'].mean())


# In[22]:


plt.plot(np.arange(1,len(bike_RMSE_MIR)+1,1),bike_RMSE_MIR)
plt.plot(np.arange(1,len(bike_RMSE_FR)+1,1),bike_RMSE_FR)
plt.plot(np.arange(1,len(bike_RMSE_MIR_RR)+1,1),bike_RMSE_MIR_RR,':')
plt.plot(np.arange(1,len(bike_RMSE_FR_RR)+1,1),bike_RMSE_FR_RR,':')
plt.plot(np.arange(1,len(bike_RMSE_MIR_LR)+1,1),bike_RMSE_MIR_LR,'--')
plt.plot(np.arange(1,len(bike_RMSE_FR_LR)+1,1),bike_RMSE_FR_LR,'--')
plt.legend(['MI, Lin. Reg.','F-Score, Lin. Reg.','MI, Ridge Reg.',
            'F-Score, Ridge Reg.','MI, Laso Reg.','F-Score, Lasso Reg.'],loc='best')
plt.grid(linestyle=':')
plt.xlabel('Top k features')
plt.ylabel('Average Test RMSE (-ve) from 10 folds CV')
plt.title('Effect of feature selection on Bike Sharing dataset')
plt.savefig('Q9a.png',dpi=300,bbox_inches='tight')
plt.show()


# In[23]:


plt.plot(np.arange(1,len(Suicide_RMSE_FR)+1,1),Suicide_RMSE_FR)
plt.plot(np.arange(1,len(Suicide_RMSE_FR_RR)+1,1),Suicide_RMSE_FR_RR,':')
plt.plot(np.arange(1,len(Suicide_RMSE_FR_LR)+1,1),Suicide_RMSE_FR_LR,'--')
plt.legend(['F-Score, Lin. Reg.','F-Score, Ridge Reg.','F-Score, Lasso Reg.'],loc='best')
plt.grid(linestyle=':')
plt.xlabel('Top k features')
plt.ylabel('Average Test RMSE (-ve) from 10 folds CV')
plt.title('Effect of feature selection on Suicide Rates dataset')
plt.savefig('Q9b.png',dpi=300,bbox_inches='tight')
plt.show()


# In[24]:


plt.plot(np.arange(1,len(Transcode_RMSE_FR)+1,1),Transcode_RMSE_FR)
plt.plot(np.arange(1,len(Transcode_RMSE_FR_RR)+1,1),Transcode_RMSE_FR_RR,':')
plt.plot(np.arange(1,len(Transcode_RMSE_FR_LR)+1,1),Transcode_RMSE_FR_LR,'--')
plt.legend(['F-Score, Lin. Reg.','F-Score, Ridge Reg.','F-Score, Lasso Reg.'],loc='best')
plt.grid(linestyle=':')
plt.xlabel('Top k features')
plt.ylabel('Average Test RMSE (-ve) from 10 folds CV')
plt.title('Effect of feature selection on Video Transcoding dataset')
plt.savefig('Q9c.png',dpi=300,bbox_inches='tight')
plt.show()


# ## Question 10 to 13

# In[25]:


k_val = 10

XBikeCur_F = SelectKBest(score_func=f_regression, k=k_val).fit_transform(XBike, YBike)
XBikeCur_MIR = SelectKBest(score_func=mutual_info_regression, k=k_val).fit_transform(XBike, YBike)
XBikeCur_FS = SelectKBest(score_func=f_regression, k=k_val).fit_transform(XBike_S, YBike)
XBikeCur_MIRS = SelectKBest(score_func=mutual_info_regression, k=k_val).fit_transform(XBike_S, YBike)

XSuicideCur_F = SelectKBest(score_func=f_regression, k=k_val).fit_transform(XSuicide, YSuicide)
XSuicideCur_FS = SelectKBest(score_func=f_regression, k=k_val).fit_transform(XSuicide_S, YSuicide)

XTranscodeCur_F = SelectKBest(score_func=f_regression, k=k_val).fit_transform(XTranscode, YTranscode)
XTranscodeCur_FS = SelectKBest(score_func=f_regression, k=k_val).fit_transform(XTranscode_S, YTranscode)


# ### Linear Regression

# In[38]:


BikeOut = cross_validate(LinearRegression(), XBikeCur_F, YBike, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1,return_train_score=True)
print('• No standardization, bike dataset, F1, linear regression: Test=',BikeOut['test_neg_root_mean_squared_error'].mean(),',Train=',BikeOut['train_neg_root_mean_squared_error'].mean())
BikeOut = cross_validate(LinearRegression(), XBikeCur_MIR, YBike, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1,return_train_score=True)
print('• No standardization, bike dataset, MI, linear regression: Test=',BikeOut['test_neg_root_mean_squared_error'].mean(),',Train=',BikeOut['train_neg_root_mean_squared_error'].mean())
BikeOut = cross_validate(LinearRegression(), XBikeCur_FS, YBike, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1,return_train_score=True)
print('• Standardization, bike dataset, F1, linear regression:  Test=',BikeOut['test_neg_root_mean_squared_error'].mean(),',Train=',BikeOut['train_neg_root_mean_squared_error'].mean())
BikeOut = cross_validate(LinearRegression(), XBikeCur_MIRS, YBike, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1,return_train_score=True)
print('• Standardization, bike dataset, MI, linear regression: Test=',BikeOut['test_neg_root_mean_squared_error'].mean(),',Train=',BikeOut['train_neg_root_mean_squared_error'].mean())

SuicideOut = cross_validate(LinearRegression(), XSuicideCur_F, YSuicide, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1,return_train_score=True)
print('• No standardization, Suicide dataset, F1, linear regression: Test=',SuicideOut['test_neg_root_mean_squared_error'].mean(),',Train=',SuicideOut['train_neg_root_mean_squared_error'].mean())
SuicideOut = cross_validate(LinearRegression(), XSuicideCur_FS, YSuicide, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1,return_train_score=True)
print('• Standardization, Suicide dataset, F1, linear regression: Test=',SuicideOut['test_neg_root_mean_squared_error'].mean(),',Train=',SuicideOut['train_neg_root_mean_squared_error'].mean())

TranscodeOut = cross_validate(LinearRegression(), XTranscodeCur_F, YTranscode, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1,return_train_score=True)
print('• No standardization, Transcode dataset, F1, linear regression: Test=',TranscodeOut['test_neg_root_mean_squared_error'].mean(),',Train=',TranscodeOut['train_neg_root_mean_squared_error'].mean())
TranscodeOut = cross_validate(LinearRegression(), XTranscodeCur_FS, YTranscode, scoring=['neg_root_mean_squared_error'], cv=10,n_jobs=-1,return_train_score=True)
print('• Standardization, Transcode dataset, F1, linear regression: Test=',TranscodeOut['test_neg_root_mean_squared_error'].mean(),',Train=',TranscodeOut['train_neg_root_mean_squared_error'].mean())


# ### Ridge Regression

# In[39]:


pipe_RR = Pipeline([('model', Ridge(random_state=42))])
param_grid = {
    'model__alpha': [10.0**x for x in np.arange(-4,4)]
}


# In[40]:


print("Testing bike..\n")
gridBikeRR_F = GridSearchCV(pipe_RR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XBikeCur_F, YBike)
gridBikeRR_FS = GridSearchCV(pipe_RR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XBikeCur_FS, YBike)
gridBikeRR_M = GridSearchCV(pipe_RR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XBikeCur_MIR, YBike)
gridBikeRR_MS = GridSearchCV(pipe_RR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XBikeCur_MIRS, YBike)
print("Testing suicide..\n")
gridSuicideRR_F = GridSearchCV(pipe_RR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XSuicideCur_F, YSuicide)
gridSuicideRR_FS = GridSearchCV(pipe_RR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XSuicideCur_FS, YSuicide)
print("Testing transcoding..\n")
gridTranscodeRR_F = GridSearchCV(pipe_RR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XTranscodeCur_F, YTranscode)
gridTranscodeRR_FS = GridSearchCV(pipe_RR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XTranscodeCur_FS, YTranscode)


# In[54]:


print('• No standardization, bike dataset, F1, ridge reg., Test RMSE:',gridBikeRR_F.best_score_,
      ',alpha:',gridBikeRR_F.best_params_,'train RMSE',max(gridBikeRR_F.cv_results_['mean_train_score']))
print('• Standardization, bike dataset, F1, ridge reg., Test RMSE:',gridBikeRR_FS.best_score_,
      ',alpha:',gridBikeRR_FS.best_params_,'train RMSE',max(gridBikeRR_FS.cv_results_['mean_train_score']))
print('• No standardization, bike dataset, MI, ridge reg., Test RMSE:',gridBikeRR_M.best_score_,
      ',alpha:',gridBikeRR_F.best_params_,'train RMSE',max(gridBikeRR_M.cv_results_['mean_train_score']))
print('• Standardization, bike dataset, MI, ridge reg., Test RMSE:',gridBikeRR_MS.best_score_,
      ',alpha:',gridBikeRR_MS.best_params_,'train RMSE',max(gridBikeRR_MS.cv_results_['mean_train_score']))
print('• No standardization, suicide dataset, F1, ridge reg., Test RMSE:',gridSuicideRR_F.best_score_,
      ',alpha:',gridSuicideRR_F.best_params_,'train RMSE',max(gridSuicideRR_F.cv_results_['mean_train_score']))
print('• Standardization, suicide dataset, F1, ridge reg., Test RMSE:',gridSuicideRR_FS.best_score_,
      ',alpha:',gridSuicideRR_FS.best_params_,'train RMSE',max(gridSuicideRR_FS.cv_results_['mean_train_score']))
print('• No standardization, transcoding dataset, F1, ridge reg., Test RMSE:',gridTranscodeRR_F.best_score_,
      ',alpha:',gridTranscodeRR_F.best_params_,'train RMSE',max(gridTranscodeRR_F.cv_results_['mean_train_score']))
print('• Standardization, transcoding dataset, F1, ridge reg., Test RMSE:',gridTranscodeRR_FS.best_score_,
      ',alpha:',gridTranscodeRR_FS.best_params_,'train RMSE',max(gridTranscodeRR_FS.cv_results_['mean_train_score']))


# ### Lasso Regression

# In[55]:


pipe_LAR = Pipeline([('model', Lasso(random_state=42))])
param_grid = {
    'model__alpha': [10.0**x for x in np.arange(-4,4)]
}


# In[56]:


print("Testing bike..\n")
gridBikeLAR_F = GridSearchCV(pipe_LAR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XBikeCur_F, YBike)
gridBikeLAR_FS = GridSearchCV(pipe_LAR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XBikeCur_FS, YBike)
gridBikeLAR_M = GridSearchCV(pipe_LAR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XBikeCur_MIR, YBike)
gridBikeLAR_MS = GridSearchCV(pipe_LAR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XBikeCur_MIRS, YBike)
print("Testing suicide..\n")
gridSuicideLAR_F = GridSearchCV(pipe_LAR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XSuicideCur_F, YSuicide)
gridSuicideLAR_FS = GridSearchCV(pipe_LAR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XSuicideCur_FS, YSuicide)
print("Testing transcoding..\n")
gridTranscodeLAR_F = GridSearchCV(pipe_LAR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XTranscodeCur_F, YTranscode)
gridTranscodeLAR_FS = GridSearchCV(pipe_LAR, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XTranscodeCur_FS, YTranscode)


# In[57]:


print('• No standardization, bike dataset, F1, lasso reg., Test RMSE:',gridBikeLAR_F.best_score_,
      ',alpha:',gridBikeLAR_F.best_params_,'train RMSE',max(gridBikeLAR_F.cv_results_['mean_train_score']))
print('• Standardization, bike dataset, F1, lasso reg., Test RMSE:',gridBikeLAR_FS.best_score_,
      ',alpha:',gridBikeLAR_FS.best_params_,'train RMSE',max(gridBikeLAR_FS.cv_results_['mean_train_score']))
print('• No standardization, bike dataset, MI, lasso reg., Test RMSE:',gridBikeLAR_M.best_score_,
      ',alpha:',gridBikeLAR_F.best_params_,'train RMSE',max(gridBikeLAR_F.cv_results_['mean_train_score']))
print('• Standardization, bike dataset, MI, lasso reg., Test RMSE:',gridBikeLAR_MS.best_score_,
      ',alpha:',gridBikeLAR_MS.best_params_,'train RMSE',max(gridBikeLAR_MS.cv_results_['mean_train_score']))
print('• No standardization, suicide dataset, F1, lasso reg., Test RMSE:',gridSuicideLAR_F.best_score_,
      ',alpha:',gridSuicideLAR_F.best_params_,'train RMSE',max(gridSuicideLAR_F.cv_results_['mean_train_score']))
print('• Standardization, suicide dataset, F1, lasso reg., Test RMSE:',gridSuicideLAR_FS.best_score_,
      ',alpha:',gridSuicideLAR_FS.best_params_,'train RMSE',max(gridSuicideLAR_FS.cv_results_['mean_train_score']))
print('• No standardization, transcoding dataset, F1, lasso reg., Test RMSE:',gridTranscodeLAR_F.best_score_,
      ',alpha:',gridTranscodeLAR_F.best_params_,'train RMSE',max(gridTranscodeLAR_F.cv_results_['mean_train_score']))
print('• Standardization, transcoding dataset, F1, lasso reg., Test RMSE:',gridTranscodeLAR_FS.best_score_,
      ',alpha:',gridTranscodeLAR_FS.best_params_,'train RMSE',max(gridTranscodeLAR_FS.cv_results_['mean_train_score']))


# ### p-value example

# In[58]:


p_ex = OLS(YSuicide, suicide_LR.loc[:, suicide_LR.columns != 'suicides/100k pop']).fit()
print(p_ex.pvalues.sort_values(ascending=True))


# ## Question 14 to 16

# ### Finding optimal degree

# In[59]:


degree_list = np.arange(1,11,1)

pipe_PR_bike = Pipeline([
    ('PR', PolynomialFeatures()),
    ('model', Ridge(random_state=42))
])

pipe_PR_suicide = Pipeline([
    ('PR', PolynomialFeatures()),
    ('model', Ridge(random_state=42))
])

pipe_PR_transcode = Pipeline([
    ('PR', PolynomialFeatures()),
    ('model', Ridge(random_state=42))
])

param_grid_PR = {
    'PR__degree': degree_list,
    'model__alpha': [10.0**x for x in np.arange(-4,4)]
    
}


# In[60]:


gridbike_PR = GridSearchCV(pipe_PR_bike, param_grid=param_grid_PR, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XBikeCur_F,YBike)


# In[69]:


poly_result = pd.DataFrame(gridbike_PR.cv_results_)[['mean_test_score','mean_train_score','param_PR__degree','param_model__alpha']]
bike_score = []
bike_train = []
bike_alpha = []
for i in degree_list:
    bike_score.append((poly_result.loc[poly_result['param_PR__degree'] == i]).max().mean_test_score)
    bike_train.append((poly_result.loc[poly_result['param_PR__degree'] == i]).max().mean_train_score)
    bike_alpha.append(float(poly_result['param_model__alpha'][
        (poly_result.loc[poly_result['param_PR__degree'] == i])
        [['mean_test_score']].idxmax()].to_numpy()))
plt.plot(degree_list,bike_score)
plt.grid(linestyle=':')
plt.xlabel('Polynomial Degree')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of Polynomial Degree on Bike Sharing Dataset (Test)')
plt.savefig('Q15a.png',dpi=300,bbox_inches='tight')
plt.show()
plt.plot(degree_list,bike_train,'r')
plt.grid(linestyle=':')
plt.xlabel('Polynomial Degree')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of Polynomial Degree on Bike Sharing Dataset (Train)')
plt.savefig('Q15b.png',dpi=300,bbox_inches='tight')
plt.show()


# In[70]:


degree_list = np.arange(1,6,1)
param_grid_PR = {
    'PR__degree': degree_list,
    'model__alpha': [10.0**x for x in np.arange(-4,4)]
    
}

gridSuicide_PR = GridSearchCV(pipe_PR_suicide, param_grid=param_grid_PR, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XSuicideCur_F,YSuicide)


# In[71]:


poly_result = pd.DataFrame(gridSuicide_PR.cv_results_)[['mean_test_score','mean_train_score','param_PR__degree','param_model__alpha']]
suicide_score = []
suicide_train = []
suicide_alpha = []
for i in degree_list:
    suicide_score.append((poly_result.loc[poly_result['param_PR__degree'] == i]).max().mean_test_score)
    suicide_train.append((poly_result.loc[poly_result['param_PR__degree'] == i]).max().mean_train_score)
    suicide_alpha.append(float(poly_result['param_model__alpha'][
        (poly_result.loc[poly_result['param_PR__degree'] == i])
        [['mean_test_score']].idxmax()].to_numpy()))
plt.plot(degree_list,suicide_score)
plt.grid(linestyle=':')
plt.xlabel('Polynomial Degree')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of Polynomial Degree on Suicide Rates Dataset (Test)')
plt.savefig('Q15c.png',dpi=300,bbox_inches='tight')
plt.show()
plt.plot(degree_list,suicide_train,'r')
plt.grid(linestyle=':')
plt.xlabel('Polynomial Degree')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of Polynomial Degree on Suicide Rates Dataset (Train)')
plt.savefig('Q15d.png',dpi=300,bbox_inches='tight')
plt.show()


# In[72]:


degree_list = np.arange(1,5,1)
param_grid_PR = {
    'PR__degree': degree_list,
    'model__alpha': [10.0**x for x in np.arange(-4,4)]
    
}

gridTranscode_PR = GridSearchCV(pipe_PR_transcode, param_grid=param_grid_PR, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XTranscodeCur_F,YTranscode)


# In[73]:


poly_result = pd.DataFrame(gridTranscode_PR.cv_results_)[['mean_test_score','mean_train_score','param_PR__degree','param_model__alpha']]
transcode_score = []
transcode_train = []
transcode_alpha = []
for i in degree_list:
    transcode_score.append((poly_result.loc[poly_result['param_PR__degree'] == i]).max().mean_test_score)
    transcode_train.append((poly_result.loc[poly_result['param_PR__degree'] == i]).max().mean_train_score)
    transcode_alpha.append(float(poly_result['param_model__alpha'][
        (poly_result.loc[poly_result['param_PR__degree'] == i])
        [['mean_test_score']].idxmax()].to_numpy()))
plt.plot(degree_list,transcode_score)
plt.grid(linestyle=':')
plt.xlabel('Polynomial Degree')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of Polynomial Degree on Video Transcoding Dataset (Test)')
plt.savefig('Q15e.png',dpi=300,bbox_inches='tight')
plt.show()
plt.plot(degree_list,transcode_train,'r')
plt.grid(linestyle=':')
plt.xlabel('Polynomial Degree')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of Polynomial Degree on Video Transcoding Dataset (Train)')
plt.savefig('Q15f.png',dpi=300,bbox_inches='tight')
plt.show()


# ### Most Salient Features

# In[74]:


chY = SelectKBest(score_func=f_regression, k=10)
XTranscode_Test = chY.fit_transform(bike_LR.loc[:, bike_LR.columns != 'cnt'], bike_LR.cnt)
column_names = bike_LR.loc[:, bike_LR.columns != 'cnt'].columns[chY.get_support()]

b_params = gridbike_PR.best_estimator_.get_params()
b_coefs = b_params['model'].coef_
b_feature_name = list(column_names)
b_names = b_params['PR'].get_feature_names(b_feature_name)
b_sorted_indice = np.argsort(-abs(b_coefs))
salient_features =[b_names[i] for i in b_sorted_indice[:5]]
print ('Top 5 Salient features (bike):',salient_features)


# In[75]:


chY = SelectKBest(score_func=f_regression, k=10)
XTranscode_Test = chY.fit_transform(suicide_LR.loc[:, suicide_LR.columns != 'suicides/100k pop'], suicide_LR["suicides/100k pop"])
column_names = suicide_LR.loc[:, suicide_LR.columns != 'suicides/100k pop'].columns[chY.get_support()]

s_params = gridSuicide_PR.best_estimator_.get_params()
s_coefs = s_params['model'].coef_
s_feature_name = list(column_names)
s_names = s_params['PR'].get_feature_names(s_feature_name)
s_sorted_indice = np.argsort(-abs(s_coefs))
salient_feature =[s_names[i] for i in s_sorted_indice[:5]]
print ('Top 5 Salient features (suicide):',salient_feature)


# In[76]:


chY = SelectKBest(score_func=f_regression, k=10)
XTranscode_Test = chY.fit_transform(transcode_LR.loc[:,  transcode_LR.columns != 'utime'], transcode_LR["utime"])
column_names = transcode_LR.loc[:,  transcode_LR.columns != 'utime'].columns[chY.get_support()]

t_params = gridTranscode_PR.best_estimator_.get_params()
t_coefs = t_params['model'].coef_
t_feature_name = list(column_names)
t_names = t_params['PR'].get_feature_names(t_feature_name)
t_sorted_indice = np.argsort(-abs(t_coefs))
salient_feature =[t_names[i] for i in t_sorted_indice[:5]]
print ('Top 5 Salient features (transcoding):',salient_feature)


# ### Testing Inverse Features

# In[77]:


chY = SelectKBest(score_func=f_regression, k=10)
XTranscode_Test = chY.fit_transform(transcode_LR.loc[:,  transcode_LR.columns != 'utime'], transcode_LR["utime"])
column_names = transcode_LR.loc[:,  transcode_LR.columns != 'utime'].columns[chY.get_support()]
XTranscode_Test = pd.DataFrame(XTranscode_Test,columns=list(column_names))
print(list(column_names))


# In[78]:


inv_feat = np.divide(np.prod(XTranscode_Test[['o_width','o_height']],axis=1),XTranscode_Test['o_bitrate'])
XTranscode_Test['inv_feat'] = inv_feat
print(XTranscode_Test)


# In[79]:


degree_list = [2]
param_grid_PR = {
    'PR__degree': degree_list,
    'model__alpha': [transcode_alpha[1]]
    
}

gridTranscode_PR_test = GridSearchCV(pipe_PR_transcode, param_grid=param_grid_PR, cv=10, n_jobs=-1, verbose=1, 
                     scoring='neg_root_mean_squared_error', return_train_score=True).fit(XTranscode_Test ,YTranscode)


# In[80]:


print('Average Test RMSE (-ve) without inverse feature (degree = 2):',transcode_score[1])
print('Average Test RMSE (-ve) with inverse feature (degree = 2):',gridTranscode_PR_test.best_score_)


# ## Question 17 to 20

# In[163]:


a_list = [10,20,30,50]
all_combinations = []
for r in range(len(a_list) + 1):
    combinations_object = itertools.combinations_with_replacement(a_list, r)
    combinations_list = list(combinations_object)
    all_combinations += combinations_list
all_combinations = all_combinations[1:]

pipe_NN = Pipeline([
    ('model', MLPRegressor(random_state=42,max_iter=1000))
])
                         
param_grid_NN = {
    'model__hidden_layer_sizes': all_combinations,
    'model__alpha': [10.0**x for x in np.arange(-4,2)],
    'model__activation': ['logistic', 'tanh', 'relu']   
}


# In[164]:


print(all_combinations)


# In[165]:


gridbike_NN = GridSearchCV(pipe_NN, param_grid=param_grid_NN, cv=10, n_jobs=-1, verbose=1, 
                                  scoring='neg_root_mean_squared_error', return_train_score=True).fit(XBikeCur_F, YBike)


# In[182]:


poly_result = pd.DataFrame(gridbike_NN.cv_results_)[['mean_test_score','mean_train_score','param_model__alpha','param_model__activation','param_model__hidden_layer_sizes']]
print('Best parameters (bike):',gridbike_NN.best_params_,',Test RMSE:',gridbike_NN.best_score_)
print('Train RMSE:',max(poly_result.mean_train_score))


# In[183]:


gridSuicide_NN = GridSearchCV(pipe_NN, param_grid=param_grid_NN, cv=10, n_jobs=-1, verbose=1, 
                                  scoring='neg_root_mean_squared_error', return_train_score=True).fit(XSuicideCur_F, YSuicide)


# In[184]:


poly_result = pd.DataFrame(gridSuicide_NN.cv_results_)[['mean_test_score','mean_train_score','param_model__alpha','param_model__activation','param_model__hidden_layer_sizes']]
print('Best parameters (suicide):',gridSuicide_NN.best_params_,',Test RMSE:',gridSuicide_NN.best_score_)
print('Train RMSE:',max(poly_result.mean_train_score))


# In[185]:


gridTranscode_NN = GridSearchCV(pipe_NN, param_grid=param_grid_NN, cv=10, n_jobs=-1, verbose=1, 
                                  scoring='neg_root_mean_squared_error', return_train_score=True).fit(XTranscodeCur_F, YTranscode)


# In[189]:


poly_result = pd.DataFrame(gridTranscode_NN.cv_results_)[['mean_test_score','mean_train_score','param_model__alpha','param_model__activation','param_model__hidden_layer_sizes']]
print('Best parameters (suicide):',gridTranscode_NN.best_params_,',Test RMSE:',gridTranscode_NN.best_score_)
print('Train RMSE:',max(poly_result.mean_train_score))


# ## Question 21 to 23, 28

# ### Optimal and effect of each hyperparameter

# In[249]:


pipe_RF = Pipeline([
    ('model', RandomForestRegressor(random_state=42, oob_score=True))
])

param_grid_RF = {
    'model__max_features': np.arange(1,11,1),
    'model__n_estimators': np.arange(10, 210, 10),
    'model__max_depth': np.arange(1, 20, 1)
    
}


# In[250]:


gridbike_RF = GridSearchCV(pipe_RF, param_grid=param_grid_RF, cv=10, n_jobs=-1, verbose=1, 
                                  scoring='neg_root_mean_squared_error', return_train_score=True).fit(XBikeCur_F, YBike)


# In[251]:


poly_result = pd.DataFrame(gridbike_RF.cv_results_)[['mean_test_score','mean_train_score','param_model__max_features','param_model__n_estimators','param_model__max_depth']]
print('Best parameters (bike):',gridbike_RF.best_params_,',Test RMSE:',gridbike_RF.best_score_)
print('Train RMSE:',max(poly_result.mean_train_score))


# In[252]:


max_features = np.arange(1,11,1).reshape(10)
n_estimators = np.arange(10, 210, 10).reshape(20)
max_depth = np.arange(1, 20, 1).reshape(19)

bike_score = list((poly_result[(poly_result['param_model__max_depth'] == 8) & (poly_result['param_model__max_features'] == 4)]).mean_test_score)
bike_train = list((poly_result[(poly_result['param_model__max_depth'] == 8) & (poly_result['param_model__max_features'] == 4)]).mean_train_score)
plt.plot(n_estimators,bike_score)
plt.grid(linestyle=':')
plt.xlabel('Number of trees (depth of each tree = 8, max number of features = 4)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of number of trees on bike dataset (Test)')
plt.savefig('Q21a.png',dpi=300,bbox_inches='tight')
plt.show()
plt.plot(n_estimators,bike_train,'r')
plt.grid(linestyle=':')
plt.xlabel('Number of trees (depth of each tree = 8, max number of features = 4)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of number of trees on bike dataset (Train)')
plt.savefig('Q21b.png',dpi=300,bbox_inches='tight')
plt.show()

bike_score = list((poly_result[(poly_result['param_model__max_depth'] == 8) & (poly_result['param_model__n_estimators'] == 20)]).mean_test_score)
bike_train = list((poly_result[(poly_result['param_model__max_depth'] == 8) & (poly_result['param_model__n_estimators'] == 20)]).mean_train_score)
plt.plot(max_features,bike_score)
plt.grid(linestyle=':')
plt.xlabel('Number of max features (depth of each tree = 8, number of trees = 20)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of number of max features on bike dataset (Test)')
plt.savefig('Q21c.png',dpi=300,bbox_inches='tight')
plt.show()
plt.plot(max_features,bike_train,'r')
plt.grid(linestyle=':')
plt.xlabel('Number of max features (depth of each tree = 8, number of trees = 20)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of number of max features on bike dataset (Train)')
plt.savefig('Q21d.png',dpi=300,bbox_inches='tight')
plt.show()

bike_score = list((poly_result[(poly_result['param_model__max_features'] == 4) & (poly_result['param_model__n_estimators'] == 20)]).mean_test_score)
bike_train = list((poly_result[(poly_result['param_model__max_features'] == 4) & (poly_result['param_model__n_estimators'] == 20)]).mean_train_score)
plt.plot(max_depth,bike_score)
plt.grid(linestyle=':')
plt.xlabel('Depth of each tree (max number of features = 4, number of trees = 20)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of depth of each tree on bike dataset (Test)')
plt.savefig('Q21e.png',dpi=300,bbox_inches='tight')
plt.show()
plt.plot(max_depth,bike_train,'r')
plt.grid(linestyle=':')
plt.xlabel('Depth of each tree  (max number of features = 4, number of trees = 20)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of depth of each tree on bike dataset (Train)')
plt.savefig('Q21f.png',dpi=300,bbox_inches='tight')
plt.show()


# In[263]:


gridSuicide_RF = GridSearchCV(pipe_RF, param_grid=param_grid_RF, cv=10, n_jobs=-1, verbose=1, 
                                  scoring='neg_root_mean_squared_error', return_train_score=True).fit(XSuicideCur_F, YSuicide)


# In[264]:


poly_result = pd.DataFrame(gridSuicide_RF.cv_results_)[['mean_test_score','mean_train_score','param_model__max_features','param_model__n_estimators','param_model__max_depth']]
print('Best parameters (suicide):',gridSuicide_RF.best_params_,',Test RMSE:',gridSuicide_RF.best_score_)
print('Train RMSE:',max(poly_result.mean_train_score))


# In[265]:


suicide_score = list((poly_result[(poly_result['param_model__max_depth'] == 6) & (poly_result['param_model__max_features'] == 3)]).mean_test_score)
suicide_train = list((poly_result[(poly_result['param_model__max_depth'] == 6) & (poly_result['param_model__max_features'] == 3)]).mean_train_score)
plt.plot(n_estimators,suicide_score)
plt.grid(linestyle=':')
plt.xlabel('Number of trees (depth of each tree = 6, max number of features = 3)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of number of trees on suicide dataset (Test)')
plt.savefig('Q21g.png',dpi=300,bbox_inches='tight')
plt.show()
plt.plot(n_estimators,suicide_train,'r')
plt.grid(linestyle=':')
plt.xlabel('Number of trees (depth of each tree = 6, max number of features = 3)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of number of trees on suicide dataset (Train)')
plt.savefig('Q21h.png',dpi=300,bbox_inches='tight')
plt.show()

suicide_score = list((poly_result[(poly_result['param_model__max_depth'] == 6) & (poly_result['param_model__n_estimators'] == 10)]).mean_test_score)
suicide_train = list((poly_result[(poly_result['param_model__max_depth'] == 6) & (poly_result['param_model__n_estimators'] == 10)]).mean_train_score)
plt.plot(max_features,suicide_score)
plt.grid(linestyle=':')
plt.xlabel('Number of max features (depth of each tree = 6, number of trees = 10)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of number of max features on suicide dataset (Test)')
plt.savefig('Q21i.png',dpi=300,bbox_inches='tight')
plt.show()
plt.plot(max_features,suicide_train,'r')
plt.grid(linestyle=':')
plt.xlabel('Number of max features (depth of each tree = 6, number of trees = 10)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of number of max features on suicide dataset (Train)')
plt.savefig('Q21j.png',dpi=300,bbox_inches='tight')
plt.show()

suicide_score = list((poly_result[(poly_result['param_model__max_features'] == 3) & (poly_result['param_model__n_estimators'] == 10)]).mean_test_score)
suicide_train = list((poly_result[(poly_result['param_model__max_features'] == 3) & (poly_result['param_model__n_estimators'] == 10)]).mean_train_score)
plt.plot(max_depth,suicide_score)
plt.grid(linestyle=':')
plt.xlabel('Depth of each tree (max number of features = 3, number of trees = 10)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of depth of each tree on suicide dataset (Test)')
plt.savefig('Q21k.png',dpi=300,bbox_inches='tight')
plt.show()
plt.plot(max_depth,suicide_train,'r')
plt.grid(linestyle=':')
plt.xlabel('Depth of each tree  (max number of features = 3, number of trees = 10)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of depth of each tree on suicide dataset (Train)')
plt.savefig('Q21l.png',dpi=300,bbox_inches='tight')
plt.show()


# In[267]:


param_grid_RF = {
    'model__max_features': np.arange(1,11,1),
    'model__n_estimators': np.arange(10, 50, 10),
    'model__max_depth': np.arange(1, 20, 1)
    
}
gridTranscode_RF = GridSearchCV(pipe_RF, param_grid=param_grid_RF, cv=10, n_jobs=-1, verbose=1, 
                                  scoring='neg_root_mean_squared_error', return_train_score=True).fit(XTranscodeCur_F, YTranscode)


# In[268]:


poly_result = pd.DataFrame(gridTranscode_RF.cv_results_)[['mean_test_score','mean_train_score','param_model__max_features','param_model__n_estimators','param_model__max_depth']]
print('Best parameters (transcode):',gridTranscode_RF.best_params_,',Test RMSE:',gridTranscode_RF.best_score_)
print('Train RMSE:',max(poly_result.mean_train_score))


# In[269]:


max_features = np.arange(1,11,1).reshape(10)
n_estimators = np.arange(10, 50, 10).reshape(4)
max_depth = np.arange(1, 20, 1).reshape(19)

transcode_score = list((poly_result[(poly_result['param_model__max_depth'] == 13) & (poly_result['param_model__max_features'] == 4)]).mean_test_score)
transcode_train = list((poly_result[(poly_result['param_model__max_depth'] == 13) & (poly_result['param_model__max_features'] == 4)]).mean_train_score)
plt.plot(n_estimators,transcode_score)
plt.grid(linestyle=':')
plt.xlabel('Number of trees (depth of each tree = 13, max number of features = 4)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of number of trees on transcode dataset (Test)')
plt.savefig('Q21m.png',dpi=300,bbox_inches='tight')
plt.show()
plt.plot(n_estimators,transcode_train,'r')
plt.grid(linestyle=':')
plt.xlabel('Number of trees (depth of each tree = 13, max number of features = 4)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of number of trees on transcode dataset (Train)')
plt.savefig('Q21n.png',dpi=300,bbox_inches='tight')
plt.show()

transcode_score = list((poly_result[(poly_result['param_model__max_depth'] == 13) & (poly_result['param_model__n_estimators'] == 40)]).mean_test_score)
transcode_train = list((poly_result[(poly_result['param_model__max_depth'] == 13) & (poly_result['param_model__n_estimators'] == 40)]).mean_train_score)
plt.plot(max_features,transcode_score)
plt.grid(linestyle=':')
plt.xlabel('Number of max features (depth of each tree = 13, number of trees = 40)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of number of max features on transcode dataset (Test)')
plt.savefig('Q21o.png',dpi=300,bbox_inches='tight')
plt.show()
plt.plot(max_features,transcode_train,'r')
plt.grid(linestyle=':')
plt.xlabel('Number of max features (depth of each tree = 13, number of trees = 40)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of number of max features on transcode dataset (Train)')
plt.savefig('Q21p.png',dpi=300,bbox_inches='tight')
plt.show()

transcode_score = list((poly_result[(poly_result['param_model__max_features'] == 4) & (poly_result['param_model__n_estimators'] == 40)]).mean_test_score)
transcode_train = list((poly_result[(poly_result['param_model__max_features'] == 4) & (poly_result['param_model__n_estimators'] == 40)]).mean_train_score)
plt.plot(max_depth,transcode_score)
plt.grid(linestyle=':')
plt.xlabel('Depth of each tree (max number of features = 4, number of trees = 40)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of depth of each tree on transcode dataset (Test)')
plt.savefig('Q21q.png',dpi=300,bbox_inches='tight')
plt.show()
plt.plot(max_depth,transcode_train,'r')
plt.grid(linestyle=':')
plt.xlabel('Depth of each tree  (max number of features = 4, number of trees = 40)')
plt.ylabel('Average RMSE (-ve) from 10 folds CV')
plt.title('Effect of depth of each tree on transcode dataset (Train)')
plt.savefig('Q21r.png',dpi=300,bbox_inches='tight')
plt.show()


# ### OOB Error for best models

# In[270]:


print('OOB, Bike:',RandomForestRegressor(random_state=42,max_depth=8,
                                         max_features=4, n_estimators=20, oob_score=True).fit(XBikeCur_F,YBike).oob_score_)
print('OOB, Suicide:',RandomForestRegressor(random_state=42,max_depth=6, max_features=3, n_estimators=10, oob_score=True).fit(XSuicideCur_F,YSuicide).oob_score_)
print('OOB, Transcode:',RandomForestRegressor(random_state=42,max_depth=13, max_features=4, n_estimators=40, oob_score=True).fit(XTranscodeCur_F,YTranscode).oob_score_)


# ### Tree Visualization

# In[272]:


vis_tree = RandomForestRegressor(random_state=42,max_depth=4, max_features=3, n_estimators=10).fit(XSuicideCur_F,YSuicide)


# In[281]:


chY = SelectKBest(score_func=f_regression, k=10)
XTranscode_Test = chY.fit_transform(suicide_LR.loc[:, suicide_LR.columns != 'suicides/100k pop'], suicide_LR["suicides/100k pop"])
column_names = suicide_LR.loc[:, suicide_LR.columns != 'suicides/100k pop'].columns[chY.get_support()]


# In[288]:


tree = vis_tree.estimators_[1]
export_graphviz(tree, out_file = 'tree.dot', feature_names = column_names, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
Image(graph.create_png())


# In[290]:


graph.write_png("Q23.png")


# ## Question 24 to 26

# In[336]:


opt = BayesSearchCV(
    lgb.LGBMRegressor(random_state=42,verbose=1,n_jobs=-1),
    {
        'boosting_type': ['gbdt', 'dart','rf'],
        'num_leaves': np.arange(20,1000,10),
        'max_depth': np.arange(1,100,10),
        'n_estimators': np.arange(10,4000,100),
        'reg_alpha': [10.0**x for x in np.arange(-4,4)],
        'reg_lambda': [10.0**x for x in np.arange(-4,4)],
        'subsample': np.arange(0.1,1,0.1),
        'subsample_freq': np.arange(0,50,5),
        'min_split_gain': [10.0**x for x in np.arange(-4,0)]
    },
    n_iter=20,
    cv=10,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    scoring = 'neg_root_mean_squared_error',
    return_train_score = True
)


# In[337]:


_ = opt.fit(XSuicideCur_F,YSuicide)


# In[348]:


print('Best parameters (suicide):',opt.best_params_,',Test RMSE:',opt.best_score_)
print('Train RMSE:',min(opt.cv_results_['mean_train_score']))


# In[365]:


optcatLG = BayesSearchCV(
    CatBoostRegressor(random_state=42,verbose=1,thread_count=-1,bootstrap_type='Bayesian'), 
    {
        'colsample_bylevel': np.arange(0.1,1,0.1),
        'num_trees': np.arange(10,4000,100),
        'l2_leaf_reg': [10.0**x for x in np.arange(-4,4)],
        'num_leaves': np.arange(20,1000,10),
        'max_depth': np.arange(1,16,2),
        'bagging_temperature': np.arange(0.1,10,1),
        'grow_policy': ['Lossguide'],
        
    },
    n_iter=20,
    cv=10,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    scoring = 'neg_root_mean_squared_error',
    return_train_score = True
)


# In[366]:


_ = optcatLG.fit(XSuicideCur_F,YSuicide)


# In[367]:


print('Best parameters (suicide):',optcatLG.best_params_,',Test RMSE:',optcatLG.best_score_)
print('Train RMSE:',min(optcatLG.cv_results_['mean_train_score']))


# In[371]:


optcat = BayesSearchCV(
    CatBoostRegressor(random_state=42,verbose=1,thread_count=-1,bootstrap_type='Bayesian',used_ram_limit='5gb'), 
    {
        'colsample_bylevel': np.arange(0.1,1,0.1),
        'num_trees': np.arange(10,1000,100),
        'l2_leaf_reg': [10.0**x for x in np.arange(-4,4)],
        'max_depth': np.arange(1,16,2),
        'bagging_temperature': np.arange(0.1,10,1),
        'grow_policy': ['SymmetricTree','Depthwise'],
        'score_function': ['Cosine','L2']
        
    },
    n_iter=10,
    cv=10,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    scoring = 'neg_root_mean_squared_error',
    return_train_score = True
)


# In[372]:


_ = optcat.fit(XSuicideCur_F,YSuicide)


# In[373]:


print('Best parameters (suicide):',optcat.best_params_,',Test RMSE:',optcat.best_score_)
print('Train RMSE:',min(optcat.cv_results_['mean_train_score']))


# In[490]:


param_list = ['param_boosting_type','param_num_leaves','param_max_depth','param_n_estimators',
             'param_reg_alpha','param_reg_lambda','param_subsample','param_subsample_freq',
              'param_min_split_gain'] 
for param in param_list:
    param_set = sorted(list(set(opt.cv_results_[param])))
    param_trainscore = []
    param_testscore = []
    for item in param_set:
        param_trainscore.append(np.mean([opt.cv_results_['mean_train_score'][k] 
                                    for k in [i for i, x in enumerate(opt.cv_results_[param]) 
                                              if x == item]])) 
        param_testscore.append(np.mean([opt.cv_results_['mean_test_score'][k] 
                                    for k in [i for i, x in enumerate(opt.cv_results_[param]) 
                                              if x == item]])) 
    plt.plot(param_set,param_trainscore,label="Train",color='b')
    if(type(param_set[0]).__name__ != 'str'):
        plt.plot(param_set,np.poly1d(np.polyfit(param_set,param_trainscore,1))(param_set),'--',label="Train Trend",color='c')
    plt.plot(param_set,param_testscore,label="Test",color='r')
    if(type(param_set[0]).__name__ != 'str'):
        plt.plot(param_set,np.poly1d(np.polyfit(param_set,param_testscore,1))(param_set),'--',label="Test Trend",color='m')
    plt.legend()
    plt.grid(linestyle=':')
    plt.xlabel(param)
    plt.ylabel('Average RMSE (-ve) from 10 folds CV')
    plt.title("Effect of %s on suicide dataset (LightGBM)" % param)
    plt.savefig('Q26lightgbm'+param+'.png',dpi=300,bbox_inches='tight')
    plt.show()


# In[607]:


def join_list(param_set_1,param_set_2,param_score_1,param_score_2):
    names = param_set_1 + param_set_2
    results_values = param_score_1 + param_score_2
    averages = {}
    counts = {}
    for name, value in zip(names, results_values):
        if name in averages:
            averages[name] += value
            counts[name] += 1
        else:
            averages[name] = value
            counts[name] = 1
    for name in averages:
        averages[name] = averages[name]/float(counts[name])
    comb_param_set = list(averages.keys())
    comb_score = list(averages.values())
    return comb_param_set, comb_score


# In[609]:


param_list = ['param_colsample_bylevel','param_num_trees','param_max_depth','param_l2_leaf_reg',
              'param_num_leaves','param_bagging_temperature'] 
for param in param_list:
    param_set = sorted(list(set(optcatLG.cv_results_[param])))
    param_set_2 = sorted(list(set(optcat.cv_results_[param])))
    param_trainscore = []
    param_trainscore_2 = []
    param_testscore = []
    param_testscore_2 = []
    for item in param_set:
        param_trainscore.append(np.mean([optcatLG.cv_results_['mean_train_score'][k] 
                                    for k in [i for i, x in enumerate(optcatLG.cv_results_[param]) 
                                              if x == item]])) 
        param_testscore.append(np.mean([optcatLG.cv_results_['mean_test_score'][k] 
                                    for k in [i for i, x in enumerate(optcatLG.cv_results_[param]) 
                                              if x == item]])) 
    for it in param_set_2:
        param_trainscore_2.append(np.mean([optcat.cv_results_['mean_train_score'][k] 
                                    for k in [i for i, x in enumerate(optcat.cv_results_[param]) 
                                              if x == it]])) 
        param_testscore_2.append(np.mean([optcat.cv_results_['mean_test_score'][k] 
                                    for k in [i for i, x in enumerate(optcat.cv_results_[param]) 
                                              if x == it]])) 
        
    comb_param_set, comb_trainscore = join_list(param_set,param_set_2,param_trainscore,param_trainscore_2)
    comb_param_set, comb_testscore = join_list(param_set,param_set_2,param_testscore,param_testscore_2)
    
    plt.plot(comb_param_set,comb_trainscore,label="Train",color='b')
    if(type(comb_param_set[0]).__name__ != 'str'):
        plt.plot(comb_param_set,np.poly1d(np.polyfit(comb_param_set,comb_trainscore,1))(comb_param_set),'--',label="Train Trend",color='c')
    plt.plot(comb_param_set,comb_testscore,label="Test",color='r')
    if(type(comb_param_set[0]).__name__ != 'str'):
        plt.plot(comb_param_set,np.poly1d(np.polyfit(comb_param_set,comb_testscore,1))(comb_param_set),'--',label="Test Trend",color='m')
    plt.legend()
    plt.grid(linestyle=':')
    plt.xlabel(param)
    plt.ylabel('Average RMSE (-ve) from 10 folds CV')
    plt.title("Effect of %s on suicide dataset (CatBoost)" % param)
    plt.savefig('Q26catboost'+param+'.png',dpi=300,bbox_inches='tight')
    plt.show()


# In[611]:


param_list = ['param_score_function'] 
for param in param_list:
    param_set = sorted(list(set(optcat.cv_results_[param])))
    param_trainscore = []
    param_testscore = []
    for item in param_set:
        param_trainscore.append(np.mean([optcat.cv_results_['mean_train_score'][k] 
                                    for k in [i for i, x in enumerate(optcat.cv_results_[param]) 
                                              if x == item]])) 
        param_testscore.append(np.mean([optcat.cv_results_['mean_test_score'][k] 
                                    for k in [i for i, x in enumerate(optcat.cv_results_[param]) 
                                              if x == item]])) 
    plt.plot(param_set,param_trainscore,label="Train",color='b')
    if(type(param_set[0]).__name__ != 'str'):
        plt.plot(param_set,np.poly1d(np.polyfit(param_set,param_trainscore,1))(param_set),'--',label="Train Trend",color='c')
    plt.plot(param_set,param_testscore,label="Test",color='r')
    if(type(param_set[0]).__name__ != 'str'):
        plt.plot(param_set,np.poly1d(np.polyfit(param_set,param_testscore,1))(param_set),'--',label="Test Trend",color='m')
    plt.legend()
    plt.grid(linestyle=':')
    plt.xlabel(param)
    plt.ylabel('Average RMSE (-ve) from 10 folds CV')
    plt.title("Effect of %s on suicide dataset (CatBoost)" % param)
    plt.savefig('Q26catboost'+param+'.png',dpi=300,bbox_inches='tight')
    plt.show()


# In[619]:


param_list = ['param_grow_policy'] 
for param in param_list:
    param_set = sorted(list(set(optcat.cv_results_[param])))
    param_trainscore = []
    param_testscore = []
    for item in param_set:
        param_trainscore.append(np.mean([optcat.cv_results_['mean_train_score'][k] 
                                    for k in [i for i, x in enumerate(optcat.cv_results_[param]) 
                                              if x == item]])) 
        param_testscore.append(np.mean([optcat.cv_results_['mean_test_score'][k] 
                                    for k in [i for i, x in enumerate(optcat.cv_results_[param]) 
                                              if x == item]])) 
    param_trainscore.append(np.mean(optcatLG.cv_results_['mean_train_score']))
    param_testscore.append(np.mean(optcatLG.cv_results_['mean_test_score']))
    param_set.append('Lossguide')
    plt.plot(param_set,param_trainscore,label="Train",color='b')
    if(type(param_set[0]).__name__ != 'str'):
        plt.plot(param_set,np.poly1d(np.polyfit(param_set,param_trainscore,1))(param_set),'--',label="Train Trend",color='c')
    plt.plot(param_set,param_testscore,label="Test",color='r')
    if(type(param_set[0]).__name__ != 'str'):
        plt.plot(param_set,np.poly1d(np.polyfit(param_set,param_testscore,1))(param_set),'--',label="Test Trend",color='m')
    plt.legend()
    plt.grid(linestyle=':')
    plt.xlabel(param)
    plt.ylabel('Average RMSE (-ve) from 10 folds CV')
    plt.title("Effect of %s on suicide dataset (CatBoost)" % param)
    plt.savefig('Q26catboost'+param+'.png',dpi=300,bbox_inches='tight')
    plt.show()


# In[ ]:




