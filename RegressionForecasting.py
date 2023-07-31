#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:22:36 2023

@author: yunbai
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
import NCLensemble
import pmdarima as pm
from EnsemblePrescriptiveTree import *
import matplotlib.pyplot as plt
import os
import pickle

def MASEh(Insample, Actual, Predicted, m):
    """
    Insample should be in the form of pandas.dataframe, and is the training data
    m is the period parameter, e.g., for monthly data. m=12
    """
    country = Insample.columns[0]
    Insample['shift'] = Insample[country].shift(m)
    Insample = Insample.dropna()
    snaiveTerm = np.mean(np.abs(Insample[country] - Insample['shift']))
    mase = np.mean(np.abs(np.array(Actual) - np.array(Predicted)))/snaiveTerm
    
    return mase

#%%
def getTextFeat():
    textDf = pd.read_excel('./textFeats.xlsx')
    textDf['Date'] = pd.to_datetime(textDf['Date'])
    textDf['year'] = textDf['Date'].dt.year
    textDf['month'] = textDf['Date'].dt.month
    grouped_text = textDf.groupby(['year','month']).mean()
    idxList = []
    for i in range(len(grouped_text)):
        idxY = str(grouped_text.index[i][0])
        if grouped_text.index[i][1] < 10:
            idxM = '0'+str(grouped_text.index[i][1])
        else:
            idxM = str(grouped_text.index[i][1])
        idxList.append(idxY+'M'+idxM)
    grouped_text.index = idxList
    
    return grouped_text

def arima_exog(df,h,allIdx):  
    model = pm.auto_arima(df, seasonal=True, m=12)
    forecast, conf_int = model.predict(n_periods=h, return_conf_int=True)
    
    foreIdx = allIdx[allIdx > df.index[-1]]
    # forecast_df = pd.DataFrame({df.columns[0]: forecast, 'Lower CI': conf_int[:, 0], 'Upper CI': conf_int[:, 1]},index=foreIdx)
    forecast_df = pd.DataFrame({df.columns[0]: forecast},index=foreIdx)

    # # Plot the original time series and the forecasted values
    # plt.figure(figsize=(10, 6))
    # plt.plot(df, label='Original Data')
    # plt.plot(forecast_df['Forecast'], label='Forecast')
    # plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='gray', alpha=0.2, label='Confidence Interval')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.title('Auto ARIMA Forecast')
    # plt.legend()
    # plt.show()  
    return pd.concat([df,forecast_df])


# get economic data
#FRED是由美国联邦储备委员会（Federal Reserve）旗下的圣路易斯联邦储备银行（St. Louis Fed）提供的经济数据库
def get_all_files_in_folder(folder_path):
    all_files = []
    for root, directories, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

def getEconomicData(country):
    all_files = get_all_files_in_folder('./Countrydata/'+country+'/')
    
    dfList = []
    for k in range(len(all_files)):
        df = pd.read_csv(all_files[k])
        allIdx = pd.date_range(start=df['DATE'][0], end='2024-07-01', freq='MS')
        
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)
        
        # get the frequency of data, change into monthly if not
        freq = pd.infer_freq(df.index)
        if 'A' in freq or 'Q' in freq:
            df = df.resample('M').ffill()
        elif 'M' in freq:
            df = df
        else:
            df = df.resample('M').mean()
            
        # fill until 2024-07 with auto-arima
        h = len(allIdx) - len(df.index)
        print('Fitting auto_arima for feature:',k)
        df = arima_exog(df,h,allIdx)
        
        # change the index list
        idxList = []
        for i in range(len(df)):
            idxY = str(df.index[i].year)
            if df.index[i].month < 10:
                idxM = '0'+str(df.index[i].month)
            else:
                idxM = str(df.index[i].month)
            idxList.append(idxY+'M'+idxM)
        df.index = idxList
        
        # concat the data with original series
        # ts = pd.concat([ts,df],axis=1)
        # ts = ts.dropna()
        dfList.append(df)
    EcoChina = pd.concat(dfList,axis=1)
    
    return EcoChina,dfList
      
#%%
def dataPrepare(dataName,textFeat,EcoChina):
    # prepare the time series for every contry
    ts = dataset[[dataName]]
    # fill the possible null value
    ts[dataName] = ts[dataName].fillna(method='ffill', limit=12)
    ts['snaive'] = ts[dataName].shift(12)
    ts = ts.dropna()
    
    # add economic featrues from China
    ts = pd.concat([ts,EcoChina],axis=1)
    
    # add text features
    # ts = pd.concat([ts,textFeat],axis=1)
    # ts = ts.fillna(0)   
    # ts = pd.concat([ts,preDf],axis=0)
    
    # add year, month, quarter features
    ts['Year'] = ts.index.str[:4].astype(float)
    ts['Month'] = ts.index.str[5:].astype(float)
    ts['Quarter'] = ts['Month'].apply(lambda x: 1 if x <= 3 
                                      else (2 if x <= 6 and x>3
                                            else (3 if x <= 9 and x>7
                                                  else 4)))
    
    # add sin and cos waves for forecasting
    ts['monthofyear_sin'] = np.sin(2*np.pi*ts.Month/12)
    ts['monthofyear_cos'] = np.cos(2*np.pi*ts.Month/12)
    ts['quarterofyear_sin'] = np.sin(2*np.pi*ts.Quarter/4)
    ts['quarterofyear_cos'] = np.cos(2*np.pi*ts.Quarter/4)
    
    # one-hot encoded month indicator
    encoder = OneHotEncoder()
    month_indicator = pd.DataFrame(data = encoder.fit_transform(ts.Month.values.reshape(-1,1)).toarray(), columns = ['m_'+str(i) for i in range(12)])
    month_indicator.index = ts.index
    ts = pd.concat([ts,month_indicator],axis=1)   
    del month_indicator
    
    # Polynomial month and trend
    ts['trend'] = np.arange(len(ts))
    ts['Month_2'] = np.power(ts['Month'].values,2)
    ts['Month_3'] = np.power(ts['Month'].values,3)
    
    return ts

def create_vanilla_predictors(dataName,preDf):
    # prepare the time series for every contry
    ts = dataset[[dataName]]
    # fill the possible null value
    ts[dataName] = ts[dataName].fillna(method='ffill', limit=12)
    ts = pd.concat([ts,preDf])
    ts['snaive'] = ts[dataName].shift(12)   
    ts['fill_Snaive'] = ts[dataName].shift(24)
    ts['fill_Snaive'] = ts['fill_Snaive'].apply(lambda x: (x+ts[dataName].mean())/2)
    ts['snaive'] = ts['snaive'].fillna(ts['fill_Snaive'])
    ts = ts[[dataName,'snaive']]
    ts = ts.dropna(subset=['snaive'])
    
    # ts = pd.concat([ts,preDf])
    # ts = ts.merge(preDf,left_on='Date', right_on='Date', how='left')
    
    encoder = OneHotEncoder()
    target_pred = []
    fixed_pred = []
    
    # add economic featrues from China
    # ts = pd.concat([ts,EcoChina],axis=1)
    
    # add text features
    # ts = pd.concat([ts,textFeat],axis=1)
    # ts = ts.fillna(0)   
    # ts = pd.concat([ts,preDf],axis=0)
    
    # add year, month, quarter features
    ts['Year'] = ts.index.str[:4].astype(float)
    ts['Month'] = ts.index.str[5:].astype(float)
    ts['Quarter'] = ts['Month'].apply(lambda x: 1 if x <= 3 
                                      else (2 if x <= 6 and x>3
                                            else (3 if x <= 9 and x>7
                                                  else 4)))
    
    # add holiday
    # ts['May'] = ts['Month'].apply(lambda x: 1 if x == 5 else (0))
    # ts['October'] = ts['Month'].apply(lambda x: 1 if x == 10 else (0))
    
    # Fourier terms: sin/cos waves for monthly and quarterly
    # ts['month_sin'] = np.sin(2*np.pi*(ts['Month'])/12)
    # ts['month_cos'] = np.cos(2*np.pi*(ts['Month'])/12)
    
    # ts['quarter_sin'] = np.sin(2*np.pi*(ts['Quarter'])/12)
    # ts['quarter_cos'] = np.cos(2*np.pi*(ts['Quarter'])/12)
    
    # # smooth_factor for snaive
    # smooth_factors = np.arange(.1, 1, .1)
    # for a in smooth_factors:    
    #     ts['s_ewm_'+str(a.round(1))] = ts['snaive'].ewm(alpha = a).mean()
    #     target_pred.append('s_ewm_'+str(a.round(1)))
    
    # One-hot encoded calendar variables
    startY,endY = int(ts.index[0][:4]),int(ts.index[-1][:4])
    month_indicator = pd.DataFrame(data = encoder.fit_transform(ts['Month'].values.astype(str).reshape(-1,1)).toarray(), columns = ['m_'+str(i) for i in range(12)])
    year_indicator = pd.DataFrame(data = encoder.fit_transform(ts['Year'].values.astype(str).reshape(-1,1)).toarray(), columns = ['y_'+str(i) for i in range(startY,endY+1)])
    quarter_indicator = pd.DataFrame(data = encoder.fit_transform(ts['Quarter'].values.astype(str).reshape(-1,1)).toarray(), columns = ['q_'+str(i) for i in range(4)])
    
    # Polynomial snaive and trend
    ts['trend'] = np.arange(len(ts))
    ts['s_2'] = np.power(ts['snaive'].values,2)
    ts['s_3'] = np.power(ts['snaive'].values,3)
    target_pred = target_pred + ['snaive','trend','s_2','s_3']
    
    # snaive-year interaction
    for c in ['y_'+str(i) for i in range(startY,endY+1)]:
        ts['s_'+c] = ts['snaive'].values*year_indicator[c].values
        ts['s_2_'+c] = np.power(ts['snaive'].values,2)*year_indicator[c].values
        ts['s_3_'+c] = np.power(ts['snaive'].values,3)*year_indicator[c].values
        target_pred = target_pred + ['s_'+c, 's_2_'+c, 's_3_'+c]
    # snaive-quarter interaction
    for c in ['q_'+str(i) for i in range(4)]:
        ts['s_'+c] = ts['snaive'].values*quarter_indicator[c].values
        ts['s_2_'+c] = np.power(ts['snaive'].values,2)*quarter_indicator[c].values
        ts['s_3_'+c] = np.power(ts['snaive'].values,3)*quarter_indicator[c].values
        target_pred = target_pred + ['s_'+c, 's_2_'+c, 's_3_'+c]
    # snaive-month interaction
    for c in ['m_'+str(i) for i in range(12)]:
        ts['s_'+c] = ts['snaive'].values*month_indicator[c].values
        ts['s_2_'+c] = np.power(ts['snaive'].values,2)*month_indicator[c].values
        ts['s_3_'+c] = np.power(ts['snaive'].values,3)*month_indicator[c].values
        target_pred = target_pred + ['s_'+c, 's_2_'+c, 's_3_'+c]
    
    # Indicator for month and hour*weekday
    ts[['m_'+str(i) for i in range(12)]] = encoder.fit_transform(ts['Month'].values.astype(str).reshape(-1,1)).toarray()
    # month_quarter_inter = ts['Month'].values.astype(int).astype(str)+ts['Quarter'].values.astype(int).astype(str)
    # ts[['m_q_int'+str(i) for i in range(12*4)]] = encoder.fit_transform(month_quarter_inter.values.reshape(-1,1)).toarray()
    
    fixed_pred = fixed_pred + ['m_'+str(i) for i in range(12)]
    
    return ts

def custom_objective_xgb(y_true, y_pred):
    grad = (y_pred - y_true) / (y_pred + 1)  # 梯度
    hess = 2 * (y_pred + 1)  # 二阶导数
    return grad, hess


#%%
def forecastModel(trainX,trainY,testX):
    #ExtraTrees
    Et_params = {'n_estimators':[10,50,100]}
    Et_model = GridSearchCV(ExtraTreesRegressor(n_jobs=-2), Et_params,cv=KFold(n_splits=5)).fit(trainX, trainY)
    Et_pred = Et_model.predict(testX)
    
    #Randomforest
    Rf_params = {'n_estimators':[10,50,100]}
    Rf_model = GridSearchCV(RandomForestRegressor(n_jobs=-2), Rf_params,cv=KFold(n_splits=5)).fit(trainX, trainY)
    Rf_pred = Rf_model.predict(testX)
    
    # lg_params = {'n_estimators':[10,50,100],'learning_rate':[0.001,0.01,0.1],
    #              'max_depth': [3, 5, 7]}
    # lg_model = GridSearchCV(LGBMRegressor(objective='poisson'), lg_params,cv=KFold(n_splits=5)).fit(trainX, trainY)
    # lg_pred = lg_model.predict(testX)
    return Et_pred,Rf_pred
 
def trainModel(ts,dataName):
    # set training and test data
    # select 2018.08-2019,07, and 2021.08-2022.07 two years as test set (pre and post-covid)
    test_18 = ts[(ts.Year==2018)&(ts.Month>=8)]
    test_19 = ts[(ts.Year==2019)&(ts.Month<=7)]
    test_21 = ts[(ts.Year==2021)&(ts.Month>=8)]
    test_22 = ts[(ts.Year==2022)&(ts.Month<=7)]
    test_pre_covid = pd.concat([test_18,test_19])
    test_post_covid = pd.concat([test_21,test_22])
    del test_18,test_19,test_21,test_22
    
    # get the test set for the competition
    test_23 = ts[(ts.Year==2023)&(ts.Month>=8)]
    test_24 = ts[(ts.Year==2024)&(ts.Month<=7)]
    test_comptition = pd.concat([test_23,test_24])
    del test_23,test_24
    
    # get the training set
    train_1 = ts[(ts.Year<2018)]
    train_2 = ts[(ts.Year==2018)&(ts.Month<8)]
    train_3 = ts[(ts.Year==2019)&(ts.Month>7)]
    train_4 = ts[(ts.Year>2019)&(ts.Year<2021)]
    train_5 = ts[(ts.Year==2021)&(ts.Month<8)]
    train_6 = ts[(ts.Year==2022)&(ts.Month>7)]
    train_7 = ts[(ts.Year==2023)&(ts.Month<8)]
    train = pd.concat([train_1,train_2,train_3,train_4,
                       train_5,train_6,train_7])
    train = train.dropna()
    train_2021 = train[(train.Year<2021)]
    
    # PreTrain and PostTrain are for calculating MASE
    PreTrain = pd.concat([train_1,train_2])
    PreTrain = PreTrain[[dataName]]
    PostTrain = pd.concat([train_2021,train_5])
    PostTrain = PostTrain[[dataName]]
    del train_1,train_2,train_3,train_4,train_5,train_6,train_7,train_2021,ts
    
    #%% 
    # get scaled data
    X_scaler,Y_scaler = MinMaxScaler(),MinMaxScaler()
    cols = list(train.columns)
    cols.remove(dataName)
    trainX,trainY = train[cols],train[[dataName]]
    
    Yidx = trainY.index
    testX_pre_covid,testY_pre_covid = test_pre_covid[cols],test_pre_covid[[dataName]]
    testX_post_covid,testY_post_covid = test_post_covid[cols],test_post_covid[[dataName]]
    
    X_scaler.fit_transform(trainX)
    Y_scaler.fit_transform(trainY)
    
    trainX,trainY = X_scaler.transform(trainX),Y_scaler.transform(trainY)
    testX_pre_covid = X_scaler.transform(testX_pre_covid)
    testX_post_covid = X_scaler.transform(testX_post_covid) 

    ## train linear models
    # linear regression
    # lr = LinearRegression(fit_intercept=True)
    # lr.fit(trainX, trainY)
    # lr_pred_pre_covid = Y_scaler.inverse_transform(lr.predict(testX_pre_covid))
    # lr_pred_post_covid = Y_scaler.inverse_transform(lr.predict(testX_post_covid))
        
    # ridge
    param_grid = {"alpha": [10**pow for pow in range(-5,4)]}
    ridge = GridSearchCV(Ridge(fit_intercept = True), param_grid,cv=KFold(n_splits=5))
    ridge.fit(trainX, trainY)
    ridge_pred_pre_covid = Y_scaler.inverse_transform(ridge.predict(testX_pre_covid))
    ridge_pred_post_covid = Y_scaler.inverse_transform(ridge.predict(testX_post_covid))
    
    # lasso
    lasso = GridSearchCV(Lasso(),param_grid,cv=KFold(n_splits=5)).fit(trainX, trainY)
    lasso_pred_pre_covid = Y_scaler.inverse_transform(lasso.predict(testX_pre_covid).reshape(-1,1))
    lasso_pred_post_covid = Y_scaler.inverse_transform(lasso.predict(testX_post_covid).reshape(-1,1))
    
    # XGBoost
    xgb_params = {'max_depth':[3],'alpha':[0.1],'eta':[0.01,0.05,0.1],
                  'n_estimators':[10,50,100],'subsample':[0.8],
                  'colsample_bytree':[0.8],'reg_lambda':[1]}
    xgb_model = GridSearchCV(XGBRegressor(n_jobs=-2), xgb_params,cv=KFold(n_splits=5)).fit(trainX, trainY)
    xgb_pred_pre_covid = Y_scaler.inverse_transform(xgb_model.predict(testX_pre_covid).reshape(-1,1))
    xgb_pred_post_covid = Y_scaler.inverse_transform(xgb_model.predict(testX_post_covid).reshape(-1,1))
    
    #ExtraTrees
    Et_params = {'n_estimators':[10,50,100]}
    Et_model = GridSearchCV(ExtraTreesRegressor(n_jobs=-2), Et_params,cv=KFold(n_splits=5)).fit(trainX, trainY)
    Et_pred_pre_covid = Y_scaler.inverse_transform(Et_model.predict(testX_pre_covid).reshape(-1,1))
    Et_pred_post_covid = Y_scaler.inverse_transform(Et_model.predict(testX_post_covid).reshape(-1,1))
    
    #Randomforest
    Rf_params = {'n_estimators':[10,50,100]}
    Rf_model = GridSearchCV(RandomForestRegressor(n_jobs=-2), Rf_params,cv=KFold(n_splits=5)).fit(trainX, trainY)
    Rf_pred_pre_covid = Y_scaler.inverse_transform(Rf_model.predict(testX_pre_covid).reshape(-1,1))
    Rf_pred_post_covid = Y_scaler.inverse_transform(Rf_model.predict(testX_post_covid).reshape(-1,1))

    
    # evaluate
    errorDict = dict()
    
    # errorDict['linear-MASE-pre-covid'] = MASEh(PreTrain,testY_pre_covid,lr_pred_pre_covid,m=12)
    # errorDict['linear-MASE-post-covid'] = MASEh(PostTrain,testY_post_covid,lr_pred_post_covid,m=12)
    # errorDict['ridge-MASE-pre-covid'] = MASEh(PreTrain,testY_pre_covid,ridge_pred_pre_covid,m=12)
    errorDict['ridge-MASE-post-covid'] = MASEh(PostTrain,testY_post_covid,ridge_pred_post_covid,m=12)
    # errorDict['lasso-MASE-pre-covid'] = MASEh(PreTrain,testY_pre_covid,lasso_pred_pre_covid,m=12)
    errorDict['lasso-MASE-post-covid'] = MASEh(PostTrain,testY_post_covid,lasso_pred_post_covid,m=12)
    # errorDict['xgb-MASE-pre-covid'] = MASEh(PreTrain,testY_pre_covid,xgb_pred_pre_covid,m=12)
    errorDict['xgb-MASE-post-covid'] = MASEh(PostTrain,testY_post_covid,xgb_pred_post_covid,m=12)
    # errorDict['Et-MASE-pre-covid'] = MASEh(PreTrain,testY_pre_covid,Et_pred_pre_covid,m=12)
    errorDict['Et-MASE-post-covid'] = MASEh(PostTrain,testY_post_covid,Et_pred_post_covid,m=12)
    # errorDict['Rf-MASE-pre-covid'] = MASEh(PreTrain,testY_pre_covid,Rf_pred_pre_covid,m=12)
    errorDict['Rf-MASE-post-covid'] = MASEh(PostTrain,testY_post_covid,Rf_pred_post_covid,m=12)
    
    return errorDict

def trainModel_postCovid(ts,dataName):
    # set training and test data
    # select 2021.08-2022.07 two years as test set (pre and post-covid)
    test_21 = ts[(ts.Year==2021)&(ts.Month>=8)]
    test_22 = ts[(ts.Year==2022)&(ts.Month<=7)]
    test_post_covid = pd.concat([test_21,test_22])
    del test_21,test_22
    
    # get the test set for the competition
    test_23 = ts[(ts.Year==2023)&(ts.Month>=8)]
    test_24 = ts[(ts.Year==2024)&(ts.Month<=7)]
    test_comptition = pd.concat([test_23,test_24])
    del test_23,test_24
    
    # get the training set
    train_1 = ts[(ts.Year<2021)]
    train_5 = ts[(ts.Year==2021)&(ts.Month<8)]
    # train_6 = ts[(ts.Year==2022)&(ts.Month>7)]
    # train_7 = ts[(ts.Year==2023)&(ts.Month<8)]
    # train = pd.concat([train_1,train_5,train_6,train_7])
    train = pd.concat([train_1,train_5])
    train = train.dropna()
    
    # PreTrain and PostTrain are for calculating MASE
    PostTrain = pd.concat([train_1,train_5])
    PostTrain = PostTrain[[dataName]]
    del train_1,train_5,ts
    
    #%% 
    # get scaled data
    X_scaler,Y_scaler = MinMaxScaler(),MinMaxScaler()
    cols = list(train.columns)
    cols.remove(dataName)
    trainX,trainY = train[cols],train[[dataName]]
  
    testX_post_covid,testY_post_covid = test_post_covid[cols],test_post_covid[[dataName]]
    
    X_scaler.fit_transform(trainX)
    Y_scaler.fit_transform(trainY)
    
    trainX,trainY = X_scaler.transform(trainX),Y_scaler.transform(trainY)
    testX_post_covid = X_scaler.transform(testX_post_covid) 
    
    test_comptition_X = test_comptition[cols]
    test_comptition_X = X_scaler.transform(test_comptition_X)
    

    # # lasso
    # param_grid = {"alpha": [10**pow for pow in range(-5,4)]}
    # lasso = GridSearchCV(Lasso(),param_grid,cv=KFold(n_splits=5)).fit(trainX, trainY)
    # lasso_pred_post_covid = Y_scaler.inverse_transform(lasso.predict(testX_post_covid).reshape(-1,1))
    
    # # XGBoost
    # xgb_params = {'max_depth':[3],'alpha':[0.1],'eta':[0.01,0.05,0.1],
    #               'n_estimators':[10,50,100],'subsample':[0.8],
    #               'colsample_bytree':[0.8],'reg_lambda':[1]}
    # xgb_model = GridSearchCV(XGBRegressor(objective=custom_objective_xgb,n_jobs=-2), xgb_params,cv=KFold(n_splits=5)).fit(trainX, trainY)
    # xgb_pred_post_covid = Y_scaler.inverse_transform(xgb_model.predict(testX_post_covid).reshape(-1,1))
    
    #ExtraTrees
    Et_params = {'n_estimators':[10,50,100]}
    Et_model = GridSearchCV(ExtraTreesRegressor(n_jobs=-2), Et_params,cv=KFold(n_splits=5)).fit(trainX, trainY)
    Et_pred_post_covid = Y_scaler.inverse_transform(Et_model.predict(testX_post_covid).reshape(-1,1))
    Et_comp = Y_scaler.inverse_transform(Et_model.predict(test_comptition_X).reshape(-1,1))
    
    # set confidence interval
    quantiles = [0.1,0.9]
    predictions_ET = np.array([tree.predict(test_comptition_X) for tree in Et_model.best_estimator_.estimators_])
    lower_bound_ET = np.percentile(predictions_ET, quantiles[0] * 100, axis=0)
    lower_bound_ET = Y_scaler.inverse_transform(lower_bound_ET.reshape(-1,1))
    upper_bound_ET = np.percentile(predictions_ET, quantiles[1] * 100, axis=0)
    upper_bound_ET = Y_scaler.inverse_transform(upper_bound_ET.reshape(-1,1))
    quantile_ETpredictions = {quantiles[0]:lower_bound_ET,quantiles[1]:upper_bound_ET}
    
    #Randomforest
    Rf_params = {'n_estimators':[10,50,100]}
    Rf_model = GridSearchCV(RandomForestRegressor(n_jobs=-2), Rf_params,cv=KFold(n_splits=5)).fit(trainX, trainY)
    Rf_pred_post_covid = Y_scaler.inverse_transform(Rf_model.predict(testX_post_covid).reshape(-1,1))
    Rf_comp = Y_scaler.inverse_transform(Rf_model.predict(test_comptition_X).reshape(-1,1))
    
    
    # lg_params = {'n_estimators':[10,50,100],'learning_rate':[0.001,0.01,0.1],
    #              'max_depth': [3, 5, 7]}
    # lg_model = GridSearchCV(LGBMRegressor(objective='poisson'), lg_params,cv=KFold(n_splits=5)).fit(trainX, trainY)
    # lg_pred_post_covid = Y_scaler.inverse_transform(lg_model.predict(testX_post_covid).reshape(-1,1))
    
    
    #Combination
    comb_pred_post_covid = (Et_pred_post_covid+Rf_pred_post_covid)/2
    comb_comp = (Et_comp+Rf_comp)/2
    
    # evaluate
    errorDict = dict()   
    # errorDict['lasso-MASE-post-covid'] = MASEh(PostTrain,testY_post_covid,lasso_pred_post_covid,m=12)
    # errorDict['xgb-MASE-post-covid'] = MASEh(PostTrain,testY_post_covid,xgb_pred_post_covid,m=12)
    errorDict['Et-MASE-post-covid'] = MASEh(PostTrain,testY_post_covid,Et_pred_post_covid,m=12)
    errorDict['Rf-MASE-post-covid'] = MASEh(PostTrain,testY_post_covid,Rf_pred_post_covid,m=12)
    # errorDict['lg-MASE-post-covid'] = MASEh(PostTrain,testY_post_covid,lg_pred_post_covid,m=12)
    
    errorDict['Comb-MASE-post-covid'] = MASEh(PostTrain,testY_post_covid,comb_pred_post_covid,m=12)
    
    return errorDict,comb_comp,quantile_ETpredictions

def trainModel_postCovid_NCL(ts,dataName):
    # set training and test data
    # select 2021.08-2022.07 two years as test set (pre and post-covid)
    test_21 = ts[ts.Year==2021]
    test_22 = ts[ts.Year==2022]
    
   
    # get the test set for the competition
    test_23 = ts[(ts.Year==2023)&(ts.Month>=8)]
    test_24 = ts[(ts.Year==2024)&(ts.Month<=7)]
    test_comptition = pd.concat([test_23,test_24])
    del test_23,test_24
    
    # get the training set
    train_21 = ts[ts.Year<2021]
    train_22 = ts[ts.Year<2022]
    # train = train.dropna()
    
    # PreTrain and PostTrain are for calculating MASE
    PostTrain = train_22[[dataName]]
    
    #%% 
    # get scaled data
    X_scaler,Y_scaler = MinMaxScaler(),MinMaxScaler()
    cols = list(train_21.columns)
    cols.remove(dataName)
    trainX_21,trainY_21 = train_21[cols],train_21[[dataName]]
    trainX_22,trainY_22 = train_22[cols],train_22[[dataName]]
  
  
    testX_21,testY_21 = test_21[cols],test_21[[dataName]]
    testX_22,testY_22 = test_22[cols],test_22[[dataName]]
    
    X_scaler.fit_transform(trainX_21)
    Y_scaler.fit_transform(trainY_21)
    
    trainX_21,trainY_21 = X_scaler.transform(trainX_21),Y_scaler.transform(trainY_21)
    trainX_22,trainY_22 = X_scaler.transform(trainX_22),Y_scaler.transform(trainY_22)
    # testY_21 = Y_scaler.transform(testY_21)
    
    testX_21 = X_scaler.transform(testX_21) 
    testX_22 = X_scaler.transform(testX_22) 
    
    
    # Forecasting
    num_of_model = 2
    Et_pred_21,Rf_pred_21 = forecastModel(trainX_21,trainY_21,testX_21)
    Et_pred_21 = Y_scaler.inverse_transform(Et_pred_21.reshape(-1,1))
    Rf_pred_21 = Y_scaler.inverse_transform(Rf_pred_21.reshape(-1,1))
    
    Y_pres_weight = np.zeros(shape=(testY_21.shape[0],num_of_model))
    Y_pres_weight[:,0],Y_pres_weight[:,1] = Et_pred_21.reshape(-1,),Rf_pred_21.reshape(-1,)
    
    param_grid = {"alpha": [10**pow for pow in range(-5,4)]}
    lasso = GridSearchCV(Lasso(),param_grid).fit(Y_pres_weight, testY_21)
    
    #combination
    Et_pred_22,Rf_pred_22 = forecastModel(trainX_22,trainY_22,testX_22)
    Et_pred_22 = Y_scaler.inverse_transform(Et_pred_22.reshape(-1,1))
    Rf_pred_22 = Y_scaler.inverse_transform(Rf_pred_22.reshape(-1,1))
    Y_pres_comb = np.zeros(shape=(testY_22.shape[0],num_of_model))
    Y_pres_comb[:,0],Y_pres_comb[:,1] = Et_pred_22.reshape(-1,),Rf_pred_22.reshape(-1,)
    
    
    test_ncl = lasso.predict(Y_pres_comb)
    # test_ncl = Y_scaler.inverse_transform(test_ncl.reshape(-1,1))
     
    #Combination
    test_avg = (Et_pred_22+Rf_pred_22)/2
    # test_avg = Y_scaler.inverse_transform(test_avg.reshape(-1,1))
    
    # evaluate
    
    errorDict = dict()   
    errorDict['ExtraTrees-2022'] = MASEh(PostTrain,testY_22,Et_pred_22,m=12)
    errorDict['RandomForest-2022'] = MASEh(PostTrain,testY_22,Rf_pred_22,m=12)
    errorDict['SimpleAverage'] = MASEh(PostTrain,testY_22,test_avg,m=12)
    errorDict['NCLcombination'] = MASEh(PostTrain,testY_22,test_ncl,m=12)
    
    return errorDict



def arima_func(dataset,country):
    ts = dataset[['Mexico']]
    ts = ts.dropna()
    ts_preCovid_train,ts_preCovid_test = ts[ts.index<'2018M08'],ts[(ts.index>='2018M08')&(ts.index<='2019M07')]
    ts_postCovid_train,ts_postCovid_test = ts[ts.index<'2021M08'],ts[(ts.index>='2021M08')&(ts.index<='2022M07')]
    
    model_preCovid = pm.auto_arima(ts_preCovid_train, seasonal=True, m=12)
    fore_preCovid, _ = model_preCovid.predict(n_periods=12, return_conf_int=True)
    MASEh(ts_preCovid_train,ts_preCovid_test,fore_preCovid,m=12)
    
    model_postCovid = pm.auto_arima(ts_postCovid_train, seasonal=True, m=12)
    fore_postCovid, _ = model_postCovid.predict(n_periods=12, return_conf_int=True)
    MASEh(ts_postCovid_train,ts_postCovid_test,fore_postCovid,m=12)
    
    model = pm.auto_arima(ts, seasonal=True, m=12)
    print(model.summary())
    forecast, conf_int = model.predict(n_periods=19, return_conf_int=True)
    
    Idx2023 = []
    for i in range(1,13):
        if i < 10:
            Idx2023.append('2023M0'+str(i))
        else:
            Idx2023.append('2023M'+str(i))
    Idx2024 = ['2024M'+str(i) for i in range(1,8)]
    
    
    forecast_df = pd.DataFrame({'Forecast': forecast, 'Lower CI': conf_int[:, 0], 'Upper CI': conf_int[:, 1]},index=Idx2023+Idx2024)
    
    # Plot the original time series and the forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(ts, label='Original Data')
    plt.plot(forecast_df['Forecast'], label='Forecast')
    plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='gray', alpha=0.2, label='Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Auto ARIMA Forecast')
    plt.legend()
    plt.show()  

#%%
if __name__ == '__main__':
    # data preparation
    # loading the original data
    dataset = pd.read_excel('Tourism forecasting competition II dataset.xlsx',sheet_name='data for forecasting')
    dataset = dataset.set_index(dataset.iloc[:,0])
    dataset.index.names = ['Date']
    dataset = dataset.drop([dataset.columns[0]],axis=1)
    
    # creat a null df for filling the predictions
    preDf = pd.DataFrame()
    preDf['Date'] = ['2023M03','2023M04','2023M05','2023M06','2023M07',
                     '2023M08','2023M09','2023M10','2023M11','2023M12','2024M01',
                     '2024M02','2024M03','2024M04','2024M05','2024M06','2024M07']
    preDf = preDf.set_index('Date')
    
    # load the filled data by bagging in R 
    
    
    # add textual features
    # textFeat = getTextFeat()
    
    # add Chinese economic features
    # EcoChina,dfList = getEconomicData('China')
    # with open('./EcoChina.pkl','rb') as f:
    #     EcoChina = pickle.load(f)
    # EcoChina = EcoChina.dropna()
    # EcoChina.index.name = 'Date'
    
    countryList = list(dataset.columns)
    
    resultsDf = pd.DataFrame()
    foreDf = pd.DataFrame()
    quantileDf = pd.DataFrame()
    
    # for each country, train linear models and save the results for pre- and post-covid
    for country in countryList:
        ts = create_vanilla_predictors(country,preDf)
        # ts = pd.concat([ts,EcoChina,textFeat],axis=1)
        # ts = ts.dropna()
        # ts = ts.fillna(0)  
        # ts = dataPrepare(country,textFeat,EcoChina)
        print('#'*50)
        print(country)
        print('#'*50)
        errorDict,fore_comp,quantile_ETpredictions = trainModel_postCovid(ts,country)
        print(errorDict)
        resultsDf = resultsDf.append(errorDict,ignore_index=True)
        foreDf[country] = list(fore_comp.reshape(-1,))
        quantileDf[country+'_0.1'] = list(quantile_ETpredictions[0.1].reshape(-1,))
        quantileDf[country+'_0.9'] = list(quantile_ETpredictions[0.9].reshape(-1,))
        
    resultsDf['Country'] = countryList
    # resultsDf.to_csv('LinearResults-comb-ET-RF.csv',index=False)
    foreDf.to_csv('Results_comp_0729.csv',index=False)
    quantileDf.to_csv('QuantileResults.csv',index=False)
    
    xixiPre = pd.read_csv('bagging_results.csv')
    xixiPre.columns = foreDf.columns
    results_avg = (foreDf + xixiPre) / 2
    results_avg.to_csv('Results_avg_with_bagging.csv',index=False)
    
    xixiLower = pd.read_csv('lower_bound_bagging.csv')
    xixiLower.columns = [country+'_0.1' for country in countryList]
    xixiLower = xixiLower.applymap(lambda x: 0 if x < 0 else x)
    
    xixiUpper = pd.read_csv('upper_bound_bagging.csv')
    xixiUpper.columns = [country+'_0.9' for country in countryList]
    quantileDfxixi = pd.DataFrame()
    for country in countryList:
        quantileDfxixi[country+'_0.1'] = xixiLower[country+'_0.1']
        quantileDfxixi[country+'_0.9'] = xixiUpper[country+'_0.9']
    quantile_avg = (quantileDf+quantileDfxixi)/2
    quantile_avg.to_csv('Quantile_avg_with_bagging.csv',index=False)
    
    
    









