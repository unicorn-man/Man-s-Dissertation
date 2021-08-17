# Import packages
import os
import urllib
import requests
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc  
from sklearn.model_selection import train_test_split
from statsmodels.stats import diagnostic
from scipy.stats import boxcox
import statsmodels.api as sm


data_camels = pd.read_csv("C://Users//Administrator//Desktop//results//datatest.csv")



# data description
dt = data_camels.iloc[:,1:]
print(dt.describe())




# multiple regression to see which variables are significant
lm = smf.ols('discharge ~ AREA + ALTBAR + ASPBAR + BFIHOST + DPLBAR + FPDBAR + LDP + PROPWET + RAINFALL + SPRHOST', data=data_camels).fit()
print(lm.summary())



# Seaborn can add a best-fitting line and a 95% confidence band
sns.pairplot(dt, x_vars=['AREA','ALTBAR','ASPBAR','BFIHOST','DPLBAR','FPDBAR','LDP','PROPWET','RAINFALL','SPRHOST'], y_vars='discharge', size=7, aspect=0.8,kind = 'reg')
plt.show()


# This is regression model with one variable

data_y = data_camels['discharge']
data_x = data_camels['LDP']



model1 = smf.ols('discharge ~ LDP', data=data_camels).fit() 
print(model1.summary())



# To predict discharge based on the regression model
predict_y = model1.predict(data_x)


fig, ax = plt.subplots(figsize=(8, 5))
plt.scatter(data_x,data_y,color='blue')
plt.plot(data_x,predict_y,color='red',linewidth=2)
ax.set_xlabel("LDP km")
ax.set_ylabel("Discharge: 15 year return Aperiod m³/s")
plt.title("41 catchments")
plt.show()



data_camels['resid'] = model1.resid_pearson
fig1 = plt.figure(figsize=(8, 5), dpi=100)
ax = fig1.add_subplot(111)
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
ax.scatter(predict_y, data_camels['resid'] )
plt.axhline(y = 0, color = 'r', linewidth = 2)
plt.title("Residual Distribution of LDP ")
plt.show()



# model evaluation
rmse = (np.sqrt(mean_squared_error(data_y, predict_y)))
r2 = r2_score(data_y, predict_y)
print(rmse)
print(r2)


cc_breus = diagnostic.het_breuschpagan(model1.resid, model1.model.exog)
print(cc_breus)




# Logarithmic Transformation

data_camels['LDP_ln'] = np.log(data_camels['LDP'])
data_camels['discharge_ln'] = np.log(data_camels['discharge'])


quadmodel = smf.ols('discharge_ln ~ LDP_ln', data=data_camels).fit() 
print(quadmodel.summary())


predict_y_new = quadmodel.predict(data_camels['LDP_ln'])


fig, ax = plt.subplots(figsize=(8, 5))
plt.scatter(data_camels['LDP_ln'],data_camels['discharge_ln'],color='blue')
plt.plot(data_camels['LDP_ln'],predict_y_new,color='red',linewidth=2)
ax.set_xlabel("ln(LDP)")
ax.set_ylabel("ln(Discharge)")
plt.title("41 catchments")
plt.show()

# model evaluation for training set
rmse_new = (np.sqrt(mean_squared_error(data_camels['discharge_ln'], predict_y_new)))
r2_new = r2_score(data_camels['discharge_ln'], predict_y_new)
print(rmse_new)
print(r2_new)


data_camels['resid_ln'] = quadmodel.resid_pearson
fig1 = plt.figure(figsize=(8, 5), dpi=100)
ax = fig1.add_subplot(111)
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
ax.scatter(predict_y_new, data_camels['resid_ln'] )
plt.axhline(y = 0, color = 'r', linewidth = 2)
plt.title("Residual Distribution of ln(LDP) ")
plt.show()



cc_breus1 = diagnostic.het_breuschpagan(quadmodel.resid, quadmodel.model.exog)
print(cc_breus1)








# Multiple Regression (OLS)

data_y = data_camels['discharge']
data_multi = data_camels.loc[:,['AREA','DPLBAR','LDP']]



# multiple regression based on area,dplbar and ldp
multi_model = smf.ols('discharge ~ AREA + DPLBAR + LDP', data=data_camels).fit() 
print(multi_model.summary())


# To predict discharge based on the regression model
predict_multi_y = multi_model.predict(data_multi)



fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(range(len(predict_multi_y)),predict_multi_y,'b',label="predict")
plt.plot(range(len(predict_multi_y)),data_y,'r',label="data")
plt.legend(loc="upper right") 
ax.set_xlabel("the number of catchments")
ax.set_ylabel("Discharge: 15 year return Aperiod m³/s")
plt.title("The prediction of multiple regression(OLS)")
plt.show()
plt.show()


# model evaluation
rmse_multi = (np.sqrt(mean_squared_error(data_y, predict_multi_y)))
r2_multi = r2_score(data_y, predict_multi_y)
print(rmse_multi)
print(r2_multi)

data_camels['resid'] = multi_model.resid_pearson
fig1 = plt.figure(figsize=(8, 5), dpi=100)
ax = fig1.add_subplot(111)
ax.scatter(predict_multi_y, data_camels['resid'] )
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
plt.axhline(y = 0, color = 'r', linewidth = 2)
plt.title("Residual Distribution of multiple regression(OLS)")
plt.show()



cc_breus = diagnostic.het_breuschpagan(multi_model.resid, multi_model.model.exog)
print(cc_breus)


fig, ax = plt.subplots(figsize=(8, 5))
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--',c='k', label="1:1 line")
plt.scatter(data_y,predict_multi_y,color='blue')
ax.set_xlabel("Calculated discharge")
ax.set_ylabel("Prediction discharge")
plt.show()


# To test on validation data set
x_test = pd.read_csv("C://Users//Administrator//Desktop//results//test.csv")
print(multi_model.predict(x_test)) 






## This is linear regression model with Weighted Least Squares


w1 = 1/(1+np.abs(multi_model.resid))



wls_fit1 = smf.wls('discharge ~ AREA + DPLBAR + LDP', data=data_camels, weights = w1).fit()
print(wls_fit1.summary())

# To predict discharge based on the regression model
predict_wls1_y = wls_fit1.predict(data_multi)



fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(range(len(predict_wls1_y)),predict_wls1_y,'b',label="predict")
plt.plot(range(len(predict_wls1_y)),data_y,'r',label="data")
plt.legend(loc="upper right") 
ax.set_xlabel("the number of catchments")
ax.set_ylabel("Discharge: 15 year return Aperiod m³/s")
plt.title("The prediction of multiple regression(WLS)")
plt.show()
plt.show()


# model evaluation
rmse_wls1 = (np.sqrt(mean_squared_error(data_y, predict_wls1_y)))
r2_wls1 = r2_score(data_y, predict_wls1_y)
print(rmse_wls1)
print(r2_wls1)




data_camels['resid1'] = wls_fit1.resid_pearson
fig1 = plt.figure(figsize=(8, 5), dpi=100)
ax = fig1.add_subplot(111)
ax.scatter(predict_wls1_y, data_camels['resid1'] )
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
plt.axhline(y = 0, color = 'r', linewidth = 2)
plt.title("Residual Distribution of WLS")
plt.show()


cc_breus_wls1 = diagnostic.het_breuschpagan(wls_fit1.resid, wls_fit1.model.exog)
print(cc_breus_wls1)


fig, ax = plt.subplots(figsize=(8, 5))
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--',c='k', label="1:1 line")
plt.scatter(data_y,predict_wls1_y,color='blue')
ax.set_xlabel("Calculated discharge")
ax.set_ylabel("Prediction discharge")
plt.show()


# To test on validation data set
x_test = pd.read_csv("C://Users//Administrator//Desktop//results//test.csv")
print(wls_fit1.predict(x_test)) 







## This is multiple logarithmic transformation

data_camels['AREA_ln'] = np.log(data_camels['AREA'])
data_camels['LDP_ln'] = np.log(data_camels['LDP'])
data_camels['DPLBAR_ln'] = np.log(data_camels['DPLBAR'])
data_camels['discharge_ln'] = np.log(data_camels['discharge'])

data_multi_new = data_camels.loc[:,['AREA_ln','DPLBAR_ln','LDP_ln']]



multi_model_new = smf.ols('discharge_ln ~ AREA_ln + DPLBAR_ln + LDP_ln', data=data_camels).fit() 
print(multi_model_new.summary())



predict_multi_y_new = multi_model_new.predict(data_multi_new)



fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(range(len(predict_multi_y_new)),predict_multi_y_new,'b',label="predict")
plt.plot(range(len(predict_multi_y_new)),data_camels['discharge_ln'],'r',label="data")
plt.legend(loc="upper right") 
ax.set_xlabel("the number of catchments")
ax.set_ylabel("ln(discharge)")
plt.title("The prediction of logarithmic transformation")
plt.show()
plt.show()


# model evaluation
rmse_multi_new = (np.sqrt(mean_squared_error(data_camels['discharge_ln'], predict_multi_y_new)))
r2_multi_new = r2_score(data_camels['discharge_ln'], predict_multi_y_new)
print(rmse_multi_new)
print(r2_multi_new)



data_camels['resid2'] = multi_model_new.resid_pearson
fig1 = plt.figure(figsize=(8, 5), dpi=100)
ax = fig1.add_subplot(111)
ax.scatter(predict_multi_y_new, data_camels['resid2'] )
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')

plt.axhline(y = 0, color = 'r', linewidth = 2)
plt.title("Residual Distribution of Logarithmic Transformation")
plt.show()



cc_breus_new = diagnostic.het_breuschpagan(multi_model_new.resid, multi_model_new.model.exog)
print(cc_breus_new)



fig, ax = plt.subplots(figsize=(8, 5))
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--',c='k', label="1:1 line")
plt.scatter(data_camels['discharge_ln'],predict_multi_y_new,color='blue')
ax.set_xlabel("Calculated discharge")
ax.set_ylabel("Prediction discharge")
plt.show()



# To test on validation data set
x_test_ln = pd.read_csv("C://Users//Administrator//Desktop//results//test-ln.csv")
y_ln = multi_model_new.predict(x_test_ln)
print(np.e**y_ln)


