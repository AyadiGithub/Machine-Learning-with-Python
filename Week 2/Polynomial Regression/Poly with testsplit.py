import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


df = pd.read_csv(r'FuelConsumptionCo2.csv')

# take a look at the dataset
df.head()

# summarize the data
df.describe()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

from sklearn.model_selection import train_test_split

xx = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
yy = cdf[['CO2EMISSIONS']]
xx_train, xx_test,yy_train, yy_test = train_test_split(xx, yy, test_size=0.20, random_state=0)


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
xx_train = np.asanyarray(xx_train)
yy_train = np.asanyarray(yy_train)

xx_test = np.asanyarray(xx_test)
yy_test = np.asanyarray(yy_test)

poly = PolynomialFeatures(degree=2)
yy_train_ = poly.fit_transform(xx_train)
yy_train_

clf = linear_model.LinearRegression()
train_y_ = clf.fit(yy_train_, yy_train)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)


from sklearn.metrics import r2_score

xx_test_poly = poly.fit_transform(xx_test)
yy_test_= clf.predict(xx_test_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(yy_test_ - yy_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((yy_test_ - yy_test) ** 2))
print("R2-score: %.2f" % r2_score(yy_test_ , yy_test) )

