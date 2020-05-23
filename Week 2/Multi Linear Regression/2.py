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



from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit (xx_train,yy_train)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

from sklearn.metrics import r2_score

y_hat= regr.predict(xx_test)
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - yy_test) ** 2))
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - yy_test)))
print("R2-score: %.2f" % r2_score(y_hat , yy_test) )