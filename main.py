import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

data = pd.read_csv('DataSet.csv')
data.head()
data.columns


plt.figure(figsize=(16, 8))
plt.scatter(
    data['weight_up_to'],
    data['weight_after'],
    c='black'
)
plt.xlabel("Принято металла")
plt.ylabel("Сдано металла")
plt.show()

X = data['weight_up_to'].values.reshape(-1,1)
y = data['weight_after'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X, y)


print(reg.coef_[0][0])
print(reg.intercept_[0])

print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))


predictions = reg.predict(X)

plt.figure(figsize=(16, 8))
plt.scatter(
    data['weight_up_to'],
    data['weight_after'],
    c='black'
)
plt.plot(
    data['weight_up_to'],
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("Принято металла")
plt.ylabel("Сдано металла")
plt.show()


predictions = reg.predict(X)

plt.figure(figsize=(16, 8))
plt.scatter(
    data['weight_up_to'],
    data['weight_after'],
    c='black'
)
plt.plot(
    data['weight_up_to'],
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("Принято металла")
plt.ylabel("Сдано металла")
plt.show()


Xs = data.drop(['weight_up_to'], axis=1)
y = data['weight_after'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(Xs, y)


print(reg.coef_)
print(reg.intercept_)

print("The linear model is: Y = {:.5} + {:.5}*weight_up_to + {:.5}*weight_after ".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][1]))

X = np.column_stack((data['weight_up_to']))
y = data['weight_after']

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())