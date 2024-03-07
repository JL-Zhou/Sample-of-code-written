import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics

import seaborn as sns

# X = 4 * np.random.rand(100, 1) - 2
# y = 4 + 2 * X + 5 * X**2 + 12 * X**2 + 30 * np.random.randn(100, 1) #random noise

# poly_features = PolynomialFeatures(degree=18, include_bias=False)
# X_poly = poly_features.fit_transform(X)


df = pd.read_csv('apple_dataset_reg.csv')

y = df['Basescore']

# X = df[['Access'], ['Complexity'], ['Authentication'], ['Conf.'], ['Integ.'], ['Avail']]

X = df[['Access', 'Complexity', 'Authentication', 'Conf.', 'Integ.', 'Avail.', 'f_impact']]
# X = df[['Access', 'Complexity', 'Authentication', 'Conf.', 'Integ.', 'Avail.']]

# print(X)
# print(df.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
reg = LinearRegression()


# reg.fit(X[['Access', 'Complexity', 'Authentication', 'Conf.', 'Integ.', 'Avail.']], y) #fit, help to find the trend
reg.fit(X_train[['Access', 'Complexity', 'Authentication', 'Conf.', 'Integ.', 'Avail.', 'f_impact']], y_train)

# X_vals = np.linspace(-2, 2, 100).reshape(-1, 1)
# X_vals_poly = poly_features.transform(X_vals) #predict on x values pulley
#
# y_vals = reg.predict(X_vals_poly)

# plt.scatter(X, y)
# plt.plot(X_vals, y_vals, color = 'r')
# plt.show()
print(X_test)
# k = reg.predict([[1, 0.61, 0.704, 0.275, 0.275, 0.275]])
k =[]
k = reg.predict(X_test[['Access', 'Complexity', 'Authentication', 'Conf.', 'Integ.', 'Avail.', 'f_impact']])

print(len(k))
# plt.scatter(X['Impact'], y)
# plt.scatter(X['f_impact'], y)
# plt.scatter(X['Exploitability'], y)
# plt.show()


score = r2_score(y_test, k)
print("The accuracy of our model is {}%".format(round(score, 2) *100))
# X1 = sm.add_constant(X_train)
#
#
# model = sm.OLS(y_train, X1).fit()
# # predictions = model.predict(X1)
#
# print_model = model.summary()
# print(print_model)
print(reg.coef_)
print(reg.intercept_)
# print(reg.params)


meanAbErr = metrics.mean_absolute_error(y_test, k)
meanSqErr = metrics.mean_squared_error(y_test, k)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, k))
print('R squared: {:.2f}'.format(reg.score(X_train, y_train)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

sns.regplot(x=y_test,y=k,ci=None, scatter_kws={"color": "darkblue"}, line_kws={"color": "red"})
plt.xlabel("Basescore")
plt.ylabel("Independent variables")
plt.title("Multiple linear regression plot")
plt.show()