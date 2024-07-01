import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
import itertools
from matplotlib import pyplot as plt
import shap

data = pd.read_csv(r'data.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

kernels = [RBF(), Matern(), RationalQuadratic(), DotProduct()]
best_mse = float('inf')
best_r2 = float('-inf')
best_mae = float('inf')
best_kernel = None


for kernel in kernels:

    model = GaussianProcessRegressor(kernel=kernel)
    param_space = {
        'alpha': (1e-6, 1e-2, 'log-uniform'),
        'n_restarts_optimizer': (5, 20)
    }
    bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, n_iter=10, cv=5,
                                 scoring='neg_mean_squared_error', random_state=42)

    bayes_search.fit(X_train, y_train)

    y_pred = bayes_search.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    if mse < best_mse:
        best_mse = mse
        best_r2 = r2
        best_mae = mae
        best_kernel = kernel
        best_params = bayes_search.best_params_


print("best kernel:", best_kernel)
print("best mse:", best_mse)
print("best r^2:", best_r2)
print("best mae:", best_mae)
print("best params:", best_params)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)



def generate_continuous_list(start, end, step):
    result = []
    current = start
    while current <= end:
        result.append(current)
        current += step
    return result

Time_range = generate_continuous_list(10, 120, 1)
Temperature_range = generate_continuous_list(25, 75, 1)


combinations = list(
    itertools.product(Time_range, Temperature_range))

combinations_df = pd.DataFrame(combinations,
                               columns=['Time', 'Temperature'])

predicted_values = bayes_search.predict(combinations_df)

max_index = predicted_values.argmax()
best_combination = combinations[max_index]
max_value = predicted_values[max_index]

print("The best values for the independent variables are:")
print("Time: {:.2f}".format(best_combination[0]))
print("Temperature: {:.2f}".format(best_combination[1]))
print("Maximum predicted value: {:.2f}".format(max_value))