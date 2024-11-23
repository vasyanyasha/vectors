import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes
'''from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score'''


def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return [X_train, X_test, y_train, y_test]


def get_regression_data() -> list[np.ndarray]:
    data = load_diabetes()
    X, y = data.data, data.target
    return preprocess(X, y)


def get_classification_data() -> list[np.ndarray]:
    data = load_breast_cancer()
    X, y = data.data, data.target
    return preprocess(X, y)


def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    model = LinearRegression()
    model.fit(X, y)
    return model


def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    model = Ridge()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    model = Lasso()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
    model.fit(X, y)
    return model


def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000)
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

'''Logistic Regression (No Regularization):
Accuracy: 0.9386
              precision    recall  f1-score   support

           0       0.89      0.95      0.92        43
           1       0.97      0.93      0.95        71

    accuracy                           0.94       114
   macro avg       0.93      0.94      0.94       114
weighted avg       0.94      0.94      0.94       114

Logistic Regression (L1 Regularization):
Accuracy: 0.9737
              precision    recall  f1-score   support

           0       0.95      0.98      0.97        43
           1       0.99      0.97      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

Logistic Regression (L2 Regularization):
Accuracy: 0.9737
              precision    recall  f1-score   support

           0       0.98      0.95      0.96        43
           1       0.97      0.99      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

--- Linear Regression ---
Mean Squared Error (MSE): 2900.1936
R² Score: 0.4526

--- Ridge Regression ---
Mean Squared Error (MSE): 2875.6676
R² Score: 0.4572

--- Lasso Regression ---
Mean Squared Error (MSE): 2824.1008
R² Score: 0.4670

For diabetes dataset, we didnt manage to get the definitive regression model that reliably predicts diabetes illness based on information. 
But Ridge and Lasso regression, both provided better results compared to linear one with both R squared and MSE improving. Lasso performed better than Ridge.

For breast cancer dataset, we managed to accurately detect it using all range of models. Regularization certainly improved the scores, getting results that can be described as definitive classification(accuracy>0.95). 
L1 and L2 performed nearly identically, so its not possible to determine which one is better without further examples.
'''