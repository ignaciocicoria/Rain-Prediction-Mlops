from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Esta clase detecta y trata valores extremos en variables numéricas.
    Para las columnas en 'self.vars_cap' utiliza el método del rango intercuartílico (IQR) para limitar los valores a un rango razonable.
    Cualquier valor fuera de los límites se lo iguala al limite mismo. Para las columnas en 'self.vars_log' aplica transformaciones logarítmicas.
    A las columnas 'Rainfall' y 'Evaporation' las categoriza en 'bajo', 'medio' y 'alto'.

    """
    def __init__(self):
        self.vars_cap = [
            'Temp9am', 'Temp3pm', 'Pressure9am', 'Pressure3pm',
            'Humidity9am', 'MinTemp', 'MaxTemp',
            'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm'
        ]
        self.vars_log = ['WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']
        self.limites_iqr = {}

    def fit(self, X, y=None):
        # Calcular límites de IQR para cada variable
        for col in self.vars_cap:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.limites_iqr[col] = (lower, upper)
        return self

    def transform(self, X):
        X_trans = X.copy()

        # Aplicar capping IQR
        for col in self.vars_cap:
            lower, upper = self.limites_iqr[col]
            X_trans[col + '_cap'] = np.where(
                X_trans[col] > upper, upper,
                np.where(X_trans[col] < lower, lower, X_trans[col])
            )

        # Aplicar log1p
        for col in self.vars_log:
            X_trans[col + '_log'] = np.log1p(X_trans[col])

        # Binning de variables continuas a categóricas
        X_trans['Rainfall_cat'] = pd.cut(
            X_trans['Rainfall'],
            bins=[-float('inf'), 10, 30, float('inf')],
            labels=['bajo', 'medio', 'alto']
        )

        X_trans['Evaporation_cat'] = pd.cut(
            X_trans['Evaporation'],
            bins=[-float('inf'), 3, 7, float('inf')],
            labels=['bajo', 'medio', 'alto']
        )

        return X_trans
