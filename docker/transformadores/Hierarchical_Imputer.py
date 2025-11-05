from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
class HierarchicalImputer(BaseEstimator, TransformerMixin):
    """
    Esta clase implementa una estrategia de imputación jerárquica.
    Primero busca completar valores faltantes según grupos (RegionCluster + fecha).
    Si no encuentra valores, busca por RegionCluster solo.
    Finalmente, si aún quedan NaNs, imputa con la mediana (numéricas) o moda (categóricas) global del conjunto.

    """
    def __init__(self, columnas_num_excluir=None, columnas_cat=None):
        self.columnas_num_excluir = columnas_num_excluir if columnas_num_excluir else ['Cloud9am', 'Cloud3pm', 'Latitude', 'Longitude']
        self.columnas_cat = columnas_cat if columnas_cat else ['Location']
        self.medianas_region_fecha = {}
        self.medianas_region = {}
        self.mediana_global = {}
        self.modas_fecha_cluster = {}
        self.modas_cluster = {}
        self.moda_global = {}

    def fit(self, X, y=None):
        df = X.copy()

        self.num_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(self.columnas_num_excluir).tolist()

        for col in self.num_cols:
            self.medianas_region_fecha[col] = df.groupby(['RegionCluster', 'fecha'])[col].median()
            self.medianas_region[col] = df.groupby('RegionCluster')[col].median()
            self.mediana_global[col] = df[col].median()


        for col in self.columnas_cat:
            self.moda_global[col] = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
            self.modas_fecha_cluster[col] = df.groupby(['fecha', 'RegionCluster'])[col].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            )
            self.modas_cluster[col] = df.groupby('RegionCluster')[col].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            )

        return self

    def transform(self, X):
        df = X.copy()

        for col in self.num_cols:
            df[col] = df.apply(
                lambda row: self._imputar_num(row, col), axis=1
            )

        for col in self.columnas_cat:
            df[col] = df.apply(
                lambda row: self._imputar_cat(row, col), axis=1
            )

        # Asegurar sin NaNs usando global
        for col in self.num_cols:
            df[col] = df[col].fillna(self.mediana_global[col])

        for col in self.columnas_cat:
            df[col] = df[col].fillna(self.moda_global[col])


        return df

    def _imputar_num(self, row, col):
        if pd.isna(row[col]):
            clave = (row['RegionCluster'], row['fecha'])
            if clave in self.medianas_region_fecha[col].index:
                return self.medianas_region_fecha[col][clave]
            elif row['RegionCluster'] in self.medianas_region[col].index:
                return self.medianas_region[col][row['RegionCluster']]
            else:
                return np.nan
        else:
            return row[col]

    def _imputar_cat(self, row, col):
        if pd.isna(row[col]):
            clave = (row['fecha'], row['RegionCluster'])
            if clave in self.modas_fecha_cluster[col].index:
                return self.modas_fecha_cluster[col][clave]
            elif row['RegionCluster'] in self.modas_cluster[col].index:
                return self.modas_cluster[col][row['RegionCluster']]
            else:
                return np.nan
        else:
            return row[col]

