from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.cluster import KMeans

class TemporalGeoTransformer(BaseEstimator, TransformerMixin):
    """
    Esta clase transforma información geográfica y temporal.
    A partir de coordenadas (Latitude, Longitude), agrupa ubicaciones en clusters regionales mediante KMeans.
    Además extrae información de la fecha (año, mes, día) y crea una columna 'fecha' simplificada.
    """

    def __init__(self, coord_df, n_clusters=9):
        self.coord_df = coord_df
        self.n_clusters = n_clusters
        self.kmeans_model = None

    def fit(self, X, y=None):
        # Unir coordenadas
        df = X.merge(self.coord_df, on='Location', how='left')

        # Obtener coordenadas únicas
        ciudades_unicas = df[['Location', 'Latitude', 'Longitude']].drop_duplicates()

        # Entrenar KMeans
        X_coords = ciudades_unicas[['Latitude', 'Longitude']].values
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self.kmeans_model.fit(X_coords)

        # Guardar asignación para cada ciudad
        ciudades_unicas['RegionCluster'] = self.kmeans_model.predict(X_coords)
        self.city_cluster_map = ciudades_unicas.set_index('Location')['RegionCluster'].to_dict()
        return self

    def transform(self, X):
        df = X.copy()

        # Parsear fecha
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['mes'] = df['Date'].dt.month
        df['anio'] = df['Date'].dt.year
        df['fecha'] = df['Date'].dt.date


        # Agregar coordenadas
        df = df.merge(self.coord_df, on='Location', how='left')

        # Asignar cluster
        df['RegionCluster'] = df['Location'].map(self.city_cluster_map)

        return df
