import joblib
import pandas as pd
import warnings
import sklearn
print(sklearn.__version__)
warnings.simplefilter('ignore')
import logging
from sys import stdout
from transformadores.Temporal_GeoTransformer import TemporalGeoTransformer
from transformadores.Hierarchical_Imputer import HierarchicalImputer
from transformadores.Outlier_Capper import OutlierCapper
from transformadores.Cyclic_Encoder_Cleaner import CyclicEncoderCleaner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

pipeline = joblib.load('pipeline.pkl')

logger.info('loaded pipeline')

df_input = pd.read_csv('/files/input.csv')

logger.info('loaded input')

print(df_input.head())

output = pipeline.predict(df_input)

logger.info('made predictions')

pd.DataFrame(output, columns=['RainTomorrow_pred']).to_csv('/files/output.csv', index=False)

logger.info('saved output')

