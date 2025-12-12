import numpy as np
import datetime

# ----- Configuración de la acción-----------#
#Nombre de la acción deseada
STOCK_NAME = "dataTERPEL.csv" 



# Ponemos TRAIN = True para entrenar el modelo con los nuevos datos
TRAIN = True 

if not TRAIN:
    # Esto se ignora si TRAIN = True, pero lo dejamos
    FILENAME = "train/hmmtrain-2023-07-17-02-52-06.mat" 
else:
    FILENAME = None

# --- Configuración de Secuencias de Entrenamiento ---
# 
SHIFT_WINDOW_BY_ONE = True

# --- Períodos de Fechas ---
# Formato YYYY-MM-DD
START_TRAIN_DATE = '2022-10-12'
END_TRAIN_DATE   = '2024-12-03'

START_PREDICTION_DATE = '2024-12-04'
# Dejar en None para predecir hasta el final de los datos
END_PREDICTION_DATE   = None 

# --- Configuración de Discretización ---
USE_DYNAMIC_EDGES = True

DISCRETIZATION_POINTS = [10, 5, 5]
TOTAL_DISCRETIZATION_POINTS = np.prod(DISCRETIZATION_POINTS)

STATIC_EDGES_LIMITS = [
    (-0.1, 0.1),  # fracChange
    (0, 0.1),     # fracHigh
    (0, 0.1)      # fracLow
]

# --- Configuración del Modelo HMM ---
UNDERLYING_STATES =4  # Número de estados ocultos
MIXTURES_NUMBER = 4   # Número de componentes de mezcla (GMM) por estado
LATENCY = 10           # Días (longitud de la secuencia de observación)

# --- Configuración de Entrenamiento ---
MAX_ITER = 1000 # Iteraciones máximas para hmmtrain

