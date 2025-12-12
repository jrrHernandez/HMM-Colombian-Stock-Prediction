import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime

def load_stock_data(csv_file_path):
    """
    Carga los datos del archivo .csv (ej. Dataecopetrol.csv)
    y los adapta al formato requerido (Open, High, Low, Close).
    """
    try:
        data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Archivo .csv no encontrado en {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error al leer el CSV: {e}")
        return None

    # --- Mapeo de Columnas ---
    # Asunción crítica: Usamos 'px_close_1d' (cierre anterior) como 'Open'
    # ya que 'Open' no está en el CSV.
    column_map = {
        'date': 'Date',
        'px_close_1d': 'Open', # ¡Asunción importante!
        'px_high': 'High',
        'px_low': 'Low',
        'px_last': 'Close'  # 'px_last' es el precio de cierre
    }
    
    # Comprobar si las columnas necesarias existen
    required_cols = ['date', 'px_close_1d', 'px_high', 'px_low', 'px_last']
    
    if not all(col in data.columns for col in required_cols):
        print("Error: El CSV no tiene todas las columnas esperadas.")
        print(f"Se esperaban: {required_cols}")
        print(f"Se encontraron: {data.columns.tolist()}")
        return None
        
    df = data[required_cols].rename(columns=column_map)
    
    # Convertir 'Date' a datetime y ponerla como índice
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        print(f"Error al convertir la columna 'date'. Asegúrate de que tenga un formato estándar. Error: {e}")
        return None
        
    df = df.set_index('Date')
    
    # Los datos en el snippet parecían estar en orden descendente.
    # Los ordenamos por fecha (ascendente) para que funcione el modelo.
    df = df.sort_index(ascending=True)
    
    # Asegurarse que los datos son numéricos
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Eliminar filas con NaN (especialmente la primera fila de 'Open'
    # que será NaN debido a 'px_close_1d')
    df = df.dropna()

    if len(df) == 0:
        print("Error: No quedaron datos válidos después de procesar el CSV.")
        return None
        
    print(f"Datos CSV cargados y procesados. {len(df)} filas válidas encontradas.")
    print("Primeras filas de datos:")
    print(df.head())
    
    return df

def get_edges(data, n_points, limits, use_dynamic=False):
    """
    Genera los bordes (edges) para la discretización. (Sin cambios)
    """
    if use_dynamic:
        return np.quantile(data.dropna(), np.linspace(0, 1, n_points + 1))
    else:
        return np.linspace(limits[0], limits[1], n_points + 1)

def discretize_data(data, edges):
    """
    Discretiza los datos usando los bordes. (Sin cambios)
    """
    discretized = np.digitize(data, edges)
    discretized[discretized == 0] = 0 
    discretized[discretized == len(edges)] = 0 
    return discretized

def map_3d_to_1d(idx_i, idx_j, idx_k, D1, D2, D3):
    """
    Mapea índices 3D (1-based) a un índice 1D (0-based). (Sin cambios)
    """
    if idx_i == 0 or idx_j == 0 or idx_k == 0:
        return -1 
        
    i, j, k = idx_i - 1, idx_j - 1, idx_k - 1
    return i * (D2 * D3) + j * D3 + k

def map_1d_to_3d(idx_1d, D1, D2, D3):
    """
    Mapea un índice 1D (0-based) a índices 3D (1-based). (Sin cambios)
    """
    k = idx_1d % D3
    j = (idx_1d // D3) % D2
    i = (idx_1d // (D3 * D2))
    return i + 1, j + 1, k + 1