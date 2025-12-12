import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from scipy.io import loadmat
import os
import joblib
from datetime import datetime

# Importar configuración y funciones existentes
import config as cfg
from helper_functions import (load_stock_data, get_edges, discretize_data, 
                              map_3d_to_1d, map_1d_to_3d)
from hmm_predictor import hmm_predict_observation

def main():
    print("Iniciando mainBIC.py (Selección de Modelo)...")
    
    # --- Cargar datos ---
    df = load_stock_data(cfg.STOCK_NAME)
    if df is None:
        return
        
    # --- Índices de Fechas ---
    try:
        start_train_idx_loc = df.index.get_loc(cfg.START_TRAIN_DATE)
        end_train_idx_loc = df.index.get_loc(cfg.END_TRAIN_DATE)
        start_pred_idx_loc = df.index.get_loc(cfg.START_PREDICTION_DATE)
        
        if cfg.END_PREDICTION_DATE:
            end_pred_idx_loc = df.index.get_loc(cfg.END_PREDICTION_DATE)
        else:
            end_pred_idx_loc = len(df) - 1 
            
    except KeyError as e:
        print(f"Error: Fecha no encontrada en los datos: {e}")
        return

    train_df = df.iloc[start_train_idx_loc : end_train_idx_loc + 1]
    
    # --- Definición de Bordes y Preprocesamiento ---
    D1, D2, D3 = cfg.DISCRETIZATION_POINTS
    
    # Lógica de carga o generación de bordes
    if not cfg.TRAIN and cfg.FILENAME:
        print(f"Cargando modelo pre-entrenado de {cfg.FILENAME}")
        try:
            if cfg.FILENAME.endswith('.joblib'):
                model_data = joblib.load(cfg.FILENAME)
            elif cfg.FILENAME.endswith('.mat'):
                 model_data = loadmat(cfg.FILENAME)
                 model_data['edgesFChange'] = model_data['edgesFChange'].flatten()
                 model_data['edgesFHigh'] = model_data['edgesFHigh'].flatten()
                 model_data['edgesFLow'] = model_data['edgesFLow'].flatten()
            else:
                raise ValueError("Formato desconocido.")
                
            ESTTR = model_data['ESTTR']
            ESTEMIT = model_data['ESTEMIT']
            edges_f_change = model_data['edgesFChange']
            edges_f_high = model_data['edgesFHigh']
            edges_f_low = model_data['edgesFLow']
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return
    else:
        # Generar bordes con datos completos
        print("Generando bordes para discretización...")
        open_prices_no_zero = df['Open'].mask(df['Open'] == 0, np.nan)

        full_frac_change = (df['Close'] - open_prices_no_zero) / open_prices_no_zero
        full_frac_high = (df['High'] - open_prices_no_zero) / open_prices_no_zero
        full_frac_low = (open_prices_no_zero - df['Low']) / open_prices_no_zero

        edges_f_change = get_edges(full_frac_change, D1, cfg.STATIC_EDGES_LIMITS[0], cfg.USE_DYNAMIC_EDGES)
        edges_f_high = get_edges(full_frac_high, D2, cfg.STATIC_EDGES_LIMITS[1], cfg.USE_DYNAMIC_EDGES)
        edges_f_low = get_edges(full_frac_low, D3, cfg.STATIC_EDGES_LIMITS[2], cfg.USE_DYNAMIC_EDGES)
        
        # Datos de entrenamiento
        open_train_no_zero = train_df['Open'].mask(train_df['Open'] == 0, np.nan)
        frac_change = (train_df['Close'] - open_train_no_zero) / open_train_no_zero
        frac_high = (train_df['High'] - open_train_no_zero) / open_train_no_zero
        frac_low = (open_train_no_zero - train_df['Low']) / open_train_no_zero

        # Variable necesaria para GMM
        continuous_observations_3d = np.stack((frac_change, frac_high, frac_low), axis=1)

        # Discretizar
        frac_change_discrete = discretize_data(frac_change, edges_f_change)
        frac_high_discrete = discretize_data(frac_high, edges_f_high)
        frac_low_discrete = discretize_data(frac_low, edges_f_low)
       
    # Mapeo a 1D
    discrete_observations_1d = np.array([
        map_3d_to_1d(fc, fh, fl, D1, D2, D3)
        for fc, fh, fl in zip(frac_change_discrete, frac_high_discrete, frac_low_discrete)
    ])
    
    # Filtrar válidos
    valid_obs_mask = (discrete_observations_1d != -1)
    discrete_observations_1d = discrete_observations_1d[valid_obs_mask]
    if 'continuous_observations_3d' in locals():
        continuous_observations_3d = continuous_observations_3d[valid_obs_mask]
    
    if len(discrete_observations_1d) == 0:
        print("Error: No hay observaciones válidas.")
        return

    # ==============================================================================
    # SECCIÓN BIC: ENTRENAMIENTO CON SELECCIÓN DE MODELO
    # ==============================================================================
    if cfg.TRAIN:
        print("\n" + "="*40)
        print(" INICIANDO SELECCIÓN DE MODELO (BIC) ")
        print("="*40)
        
        # --- Configuración del Grid Search ---
        min_states = 2
        max_states = 8 
        states_range = range(min_states, max_states + 1)
        
        bic_history = []
        best_bic = np.inf
        best_n_states = 0
        
        # Variables para almacenar el mejor resultado
        best_model = None
        best_ESTTR = None
        best_ESTEMIT = None

        # Preparar datos de entrenamiento (una sola vez)
        training_set = []
        lengths = []
        
        if cfg.SHIFT_WINDOW_BY_ONE:
            total_sequences = len(discrete_observations_1d) - cfg.LATENCY + 1
            for i in range(total_sequences):
                seq = discrete_observations_1d[i : i + cfg.LATENCY]
                training_set.extend(seq)
                lengths.append(cfg.LATENCY)
        else:
            total_sequences = len(discrete_observations_1d) // cfg.LATENCY
            for i in range(total_sequences):
                start = i * cfg.LATENCY
                end = start + cfg.LATENCY
                seq = discrete_observations_1d[start : end]
                training_set.extend(seq)
                lengths.append(cfg.LATENCY)

        training_set_concat = np.array(training_set).reshape(-1, 1)
        n_samples = len(training_set_concat)
        
        # Pre-calcular el grid de coordenadas para las emisiones (optimización)
        grid_x, grid_y, grid_z = np.meshgrid(edges_f_change[:-1], edges_f_high[:-1], edges_f_low[:-1], indexing='ij')
        observation_grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

        # --- BUCLE PRINCIPAL ---
        for n_states in states_range:
            print(f"\n[Evaluando Modelo con {n_states} Estados Ocultos]...")
            
            try:
                # 1. Ajustar GMM
                n_components_gmm = cfg.MIXTURES_NUMBER * n_states
                
                gmm = GaussianMixture(n_components=n_components_gmm, 
                                      covariance_type='diag', 
                                      reg_covar=1e-6,
                                      n_init=3,
                                      random_state=42).fit(continuous_observations_3d)
                
                # 2. Ordenar componentes
                sorted_idx = np.argsort(gmm.means_[:, 0])
                sorted_mu = gmm.means_[sorted_idx]
                sorted_sigma = gmm.covariances_[sorted_idx]
                sorted_weights = gmm.weights_[sorted_idx]

                # 3. Calcular Matriz de Emisiones
                emission_probs = np.zeros((n_states, cfg.TOTAL_DISCRETIZATION_POINTS))
                
                for i in range(n_states):
                    start_idx = i * cfg.MIXTURES_NUMBER
                    end_idx = (i + 1) * cfg.MIXTURES_NUMBER
                    
                    # Extraer sub-mezcla para este estado
                    state_weights = sorted_weights[start_idx:end_idx]
                    state_weights /= (state_weights.sum() + 1e-10) # Normalizar
                    
                    state_gmm = GaussianMixture(n_components=cfg.MIXTURES_NUMBER, covariance_type='diag')
                    state_gmm.means_ = sorted_mu[start_idx:end_idx]
                    state_gmm.covariances_ = sorted_sigma[start_idx:end_idx]
                    state_gmm.weights_ = state_weights
                    state_gmm.precisions_cholesky_ = np.sqrt(1.0 / state_gmm.covariances_)
                    
                    # Score samples (log prob) -> exp
                    log_prob = state_gmm.score_samples(observation_grid)
                    emission_probs[i, :] = np.exp(log_prob) + 1e-10

                # Normalizar filas de emisiones
                emission_probs /= emission_probs.sum(axis=1, keepdims=True)

                # 4. Entrenar HMM
                model = hmm.CategoricalHMM(n_components=n_states,
                                           n_features=cfg.TOTAL_DISCRETIZATION_POINTS,
                                           n_iter=cfg.MAX_ITER,
                                           tol=1e-4,
                                           verbose=False,
                                           random_state=42,
                                           params='ste',
                                           init_params='')
                
                model.startprob_ = np.full(n_states, 1.0 / n_states)
                model.transmat_ = np.full((n_states, n_states), 1.0 / n_states)
                model.emissionprob_ = emission_probs
                
                model.fit(training_set_concat, lengths)
                
                # 5. Calcular BIC
                log_likelihood = model.score(training_set_concat, lengths)
                
                n_features = cfg.TOTAL_DISCRETIZATION_POINTS
                n_params = (n_states - 1) + \
                           (n_states * (n_states - 1)) + \
                           (n_states * (n_features - 1))
                           
                bic = -2 * log_likelihood + n_params * np.log(n_samples)
                bic_history.append(bic)
                
                print(f"   LogL: {log_likelihood:.2f} | k: {n_params} | BIC: {bic:.2f}")
                
                # Guardar si es el mejor
                if bic < best_bic:
                    print(f"   -> ¡Nuevo récord! (Anterior: {best_bic:.2f})")
                    best_bic = bic
                    best_n_states = n_states
                    best_model = model
                    best_ESTTR = model.transmat_
                    best_ESTEMIT = model.emissionprob_
                    
            except Exception as e:
                print(f"   Error con {n_states} estados: {e}")
                bic_history.append(np.inf)
        
        print("="*40)
        print(f"GANADOR: {best_n_states} Estados Ocultos (BIC: {best_bic:.2f})")
        print("="*40)

        # Visualizar Curva BIC
        plt.figure(figsize=(10, 5))
        plt.plot(states_range, bic_history, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Número de Estados Ocultos')
        plt.ylabel('BIC Score (Menor es mejor)')
        plt.title(f'Selección de Modelo BIC (Ganador: {best_n_states})')
        plt.grid(True, linestyle='--', alpha=0.7)
        winner_idx = list(states_range).index(best_n_states)
        plt.plot(best_n_states, best_bic, 'ro', markersize=12, label='Mejor Modelo')
        plt.legend()
        plt.show() 

        # Asignar variables ganadoras para la predicción
        ESTTR = best_ESTTR
        ESTEMIT = best_ESTEMIT
        
        # Guardar el mejor modelo
        if not os.path.exists("train"):
            os.makedirs("train")
        
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename_out = f"train/hmmBIC-states{best_n_states}-{timestamp}.joblib"
        
        joblib.dump({
            "ESTTR": ESTTR,
            "ESTEMIT": ESTEMIT,
            "best_n_states": best_n_states,
            "bic_score": best_bic,
            "edgesFChange": edges_f_change,
            "edgesFHigh": edges_f_high,
            "edgesFLow": edges_f_low
        }, filename_out)
        print(f"Mejor modelo guardado en: {filename_out}")


    # ==============================================================================
    # PREDICCIÓN (Idéntico a main.py, usando ESTTR y ESTEMIT del modelo ganador)
    # ==============================================================================
    print("\nIniciando Predicción...")
    
    prediction_indexes = df.index[start_pred_idx_loc : end_pred_idx_loc + 1]
    prediction_length = len(prediction_indexes)
    
    predicted_observations_3d = np.full((prediction_length, 3), np.nan)
    predicted_close = np.full(prediction_length, np.nan)
    possible_obs_1d = np.arange(cfg.TOTAL_DISCRETIZATION_POINTS)

    for i in range(prediction_length):
        current_prediction_idx_loc = start_pred_idx_loc + i
        
        window_start_idx_loc = current_prediction_idx_loc - cfg.LATENCY + 1
        window_end_idx_loc = current_prediction_idx_loc
        
        if window_start_idx_loc < 0:
            continue

        window_df = df.iloc[window_start_idx_loc : window_end_idx_loc]
        open_win_no_zero = window_df['Open'].mask(window_df['Open'] == 0, np.nan)
        
        w_frac_change = (window_df['Close'] - open_win_no_zero) / open_win_no_zero
        w_frac_high = (window_df['High'] - open_win_no_zero) / open_win_no_zero
        w_frac_low = (open_win_no_zero - window_df['Low']) / open_win_no_zero

        w_fc_disc = discretize_data(w_frac_change, edges_f_change)
        w_fh_disc = discretize_data(w_frac_high, edges_f_high)
        w_fl_disc = discretize_data(w_frac_low, edges_f_low)

        current_window_1d = np.array([
            map_3d_to_1d(fc, fh, fl, D1, D2, D3)
            for fc, fh, fl in zip(w_fc_disc, w_fh_disc, w_fl_disc)
        ])
        
        current_window_1d = current_window_1d[current_window_1d != -1]
        
        if len(current_window_1d) < 1: continue

        # Mostrar progreso cada 10 iteraciones
        if i % 10 == 0:
            print(f"Prediciendo {i+1}/{prediction_length}...", end='\r')
        
        predicted_obs_1d = hmm_predict_observation(
            current_window_1d, ESTTR, ESTEMIT, 
            possible_observations=possible_obs_1d, verbose=False
        )

        if not np.isnan(predicted_obs_1d):
            pred_fc_idx, pred_fh_idx, pred_fl_idx = map_1d_to_3d(int(predicted_obs_1d), D1, D2, D3)
            
            pred_fc_val = edges_f_change[pred_fc_idx - 1]
            predicted_observations_3d[i, :] = [pred_fc_val, 
                                               edges_f_high[pred_fh_idx - 1], 
                                               edges_f_low[pred_fl_idx - 1]]
            
            current_open = df.iloc[current_prediction_idx_loc]['Open']
            predicted_close[i] = current_open * (1 + pred_fc_val)

    print("\nPredicción finalizada.")
    
    # ==============================================================================
    # GRÁFICOS Y RESULTADOS (Sin Dummy Investor)
    # ==============================================================================
    results_df = df.iloc[start_pred_idx_loc : end_pred_idx_loc + 1].copy()
    results_df['Predicted_Close'] = predicted_close
    
    # --- 1. Candlestick (Sin cambios mayores) ---
    actual_dir = np.sign(results_df['Close'] - results_df['Open']).replace(0, 1)
    pred_dir = np.sign(results_df['Predicted_Close'] - results_df['Open']).replace(0, 1)
    is_good = (actual_dir == pred_dir)
    
    green_dots = results_df['Predicted_Close'].where(is_good, np.nan)
    red_dots = results_df['Predicted_Close'].where(~is_good, np.nan)
    
    ap_dots = []
    if not green_dots.isnull().all():
        ap_dots.append(mpf.make_addplot(green_dots, type='scatter', marker='.', color='#378333', markersize=50))
    if not red_dots.isnull().all():
        ap_dots.append(mpf.make_addplot(red_dots, type='scatter', marker='.', color='#A80303', markersize=50))
        
    mpf.plot(results_df, type='candle', style='yahoo', 
             title=f"{cfg.STOCK_NAME} - Predicciones (Modelo óptimo: {best_n_states if cfg.TRAIN else 'Cargado'} estados)",
             addplot=ap_dots, figscale=1.5, warn_too_much_data=10000)

    # --- 2. Gráfico Comparativo (Real vs Predicho) ---
    fig2, ax1 = plt.subplots(figsize=(15, 8))
    
    # Eje Izquierdo: Precio
    ax1.plot(results_df.index, results_df['Close'], 'b-', alpha=0.5, label='Precio Acción', linewidth=1)
    
    # Dibujar líneas de predicción (verde/rojo)
    for k in range(len(results_df) - 1):
        if np.isnan(results_df['Predicted_Close'].iloc[k+1]): continue
        c = 'g' if is_good.iloc[k+1] else 'r'
        ax1.plot([results_df.index[k], results_df.index[k+1]], 
                 [results_df['Close'].iloc[k], results_df['Predicted_Close'].iloc[k+1]], 
                 f'{c}.-', markersize=3, linewidth=0.5, alpha=0.7)

    ax1.set_ylabel(f'Precio Acción ({cfg.STOCK_NAME})', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Leyenda personalizada
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='green', marker='.', lw=1),
                    Line2D([0], [0], color='red', marker='.', lw=1)]
    ax1.legend(custom_lines, ['Precio Real', 'Predicción Correcta', 'Predicción Incorrecta'], loc='upper left')

    plt.title(f"Predicciones del Modelo vs Mercado Real")
    plt.tight_layout()
    plt.show()

    # --- Métricas Finales ---
    valid = results_df.dropna(subset=['Predicted_Close'])
    if len(valid) > 0:
        acc = is_good.loc[valid.index].mean() * 100
        mape = np.mean(np.abs((valid['Close'] - valid['Predicted_Close']) / valid['Close'])) * 100
        print("\n" + "="*30)
        print(" RESULTADOS FINALES ")
        print("="*30)
        print(f"Precisión Direccional: {acc:.2f}%")
        print(f"MAPE: {mape:.2f}%")
        print("="*30)

if __name__ == "__main__":
    main()