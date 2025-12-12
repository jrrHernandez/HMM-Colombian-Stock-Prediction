import numpy as np
from hmmlearn import hmm

def hmm_predict_observation(obs_seq, trans_matrix, emiss_matrix, 
                            possible_observations=None, 
                            dynamic_window=True, verbose=False):
    """
    Traducción de hmmPredictObservation.
    Predice la siguiente observación maximizando la probabilidad de la secuencia.
    
    Nota: obs_seq y possible_observations deben ser 0-based.
    """
    
    if possible_observations is None:
        # Esta lógica es la alternativa (no usada por main.m)
        # Requeriría una implementación más compleja.
        raise NotImplementedError("La predicción sin 'possibleObservations' no está implementada.")

    n_states = trans_matrix.shape[0]
    
    # Configurar un modelo HMM temporal con los parámetros entrenados
    model = hmm.CategoricalHMM(n_components=n_states, n_features=emiss_matrix.shape[1])
    model.transmat_ = trans_matrix
    model.emissionprob_ = emiss_matrix
    # Asumir probabilidades iniciales uniformes (MATLAB hmmdecode hace algo similar)
    model.startprob_ = np.full(n_states, 1.0 / n_states)

    max_log_p_seq = -np.inf
    most_likely_obs = np.nan
    
    current_obs_seq = list(obs_seq) # Copiar la secuencia
    
    if dynamic_window:
        num_trials = len(current_obs_seq) - 3 if len(current_obs_seq) > 3 else 1
    else:
        num_trials = 1

    for _ in range(num_trials):
        max_log_p_seq = -np.inf # Reiniciar para este trial
        
        for possible_obs in possible_observations:
            seq_to_test = np.array(current_obs_seq + [possible_obs]).reshape(-1, 1)
            
            try:
                log_p_seq = model.score(seq_to_test)
            except Exception as e:
                # Esto puede pasar si una proba es 0
                log_p_seq = -np.inf 

            if log_p_seq > max_log_p_seq:
                max_log_p_seq = log_p_seq
                most_likely_obs = possible_obs
        
        if max_log_p_seq > -np.inf:
            # Convergencia alcanzada
            break
        else:
            # Si no converge (todo es -inf), acortar la ventana
            if len(current_obs_seq) > 1:
                current_obs_seq = current_obs_seq[1:]
            else:
                break # No se puede acortar más

    if verbose:
        print(f"Log probability of sequence: {max_log_p_seq:.4f}")

    return most_likely_obs