import numpy as np
import os
import scipy.io
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping
from copy import deepcopy

def train_L24O_cv(model_, X, y, sbjs, model_args=None, compile_args=None, folds=None, model_name=''):
    cv_scores = []

    for fold, (train_subjects, test_subjects) in enumerate(folds):
        print(f"Fold {fold+1}/{len(folds)}. Test subjects: {test_subjects}")

        train_idx = [i for i, sbj in enumerate(sbjs) if sbj in train_subjects]
        test_idx = [i for i, sbj in enumerate(sbjs) if sbj in test_subjects]

        if model_name == 'spatio_temporal':
            freq, temp, spat = X

            X_train = [freq[train_idx], temp[train_idx], spat[train_idx]]
            X_test = [freq[test_idx], temp[test_idx], spat[test_idx]]

        else:
            X_train, X_test = X[train_idx], X[test_idx]
            
        y_train, y_test = y[train_idx], y[test_idx]

        if model_args is not None:
                # Crear y compilar el modelo
                model = model_(**model_args)

                compile_args_local = deepcopy(compile_args)

                if callable(compile_args_local['optimizer']):
                    compile_args_local['optimizer'] = compile_args_local['optimizer']()

                model.compile(**compile_args_local)

                early_stopping = EarlyStopping(
                monitor='val_loss', patience=20, min_delta=0.01, restore_best_weights=True
                )
                    
                # Entrenar el modelo
                model.fit(
                    X_train, y_train, 
                    epochs=100, 
                    validation_data=(X_test, y_test), 
                    verbose=0, 
                    batch_size=16,
                    callbacks=[early_stopping]
                )
        else:
                model = model_
                model.fit(X_train, y_train)

        # Predicciones

        if model_name == 'GMRRNet':
            y_pred_probs = model.predict(X_test)[0]
        else:    
            y_pred_probs = model.predict(X_test)

        y_pred = np.argmax(y_pred_probs, axis=1) if len(y_pred_probs.shape) > 1 else (y_pred_probs > 0.5).astype(int).flatten()
        y_true = y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1)

        # Evaluaciones
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        kappa = cohen_kappa_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')

        fold_metrics = {
            'accuracy': acc,
            'recall': recall,
            'precision': precision,
            'kappa': kappa,
            'auc': auc
        }

        print(f"Fold metrics: {fold_metrics}")

        cv_scores.append(fold_metrics)

    return cv_scores

def train_LOSO(model_, X, y, sbjs, model_args=None, compile_args=None, sbj_in=None, sbj_fin=None, model_name=''):
    logo = LeaveOneGroupOut()
    resultados = {}  # Diccionario para almacenar las métricas por sujeto
    
    i = 0    
    for train_idx, test_idx in logo.split(y, groups=sbjs):
        i += 1
        if sbj_in <= i <= sbj_fin:
            # Obtener el sujeto que se está utilizando como conjunto de prueba
            sujeto_prueba = list(set(sbjs[j] for j in test_idx))[0]
            
            # Imprimir el sujeto que está siendo evaluado
            print(f"Evaluando modelo para el sujeto #{i}: {sujeto_prueba}")
        
            # Dividir los datos en entrenamiento y prueba
            if model_name == 'spatio_temporal':
                freq, temp, spat = X
    
                X_train = [freq[train_idx], temp[train_idx], spat[train_idx]]
                X_test = [freq[test_idx], temp[test_idx], spat[test_idx]]
    
            else:
                X_train, X_test = X[train_idx], X[test_idx]

            y_train, y_test = np.array([y[j] for j in train_idx]), np.array([y[j] for j in test_idx])
            
            # Crear el callback de early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',        
                patience=10,               
                min_delta=0.01,             
                restore_best_weights=True  
            )
            
            if model_args is not None:
                    # Crear y compilar el modelo
                    model = model_(**model_args)
    
                    compile_args_local = deepcopy(compile_args)
    
                    if callable(compile_args_local['optimizer']):
                        compile_args_local['optimizer'] = compile_args_local['optimizer']()
    
                    model.compile(**compile_args_local)
    
                    early_stopping = EarlyStopping(
                    monitor='val_loss', patience=10, min_delta=0.01, restore_best_weights=True
                    )
                        
                    # Entrenar el modelo
                    model.fit(
                        X_train, y_train, 
                        epochs=50, 
                        validation_data=(X_test, y_test), 
                        verbose=0, 
                        batch_size=16,
                        callbacks=[early_stopping]
                    )
            else:
                    model = model_
                    model.fit(X_train, y_train)
    
            # Predicciones
    
            if model_name == 'GMRRNet':
                y_pred_probs = model.predict(X_test)[0]
            else:    
                y_pred_probs = model.predict(X_test)

            
            y_pred = np.argmax(y_pred_probs, axis=1) if len(y_pred_probs.shape) > 1 else (y_pred_probs > 0.5).astype(int).flatten()
            y_true = y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1)

            # Calcular métricas
            acc = accuracy_score(y_true, y_pred)
            # kappa = cohen_kappa_score(y_true, y_pred)
            # try:
            #     auc = roc_auc_score(y_true, y_pred_probs[:,1]) if y_pred_probs.shape[-1] > 1 else roc_auc_score(y_true, y_pred_probs)
            # except Exception as e:
            #     print(f"No se pudo calcular AUC para sujeto {sujeto_prueba}: {e}")
            #     auc = np.nan
            # precision = precision_score(y_true, y_pred, zero_division=0)
            # recall = recall_score(y_true, y_pred, zero_division=0)

            # Guardar en el diccionario
            resultados[sujeto_prueba] = {
                'accuracy': acc,
                # 'kappa': kappa,
                # 'auc': auc,
                # 'precision': precision,
                # 'recall': recall
            }
            
            print(f"Métricas para {sujeto_prueba}: {resultados[sujeto_prueba]}\n")
    
    if resultados:
        # Mostrar resumen
        print("Resumen de métricas por sujeto:")
        for sujeto, metricas in resultados.items():
            print(f"Sujeto {sujeto}: {metricas}")
    else:
        print("No se entrenó ningún modelo dentro del rango especificado.")
    
    return resultados  # <- Ahora devuelve el diccionario de resultados


def segmentar_senales(db, labels):
    """
    Divide las señales EEG en segmentos de 512 instantes con un traslape del 50%.
    
    Args:
        db (dict): Diccionario donde las claves son los nombres de los sujetos y los valores
                   son matrices de forma CxT_i (C = canales, T_i = tiempo).
    
    Returns:
        dict: Nuevo diccionario con los segmentos de cada sujeto.
    """
    segmentos_db = {}
    segmento_tamano = 512
    paso = int(segmento_tamano * 0.5)  # Traslape del 50%
    i = 0
    
    segmentos = []
    y = []
    sbjs = []
    
    for sujeto, senal in db.items():
        C, T = senal.shape
        
        # Crear segmentos con traslape
        for inicio in range(0, T - segmento_tamano + 1, paso):
            segmento = senal[:, inicio:inicio + segmento_tamano]
            segmentos.append(segmento)
            y.append(labels[i])
            sbjs.append(sujeto)

        i += 1
    return np.array(segmentos), np.array(y), sbjs
    

def get_segmented_data():
    """
    Se tiene que agregar en kaggle la base de datos
    """
    ruta_carpeta_TDAH = '/kaggle/input/ieee-tdah-control-database/ieee/ADHD_group'  
    ruta_carpeta_control = '/kaggle/input/ieee-tdah-control-database/ieee/Control_group'  
    
    # Nombre de cada sujeto
    sujetos_TDAH = [archivo[:-4] for archivo in os.listdir(ruta_carpeta_TDAH) if archivo.endswith('.mat')]
    sujetos_TDAH.pop()
    sujetos_control = [archivo[:-4] for archivo in os.listdir(ruta_carpeta_control) if archivo.endswith('.mat')]
    
    diagnostico = {}
    
    for sbj in sujetos_TDAH:
        diagnostico[sbj] = 1
    
    for sbj in sujetos_control:
        diagnostico[sbj] = 0
    
    # organizamos los datos de los sujetos con TDAH en un diccionario
    eeg_tdah = {}
    
    for i in range(len(sujetos_TDAH)):
        sbj = sujetos_TDAH[i]
        mat_file_path = ruta_carpeta_TDAH+'/'+sbj+'.mat'
        data = scipy.io.loadmat(mat_file_path)
        columna = list(data.keys())[-1]
        eeg_tdah[sbj] = data[columna].T
    
    # organizamos los datos de los sujetos de control en un diccionario
    eeg_control = {}
    
    for i in range(len(sujetos_control)):
        sbj = sujetos_control[i]
        mat_file_path = ruta_carpeta_control+'/'+sbj+'.mat'
        data = scipy.io.loadmat(mat_file_path)
        columna = list(data.keys())[-1]
        eeg_control[sbj] = data[columna].T
    
    db = eeg_control | eeg_tdah
    zeros = np.zeros(len(eeg_control))
    ones = np.ones(len(eeg_tdah))
    labels = np.hstack((zeros, ones))
    
    X, y, sbjs = segmentar_senales(db, labels)
    
    encoder = OneHotEncoder(sparse_output=False)
    
    # X = np.expand_dims(X, axis=-1)
    y = encoder.fit_transform(y.reshape(-1, 1))

    return X, y, sbjs
