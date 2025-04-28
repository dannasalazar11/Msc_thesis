import numpy as np
import os
import scipy.io
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping

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
