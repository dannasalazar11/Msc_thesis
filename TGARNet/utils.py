import os
import numpy as np
import scipy.io
from sklearn.preprocessing import OneHotEncoder

def segmentar_senales(db, labels):
    """
    Divide las señales EEG en segmentos de 512 instantes con un traslape del 50%.

    Args:
        db (dict): Diccionario donde las claves son los nombres de los sujetos y los valores
                   son matrices de forma CxT_i (C = canales, T_i = tiempo).

    Returns:
        tuple:
            - segmentos: array de segmentos
            - y: array de etiquetas
            - sbjs: lista de sujetos por segmento
            - window_ids: lista con identificador de ventana por segmento
    """
    segmento_tamano = 512
    paso = int(segmento_tamano * 0.5)  # 50% overlap
    i = 0

    segmentos = []
    y = []
    sbjs = []
    window_ids = []

    for sujeto, senal in db.items():
        C, T = senal.shape
        window_count = 1

        for inicio in range(0, T - segmento_tamano + 1, paso):
            segmento = senal[:, inicio:inicio + segmento_tamano]
            segmentos.append(segmento)
            y.append(labels[i])
            sbjs.append(sujeto)
            window_ids.append(f"Window {window_count}")
            window_count += 1

        i += 1

    return np.array(segmentos), np.array(y), sbjs, window_ids


def get_segmented_data():
    """
    Se tiene que agregar en kaggle la base de datos
    """
    ruta_carpeta_TDAH = '/kaggle/input/datasets/daprosero/mi-tdah-dataset/MI_TDAH_Dataset/TDAH/ieee/ADHD_group'
    ruta_carpeta_control = '/kaggle/input/datasets/daprosero/mi-tdah-dataset/MI_TDAH_Dataset/TDAH/ieee/Control_group'

    sujetos_TDAH = [archivo[:-4] for archivo in os.listdir(ruta_carpeta_TDAH) if archivo.endswith('.mat')]
    sujetos_TDAH.pop()
    sujetos_control = [archivo[:-4] for archivo in os.listdir(ruta_carpeta_control) if archivo.endswith('.mat')]

    diagnostico = {}

    for sbj in sujetos_TDAH:
        diagnostico[sbj] = 1

    for sbj in sujetos_control:
        diagnostico[sbj] = 0

    eeg_tdah = {}
    for sbj in sujetos_TDAH:
        mat_file_path = ruta_carpeta_TDAH + '/' + sbj + '.mat'
        data = scipy.io.loadmat(mat_file_path)
        columna = list(data.keys())[-1]
        eeg_tdah[sbj] = data[columna].T

    eeg_control = {}
    for sbj in sujetos_control:
        mat_file_path = ruta_carpeta_control + '/' + sbj + '.mat'
        data = scipy.io.loadmat(mat_file_path)
        columna = list(data.keys())[-1]
        eeg_control[sbj] = data[columna].T

    db = eeg_control | eeg_tdah
    zeros = np.zeros(len(eeg_control))
    ones = np.ones(len(eeg_tdah))
    labels = np.hstack((zeros, ones))

    X, y, sbjs, window_ids = segmentar_senales(db, labels)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    return X, y, sbjs, window_ids
