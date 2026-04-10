import os
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam

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


class DynamicSchedule(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, optimizer, eta_0=1e-3, alpha=10, beta=0.75, delta=10):
        super().__init__()
        self.total_epochs = total_epochs
        self.optimizer = optimizer
        self.eta_0 = eta_0
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.lambda_val = 0.0

    def get_eta(self, epoch):
        progress = epoch / self.total_epochs
        return self.eta_0 * (1 + self.alpha * progress) ** (-self.beta)

    def get_lambda(self, epoch):
        progress = epoch / self.total_epochs
        return 2 * (1 - np.exp(-self.delta * progress)) / (1 + np.exp(-self.delta * progress))

    def on_epoch_begin(self, epoch, logs=None):
        del logs
        new_lr = self.get_eta(epoch)
        self.lambda_val = self.get_lambda(epoch)

        if hasattr(self.optimizer.learning_rate, "assign"):
            self.optimizer.learning_rate.assign(new_lr)
        else:
            tf.keras.backend.set_value(self.optimizer.learning_rate, new_lr)

        if hasattr(self.model, "loss_weights") and isinstance(self.model.loss_weights, dict):
            self.model.loss_weights["out_activation"] = 1.0
            self.model.loss_weights["entropies_out"] = self.lambda_val

        print(f"[Epoch {epoch + 1}] LR={float(new_lr):.6f} | lambda={self.lambda_val:.3f}")


def SGKF(
    model_builder,
    X,
    y,
    sbjs,
    model_args,
    compile_args,
    folds,
    model_name="",
    delta=10,
    seed=42,
):
    del model_name
    all_fold_metrics = []

    for fold, (train_subjects, test_subjects) in enumerate(folds):
        print(f"\n{'-' * 60}")
        print(f"Fold {fold + 1}/{len(folds)}  |  Test subjects: {test_subjects}")
        print(f"{'-' * 60}")

        train_idx = [i for i, sbj in enumerate(sbjs) if sbj in train_subjects]
        test_idx = [i for i, sbj in enumerate(sbjs) if sbj in test_subjects]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        sbjs_test = [sbjs[i] for i in test_idx]

        tf.keras.backend.clear_session()

        current_seed = seed + fold
        np.random.seed(current_seed)
        random.seed(current_seed)
        tf.random.set_seed(current_seed)

        model = model_builder(**model_args)
        optimizer = Adam(learning_rate=1e-3)

        compile_args_local = deepcopy(compile_args)
        compile_args_local["optimizer"] = optimizer
        model.compile(**compile_args_local)

        num_kernels = model_args["num_kernels"]
        dynamic_schedule = DynamicSchedule(
            total_epochs=100,
            optimizer=optimizer,
            delta=delta,
        )

        model.fit(
            X_train,
            {
                "out_activation": y_train,
                "entropies_out": np.zeros((len(y_train), 4)),
                "kernel_weights_out": np.zeros((len(y_train), num_kernels)),
            },
            validation_data=(
                X_test,
                {
                    "out_activation": y_test,
                    "entropies_out": np.zeros((len(y_test), 4)),
                    "kernel_weights_out": np.zeros((len(y_test), num_kernels)),
                },
            ),
            epochs=100,
            batch_size=16,
            callbacks=[dynamic_schedule],
            verbose=0,
        )

        preds = model.predict(X_test, verbose=0)
        y_pred_probs = preds["out_activation"]
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        fold_metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_pred_probs[:, 1]),
        }
        all_fold_metrics.append(fold_metrics)
        print(f"Fold {fold + 1} Metrics: {fold_metrics}")

        subject_correct = defaultdict(list)
        for y_true_value, y_pred_value, subject in zip(y_true, y_pred, sbjs_test):
            subject_correct[subject].append(int(y_true_value == y_pred_value))

        subject_accuracies = {
            subject: np.mean(correct_predictions)
            for subject, correct_predictions in subject_correct.items()
        }
        print("Accuracy promedio por sujeto:")
        for subject in test_subjects:
            subject_accuracy = subject_accuracies.get(subject)
            if subject_accuracy is not None:
                print(f"  {subject}: {subject_accuracy:.4f}")

    print("\nIndividual Fold Accuracies:")
    accs_general = []
    for fold_index, fold_metric in enumerate(all_fold_metrics, start=1):
        print(f"  Fold {fold_index}: {fold_metric['accuracy']:.4f}")
        accs_general.append(fold_metric["accuracy"])

    return np.mean(accs_general)


