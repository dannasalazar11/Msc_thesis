import numpy as np
import random
from collections import defaultdict
from copy import deepcopy

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    cohen_kappa_score,
    roc_auc_score
)


def train_L24O_cv(model_builder, X, y, sbjs, model_args, compile_args, folds, model_name='', delta=10, seed=42):
    all_fold_metrics = []
    total_histories = []
    models = {}

    # ------------------------------------------
    # 1. Construir un diccionario sujeto → clase
    # ------------------------------------------
    # Convertimos one-hot a clase entera
    y_classes = np.argmax(y, axis=1)

    subject_label = {}
    for sbj in set(sbjs):
        # tomo el primer segmento del sujeto (todos son iguales)
        idx = sbjs.index(sbj)
        subject_label[sbj] = y_classes[idx]   # 0 = CTRL, 1 = ADHD

    # ==========================================
    # INICIO CV
    # ==========================================
    for fold, (train_subjects, test_subjects) in enumerate(folds):

        print(f"\n{'-'*60}")
        print(f"Fold {fold+1}/{len(folds)}  |  Test subjects: {test_subjects}")
        print(f"{'-'*60}")

        # --- Índices originales ---
        train_idx = [i for i, sbj in enumerate(sbjs) if sbj in train_subjects]
        test_idx  = [i for i, sbj in enumerate(sbjs) if sbj in test_subjects]

        # ==========================================
        # 2. SELECCIÓN ESTRATIFICADA DE VALIDACIÓN
        # ==========================================

        # Separar los sujetos del fold según clase
        train_ADHD = [s for s in train_subjects if subject_label[s] == 1]
        train_CTRL = [s for s in train_subjects if subject_label[s] == 0]

        # Semilla reproducible por fold
        rng = np.random.default_rng(seed + fold)

        # Seleccionar 8 por clase
        val_ADHD = rng.choice(train_ADHD, size=8, replace=False)
        val_CTRL = rng.choice(train_CTRL, size=8, replace=False)

        val_subjects = set(val_ADHD.tolist() + val_CTRL.tolist())

        # índices de validación
        val_idx = [i for i, sbj in enumerate(sbjs) if sbj in val_subjects]

        # entrenamiento final = train_idx - val_idx
        train_idx_final = [i for i in train_idx if sbjs[i] not in val_subjects]

        # Datos finales
        X_train_final, y_train_final = X[train_idx_final], y[train_idx_final]
        X_val,         y_val         = X[val_idx],        y[val_idx]
        X_test,        y_test        = X[test_idx],       y[test_idx]
        sbjs_test = [sbjs[i] for i in test_idx]

        # ==========================================
        # 3. MODELO
        # ==========================================
        tf.keras.backend.clear_session()
        np.random.seed(seed + fold)
        random.seed(seed + fold)
        tf.random.set_seed(seed + fold)
        
        # --- Callbacks ---
        # CALLBACKS
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=30, min_delta=1e-4, restore_best_weights=True, verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=30, min_lr=1e-6, verbose=1
        )

        # --- Build and Compile Model for each fold ---
        tf.keras.backend.clear_session() #<-- Clear session to prevent any state leakage
        
        # Re-set seeds for each fold for perfect reproducibility of weight initialization
        np.random.seed(seed + fold)
        random.seed(seed + fold)
        tf.random.set_seed(seed + fold)

        model = model_builder(**model_args)
        # Use a deepcopy to prevent the optimizer state from carrying over
        compile_args_local = deepcopy(compile_args)
        if callable(compile_args_local["optimizer"]):
            compile_args_local["optimizer"] = compile_args_local["optimizer"]()  # <-- aquí se reinicia
        model.compile(**compile_args_local)

        # --- Train the Model ---
        model.fit(
            X_train_final, y_train_final,
            epochs=100,  #<-- Increased epochs to give LR scheduler more time to work
            validation_data=(X_val, y_val),
            verbose=0, #<-- Verbose=2 gives one line per epoch, cleaner log
            batch_size=16,
            callbacks=[early_stopping, 
                       reduce_lr]
        )

        # --- Predictions and Evaluation ---
        y_pred_probs = model.predict(X_test)
        print(y_pred_probs.shape)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Overall fold metrics
        fold_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'kappa': cohen_kappa_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_probs[:, 1]) # Use probabilities for AUC
        }
        print(f"\nFold {fold+1} Metrics: {fold_metrics}")
        all_fold_metrics.append(fold_metrics)
        models[fold] = model

        # Accuracy por sujeto de test
        subject_correct = defaultdict(list)
        for yt, yp, sbj in zip(y_true, y_pred, sbjs_test):
            subject_correct[sbj].append(int(yt == yp))

        subject_accuracies = {
            sbj: np.mean(subject_correct[sbj]) for sbj in subject_correct
        }

        print("Average accuracy per test subject:")
        for sbj in test_subjects:
            acc_sbj = subject_accuracies.get(sbj, None)
            if acc_sbj is not None:
                print(f"  {sbj}: {acc_sbj:.4f}")
                
        
    # --- Final Comprehensive Report ---
    print("\n" + "="*50)
    print("Cross-Validation Final Results")
    print("="*50)
    
    # Calculate mean and std dev for each metric
    mean_metrics = {}
    for key in all_fold_metrics[0].keys():
        values = [f[key] for f in all_fold_metrics]
        mean_metrics[f'mean_{key}'] = np.mean(values)
        mean_metrics[f'std_{key}'] = np.std(values)

    print("Individual Fold Accuracies:")
    for i, f in enumerate(all_fold_metrics):
        print(f"  Fold {i+1}: {f['accuracy']:.4f}")
        
    print("\nAverage Performance across all folds:")
    for key, value in mean_metrics.items():
        print(f"  {key}: {value:.4f}")
        
    return all_fold_metrics
