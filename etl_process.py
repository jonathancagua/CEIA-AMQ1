# etl_process.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # Puedes cambiar la métrica según sea necesario

# Función que realiza el particionado completo
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    """
    Realiza la división en tres conjuntos: entrenamiento, validación y prueba.
    
    Parámetros:
    - df: DataFrame de pandas con los datos a dividir.
    - rstate: Semilla aleatoria para la reproducibilidad.
    - shuffle: Booleano, si los datos deben ser barajados antes de dividir.
    - stratify: Columna para aplicar estratificación (puede ser None).
    
    Retorna:
    - Conjuntos de entrenamiento, validación y prueba.
    """
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    
    return train_set, val_set, test_set

# Función para eliminar las etiquetas (columnas objetivo)
def remove_labels(df, label_name):
    """
    Separa las características de la etiqueta objetivo.
    
    Parámetros:
    - df: DataFrame de pandas con los datos.
    - label_name: Nombre de la columna que contiene las etiquetas.
    
    Retorna:
    - X: Conjunto de características (features).
    - y: Conjunto de etiquetas (target).
    """
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return X, y
