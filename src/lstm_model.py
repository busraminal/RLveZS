import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ============================================================
# LSTM MODELİ
# Amaç: lead_time dizisini öğrenip geleceği tahmin etmek
# ============================================================

def create_lstm():
    """
    10 adımlık geçmişten 1 adımlık tahmin yapan LSTM modeli.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(10, 1)),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_sequences(series, window=10):
    """
    Zaman serisini LSTM’in anlayacağı sequence yapısına çevirir.
    """
    X, y = [], []

    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])

    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    return X, y
