"""Chương 4 — mô hình Keras tương đương PyTorch (TensorFlow trên Windows chạy CPU)."""
import time

import numpy as np


def build_keras_model(n_features, arch="lstm", window=30, hidden=64, dropout=0.2, lr=1e-3):
    """SimpleRNN / LSTM / GRU / Bidirectional(LSTM) 64 đơn vị → Dropout → Dense(1), cùng cấu hình bản PyTorch."""
    import keras
    from keras import layers
    rec = {"rnn": lambda: layers.SimpleRNN(hidden, activation="tanh"),
           "lstm": lambda: layers.LSTM(hidden),
           "gru": lambda: layers.GRU(hidden),
           "bilstm": lambda: layers.Bidirectional(layers.LSTM(hidden))}[arch]
    model = keras.Sequential([
        keras.Input(shape=(window, n_features)),
        rec(),
        layers.Dropout(dropout),
        layers.Dense(1),
    ], name=f"keras_{arch}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0), loss="mse")
    return model


def fit_keras(arch, data, epochs=150, patience=15, batch=64, hidden=64, seed=42):
    """Huấn luyện với EarlyStopping(restore_best_weights) và ReduceLROnPlateau; trả (model, history, giây)."""
    import keras
    keras.utils.set_random_seed(seed)
    Xtr, ytr = data.part("train")
    Xva, yva = data.part("val")
    model = build_keras_model(Xtr.shape[-1], arch, Xtr.shape[1], hidden)
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6)]
    t0 = time.perf_counter()
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch, shuffle=True,
                  callbacks=cbs, verbose=0)
    secs = time.perf_counter() - t0
    hist = {"loss": [float(v) for v in h.history["loss"]], "val_loss": [float(v) for v in h.history["val_loss"]]}
    print(f"  keras {arch}: {len(hist['loss'])} epoch, best val {min(hist['val_loss']):.4f}, {secs:.1f}s", flush=True)
    return model, hist, secs


def predict_keras(model, X):
    return np.asarray(model.predict(X, batch_size=1024, verbose=0)).reshape(-1)
