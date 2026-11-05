"""Mục 2.4.3: mạng MLP bằng Keras (TensorFlow). TensorFlow trên Windows không dùng GPU nên chạy CPU."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

# Giới hạn số luồng để tiến trình Keras không tranh CPU với scikit-learn và NumPy chạy song song.
_N = int(os.environ.get("TL_TF_THREADS", "8"))
tf.config.threading.set_intra_op_parallelism_threads(_N)
tf.config.threading.set_inter_op_parallelism_threads(2)
import keras  # noqa: E402
from keras import layers  # noqa: E402

HEADS = {"binary": (1, "sigmoid", "binary_crossentropy"),
         "multi": (None, "softmax", "sparse_categorical_crossentropy"),
         "reg": (1, None, "mse")}


def build_keras_mlp(input_dim, task="binary", n_classes=3, hidden=(128, 64), dropout=0.2, lr=1e-3):
    units, act, loss = HEADS[task]
    model = keras.Sequential([keras.Input(shape=(input_dim,))])
    for h in hidden:
        model.add(layers.Dense(h, activation="relu"))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(units or n_classes, activation=act))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)
    return model


def fit(model, X, y, X_val, y_val, epochs=30, batch=256, patience=8):
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    h = model.fit(X, y, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch,
                  callbacks=[stop], verbose=0)
    return {"loss": h.history["loss"], "val_loss": h.history["val_loss"]}
