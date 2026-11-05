"""Mục 4.3.2: cùng cấu hình mạng hồi quy many-to-one bằng Keras (chạy CPU)."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

_N = int(os.environ.get("TL_TF_THREADS", "8"))
tf.config.threading.set_intra_op_parallelism_threads(_N)
tf.config.threading.set_inter_op_parallelism_threads(2)
import keras  # noqa: E402
from keras import layers  # noqa: E402

RNN_LAYERS = {"rnn": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU, "bilstm": layers.LSTM}


def build_keras_seq_model(cell="lstm", input_shape=(30, 9), hidden=64, num_layers=2, out="reg",
                          dropout=0.2, lr=1e-3):
    model = keras.Sequential([keras.Input(shape=input_shape)])
    for i in range(num_layers):
        rnn = RNN_LAYERS[cell](hidden, return_sequences=i < num_layers - 1)
        model.add(layers.Bidirectional(rnn) if cell == "bilstm" else rnn)
        if i < num_layers - 1:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(dropout))
    if out == "reg":
        model.add(layers.Dense(1))                       # giá đóng cửa phiên kế tiếp (thang chuẩn hóa)
        model.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0), loss="mse")
    else:
        model.add(layers.Dense(1, activation="sigmoid"))  # xác suất rời bỏ
        model.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0), loss="binary_crossentropy")
    return model


def fit(model, X, y, Xv, yv, epochs=80, batch=64, patience=12):
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    h = model.fit(X, y, validation_data=(Xv, yv), epochs=epochs, batch_size=batch, callbacks=[stop], verbose=0)
    return {"loss": h.history["loss"], "val_loss": h.history["val_loss"]}
