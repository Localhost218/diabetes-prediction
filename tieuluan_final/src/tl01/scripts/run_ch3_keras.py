"""Mục 3.4 - tiến trình CPU: Custom CNN bằng Keras trên SVHN và EuroSAT, để đối chuẩn với bản PyTorch.

Ảnh giữ ở uint8, chuẩn hóa trong tf.data để không phải giữ bản float32 cỡ gần 1 GB trong RAM.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl import data_vision as dv  # noqa: E402
from tl import metrics as M  # noqa: E402
from tl.cnn_keras import augmenter, build_keras_custom_cnn, tf  # noqa: E402
from tl.common import MODELS, file_kb, log, merge_json, seed_all  # noqa: E402

EPOCHS = {"svhn": 15, "eurosat": 30}
LOGN = "ch3_keras"


def run(name):
    seed_all()
    tf.random.set_seed(42)
    d = dv.load(name)
    xtr = d["x_train"].transpose(0, 2, 3, 1)           # NCHW -> NHWC cho Keras
    xte = d["x_test"].transpose(0, 2, 3, 1)
    # thống kê từng kênh bằng float32 để không cấp phát bản float64 cỡ 2 GB của cả tập ảnh
    chans = [xtr[..., c].astype(np.float32) / 255 for c in range(3)]
    mean = np.array([ch.mean() for ch in chans], np.float32)
    std = np.array([ch.std() for ch in chans], np.float32)
    del chans
    norm = lambda x: (tf.cast(x, tf.float32) / 255.0 - mean) / std  # noqa: E731
    aug = augmenter(name)
    tr = (tf.data.Dataset.from_tensor_slices((xtr, d["y_train"])).shuffle(20000, seed=42).batch(128)
          .map(lambda x, y: (aug(norm(x), training=True), y), num_parallel_calls=4).prefetch(4))
    te = tf.data.Dataset.from_tensor_slices((xte, d["y_test"])).batch(512).map(lambda x, y: (norm(x), y))
    model = build_keras_custom_cnn(xtr.shape[1:])
    steps = int(np.ceil(len(xtr) / 128)) * EPOCHS[name]
    model.optimizer.learning_rate = tf.keras.optimizers.schedules.CosineDecay(1e-3, steps)

    class Log(tf.keras.callbacks.Callback):
        def on_epoch_end(self, ep, logs=None):
            log(f"{name} epoch {ep + 1}: {({k: round(v, 4) for k, v in logs.items()})}", LOGN)

    t0 = time.perf_counter()
    h = model.fit(tr, validation_data=te, epochs=EPOCHS[name], verbose=0, callbacks=[Log()])
    train_s = time.perf_counter() - t0
    prob = model.predict(te, verbose=0)
    res = M.classification(d["y_test"], prob.argmax(1), binary=False)
    path = MODELS / "ch3" / f"{name}_custom_keras.keras"
    model.save(path)
    f = tf.function(lambda a: model(a, training=False))
    x1 = norm(xte[:1])
    for _ in range(10):
        f(x1)
    t0 = time.perf_counter()
    for _ in range(50):
        f(x1)
    lat = (time.perf_counter() - t0) / 50 * 1000
    hist = {"loss": h.history["loss"], "acc": h.history["accuracy"],
            "val_loss": h.history["val_loss"], "val_acc": h.history["val_accuracy"]}
    res.update(train_s=train_s, epochs=EPOCHS[name], hist=hist, params=model.count_params(),
               size_kb=file_kb(path), lat_cpu_ms=lat)
    merge_json(f"ch3_{name}", "custom_keras", res)
    log(f"{name}: acc={res['acc']:.4f} f1={res['f1']:.4f} train={train_s:.0f}s lat={lat:.2f}ms", LOGN)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["svhn", "eurosat"]:
        run(n)
