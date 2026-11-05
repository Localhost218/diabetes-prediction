"""Chương 3 — Basic CNN và Improved CNN bằng TensorFlow/Keras (CPU: TensorFlow ≥ 2.11 trên Windows không hỗ trợ GPU)."""
import time

import numpy as np


def build_basic(in_shape, n_classes):
    """Basic CNN Keras tương đương bản NumPy và PyTorch: Conv32 → Pool → Conv64 → Pool → Dense128 → Dense(K)."""
    from keras import Sequential, layers
    return Sequential([
        layers.Input(in_shape),
        layers.Conv2D(32, 3, padding="same", activation="relu"), layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"), layers.MaxPooling2D(2),
        layers.Flatten(), layers.Dense(128, activation="relu"), layers.Dense(n_classes, activation="softmax"),
    ], name="basic_cnn")


def augmentation(cfg):
    """Các tầng tăng cường dữ liệu của Keras — chỉ hoạt động khi training=True, tự tắt lúc suy luận."""
    from keras import Sequential, layers
    aug = []
    if cfg["flip"]:
        aug.append(layers.RandomFlip("horizontal"))
    if cfg["rot"]:
        aug.append(layers.RandomRotation(cfg["rot"] / 360.0))
    aug.append(layers.RandomTranslation(cfg["shift"], cfg["shift"]))
    if cfg["zoom"]:
        aug.append(layers.RandomZoom(cfg["zoom"]))
    if cfg["color"]:
        aug += [layers.RandomContrast(cfg["color"]), layers.RandomBrightness(cfg["color"], value_range=(0, 1))]
    return Sequential(aug, name="augment")


def res_block(x, filters):
    """Khối dư Keras: Conv-BN-ReLU-Conv-BN + lối tắt (Conv 1×1 + BN khi đổi số kênh) → ReLU."""
    from keras import layers
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    y = layers.ReLU()(layers.BatchNormalization()(y))
    y = layers.BatchNormalization()(layers.Conv2D(filters, 3, padding="same", use_bias=False)(y))
    if x.shape[-1] != filters:
        x = layers.BatchNormalization()(layers.Conv2D(filters, 1, use_bias=False)(x))
    return layers.ReLU()(layers.Add()([x, y]))


def build_improved(in_shape, n_classes, cfg, mean, std):
    """Improved CNN Keras: tăng cường → chuẩn hoá theo kênh → stem → [ResBlock → MaxPool → Dropout] → GAP → Dense."""
    import keras
    from keras import layers
    inp = layers.Input(in_shape)
    x = augmentation(cfg)(inp)
    x = layers.Normalization(mean=mean, variance=std ** 2)(x)
    x = layers.Conv2D(cfg["widths"][0], 3, padding="same", use_bias=False)(x)
    x = layers.ReLU()(layers.BatchNormalization()(x))
    for i, w in enumerate(cfg["widths"]):
        x = res_block(x, w)
        x = layers.Dropout(0.1 + 0.1 * i)(layers.MaxPooling2D(2)(x))
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(layers.Dense(256, activation="relu")(x))
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="improved_cnn")


class BestWeights:
    """Callback giữ trọng số tốt nhất trên validation và khôi phục khi kết thúc (giống bản PyTorch).

    "Tốt nhất" là val_accuracy cao nhất, hòa thì lấy val_loss thấp hơn (cần khi val_accuracy bão hòa ở 1,0).
    patience: dừng sớm sau chừng ấy epoch không cải thiện; budget_s: dừng khi epoch kế tiếp sẽ vượt ngân sách thời gian.
    ckpt: thư mục điểm lưu. Sau mỗi epoch ghi mô hình đầy đủ (kèm trạng thái optimizer, tức vị trí trên lịch học), trọng
    số tốt nhất và trạng thái callback; chạy lại sau sự cố sẽ tiếp tục từ epoch kế tiếp thay vì từ đầu.
    """

    def __new__(cls, patience=None, budget_s=None, ckpt=None, max_epochs=None):
        import json
        import os
        import shutil

        import keras

        def atomic(save, path):
            tmp = path.with_name("tmp." + path.name)
            save(tmp)
            os.replace(tmp, path)

        class _CB(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best, self.best_key, self.best_ep, self.times, self._t = None, (-1.0, 0.0), 0, [], 0.0
                self.wait, self.stop_reason, self.done = 0, "max_epochs", False
                self.hist = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
                state = ckpt / "state.json" if ckpt is not None else None
                if state is not None and state.exists():
                    s = json.loads(state.read_text(encoding="utf-8"))
                    if s["max_epochs"] != max_epochs:          # điểm lưu của cấu hình khác: bỏ, chạy từ đầu
                        shutil.rmtree(ckpt)
                    else:
                        self.best_key, self.best_ep, self.times = tuple(s["best_key"]), s["best_ep"], s["times"]
                        self.wait, self.hist, self.stop_reason, self.done = s["wait"], s["hist"], s["stop"], s["done"]
                if ckpt is not None:
                    ckpt.mkdir(parents=True, exist_ok=True)
                self.start = len(self.times)

            def on_epoch_begin(self, epoch, logs=None):
                self._t = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                self.times.append(time.perf_counter() - self._t)
                for k, src in (("loss", "loss"), ("acc", "accuracy"), ("val_loss", "val_loss"),
                               ("val_acc", "val_accuracy")):
                    self.hist[k].append(float(logs[src]))
                key = (logs["val_accuracy"], -logs["val_loss"])
                if key > self.best_key:
                    self.best_key, self.best_ep, self.wait = key, epoch + 1, 0
                    self.best = self.model.get_weights()
                    if ckpt is not None:
                        atomic(self.model.save_weights, ckpt / "best.weights.h5")
                else:
                    self.wait += 1
                if epoch + 1 == max_epochs:     # đã chạy đủ: lý do dừng giữ "max_epochs"
                    self.done = True
                elif patience is not None and self.wait >= patience:
                    self.model.stop_training, self.stop_reason, self.done = True, "early_stop", True
                elif budget_s is not None and sum(self.times) + self.times[-1] > budget_s:
                    self.model.stop_training, self.stop_reason, self.done = True, "time_budget", True
                if ckpt is not None:
                    atomic(self.model.save, ckpt / "last.keras")
                    s = {"max_epochs": max_epochs, "best_key": list(self.best_key), "best_ep": self.best_ep,
                         "times": self.times, "wait": self.wait, "hist": self.hist, "stop": self.stop_reason,
                         "done": self.done}
                    atomic(lambda p: p.write_text(json.dumps(s), encoding="utf-8"), ckpt / "state.json")

            def on_train_end(self, logs=None):
                if self.best is not None:
                    self.model.set_weights(self.best)
                else:                               # tiếp tục sau sự cố mà chưa có epoch nào tốt hơn: đọc từ điểm lưu
                    self.model.load_weights(ckpt / "best.weights.h5")

        return _CB()


def train_keras(model, xtr, ytr, xva, yva, cfg, epochs, improved, patience=None, budget_s=None, ckpt=None):
    """Basic: Adam 1e-3, sparse CE. Improved: AdamW + cosine decay có warmup, CE với label smoothing 0,1 (nhãn one-hot).

    epochs là số epoch tối đa (lịch cosine trải trên đúng chừng ấy epoch); patience/budget_s cho phép dừng sớm; ckpt cho
    phép tiếp tục sau sự cố. Trả (lịch sử, epoch tốt nhất, lý do dừng, mô hình đã nạp trọng số tốt nhất).
    """
    import keras
    k = int(ytr.max()) + 1
    steps = int(np.ceil(len(xtr) / cfg["batch"])) * epochs
    if improved:
        lr = keras.optimizers.schedules.CosineDecay(1e-4, steps - steps // 4, warmup_target=3e-3,
                                                    warmup_steps=steps // 4)
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        model.compile(keras.optimizers.AdamW(lr, weight_decay=5e-4), loss, metrics=["accuracy"])
        y_fit, y_val = keras.utils.to_categorical(ytr, k), keras.utils.to_categorical(yva, k)
    else:
        model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
        y_fit, y_val = ytr, yva
    cb = BestWeights(patience, budget_s, ckpt, epochs)
    if cb.start:    # tiếp tục: mô hình đầy đủ gồm trọng số, optimizer và số bước đã đi trên lịch học
        model = keras.models.load_model(ckpt / "last.keras")
    if cb.done:
        cb.set_model(model)
        cb.on_train_end()
    else:
        model.fit(xtr, y_fit, validation_data=(xva, y_val), epochs=epochs, initial_epoch=cb.start,
                  batch_size=cfg["batch"], callbacks=[cb], verbose=2, shuffle=True)
    return {**cb.hist, "time": cb.times}, cb.best_ep, cb.stop_reason, model
