"""Mục 3.3.2: Custom CNN với khối kiểu VGG bằng Keras - cùng kiến trúc với CustomCNN của PyTorch."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

_N = int(os.environ.get("TL_TF_THREADS", "8"))
tf.config.threading.set_intra_op_parallelism_threads(_N)
tf.config.threading.set_inter_op_parallelism_threads(2)
import keras  # noqa: E402
from keras import layers  # noqa: E402


def build_vgg_block(x, filters, num_convs=2, drop=0.2):
    """Khối VGG: các Conv 3x3 xếp chồng (mỗi conv kèm BatchNorm + ReLU) rồi MaxPool 2x2."""
    for _ in range(num_convs):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    return layers.Dropout(drop)(x)


def build_keras_custom_cnn(input_shape=(32, 32, 3), n_classes=10, widths=(32, 64, 128), drops=(0.2, 0.3, 0.4)):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for w, p in zip(widths, drops):
        x = build_vgg_block(x, w, 2, p)
    x = layers.GlobalAveragePooling2D()(x)          # GAP thay cho Flatten + Dense lớn
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def augmenter(name):
    """Tăng cường dữ liệu như bản PyTorch: dịch +-4 điểm (1/8 cạnh ảnh); EuroSAT thêm lật hai chiều."""
    aug = [layers.RandomTranslation(0.125, 0.125, fill_mode="reflect")]
    if name == "eurosat":
        aug.append(layers.RandomFlip("horizontal_and_vertical"))
    return keras.Sequential(aug)
