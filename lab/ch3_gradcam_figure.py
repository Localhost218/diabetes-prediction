"""
Vẽ lại hình Grad-CAM của Chương 3 từ mô hình đã lưu (data/models/ch3_resnet18.pt), không cần huấn luyện lại.

    .venv/bin/python lab/ch3_gradcam_figure.py
"""

import numpy as np

from common import DATA, plt, save_fig
from src.models.cnn_vision import RetinaGrader, preprocess

from PIL import Image


def main():
    grader = RetinaGrader()
    data = np.load(DATA / "retina" / "retinamnist_128.npz", allow_pickle=False)
    y = data["test_labels"].ravel()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.9))
    for c in range(5):
        i = int(np.where(y == c)[0][1])
        img = Image.fromarray(data["test_images"][i])
        x = preprocess(img)
        pred = int(grader.model(x).argmax(1))
        cam = grader.grad_cam(x.clone().requires_grad_(True), pred)
        axes[0, c].imshow(img)
        axes[0, c].set_title(f"nhãn thật: mức {c}", fontsize=9)
        axes[1, c].imshow(img)
        axes[1, c].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, c].set_title(f"dự đoán: mức {pred}", fontsize=9)
        for a in axes[:, c]:
            a.axis("off")
    fig.suptitle("Grad-CAM trên layer4 của ResNet-18: vùng ảnh quyết định dự đoán", y=1.0)
    fig.tight_layout(h_pad=1.6)
    save_fig(fig, "ch3_gradcam")


if __name__ == "__main__":
    main()
