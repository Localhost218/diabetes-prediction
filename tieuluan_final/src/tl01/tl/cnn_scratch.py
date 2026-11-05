"""Mục 3.3.1: tích chập 2D nhiều kênh và Max Pooling viết tay bằng NumPy.

Hai cách hiện thực cùng một công thức (3.1):
    conv2d_loop   : bốn vòng lặp tường minh, bám sát ký hiệu toán học, dùng để giảng giải
    conv2d_im2col : trải các cửa sổ thành ma trận rồi nhân một lần - cách các thư viện thật làm
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv2d_loop(X, W, b, stride=1, padding=0):
    """X: [N, Cin, H, W], W: [Cout, Cin, kh, kw], b: [Cout] -> [N, Cout, Hout, Wout]."""
    N, C_in, H, W_ = X.shape
    C_out, _, kh, kw = W.shape
    Xp = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding))) if padding else X
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W_ + 2 * padding - kw) // stride + 1
    out = np.zeros((N, C_out, H_out, W_out), dtype=X.dtype)
    for i in range(H_out):
        for j in range(W_out):
            patch = Xp[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]   # [N, Cin, kh, kw]
            for c in range(C_out):
                out[:, c, i, j] = np.sum(patch * W[c], axis=(1, 2, 3)) + b[c]
    return out


def conv2d_im2col(X, W, b, stride=1, padding=0):
    Xp = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding))) if padding else X
    C_out, C_in, kh, kw = W.shape
    win = sliding_window_view(Xp, (kh, kw), axis=(2, 3))[:, :, ::stride, ::stride]    # [N, Cin, Ho, Wo, kh, kw]
    out = np.einsum("nchwij,ocij->nohw", win, W, optimize=True)
    return out + b[None, :, None, None]


def maxpool2d(X, size=2, stride=2):
    win = sliding_window_view(X, (size, size), axis=(2, 3))[:, :, ::stride, ::stride]
    return win.max(axis=(-1, -2))


def batchnorm_eval(X, gamma, beta, mean, var, eps=1e-5):
    return gamma[None, :, None, None] * (X - mean[None, :, None, None]) / np.sqrt(var[None, :, None, None] + eps) \
        + beta[None, :, None, None]


def relu(X):
    return np.maximum(X, 0)
