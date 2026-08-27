# PrimitiveCNN

A convolutional neural network built entirely from scratch with NumPy- no PyTorch, TensorFlow, or Keras. It performs face *recognition* (identifying **who** is in an image, not just detecting that a face is present), with the end goal of running on a Raspberry Pi as the vision system for a robot.

OpenCV is used only as a tool to locate and crop faces (via a pre-trained Haar cascade); every layer of the actual network- convolution, pooling, dense layers, activations, loss, backprop, and the optimizer is hand-implemented.

## Why "primitive"

Every forward and backward pass is written manually:

- `im2col` / `col2im` for fast, vectorized convolution and pooling (no explicit sliding-window loops)
- Manual backprop through every layer, derived and coded by hand
- A from-scratch SGD-with-momentum optimizer

The goal is to understand, and be able to explain, exactly what happens at every step of a CNN, and to end up with something small and dependency-light enough to run inference on constrained hardware like a Raspberry Pi.

## Architecture

```
Input (1 x 64 x 64 grayscale face)
  -> Conv2D(8 filters, 3x3, pad 1)  -> ReLU -> MaxPool(2x2)
  -> Conv2D(16 filters, 3x3, pad 1) -> ReLU -> MaxPool(2x2)
  -> Flatten
  -> Dense(-> 128) -> ReLU
  -> Dense(-> n_classes)
  -> Softmax + Categorical Cross-Entropy
```

- **Weight init:** He initialization (`sqrt(2 / fan_in)`) to keep ReLU units from dying early in training.
- **Optimizer:** SGD with momentum.
- **Data augmentation:** each cropped face is horizontally flipped and added as an extra training sample, roughly doubling the effective dataset size.

## Pipeline

1. **Face detection (not learned):** `cv2.CascadeClassifier` (Haar cascade) finds the largest face in each image and crops it. If no face is found, the whole image is used as a fallback.
2. **Preprocessing:** crop is resized to 64x64, converted to grayscale, and normalized to `[0, 1]`.
3. **Training:** the CNN above is trained from scratch to classify *which person* each cropped face belongs to.

## Dataset layout

Images live under `faces/`, one subfolder per person (the folder name becomes the class label):

```
faces/
  luca/
    img001.jpg
    img002.jpg
    ...
  example/
    img001.jpg
    ...
```


## Usage

```
python pl.py
```

This loads `faces/`, splits into train/test (80/20), and trains for 60 epochs, printing per-epoch train/test loss and accuracy. Model layers and class names are returned by `train()` but are **not currently saved to disk** — see Roadmap.

## Roadmap: Raspberry Pi / robot integration

The network is deliberately dependency-light (just NumPy for math), which makes it a good fit for a Pi, but a few pieces are still needed before it can run on-device:

- [ ] **Save/load trained weights** (e.g. via `np.savez`) instead of only holding them in memory after `train()`
- [ ] **Separate inference script** that loads saved weights and runs a single forward pass on a new image, without needing the training/dataset code
- [ ] **Live camera loop** using `picamera2` (or OpenCV's `VideoCapture`) to grab frames, run the existing `extract_face` + forward pass on each one, and act on the prediction
- [ ] **Inference speed check** on Pi hardware — the `im2col`-based convolution is vectorized but still pure NumPy; may need to shrink the network (fewer filters, smaller input) if it's too slow for real-time use
- [ ] Consider quantizing/pruning weights if memory or speed becomes a bottleneck on-device

## Status

Actively evolving learning project — architecture and training loop work end-to-end on a laptop/desktop; Raspberry Pi deployment is the current focus.
