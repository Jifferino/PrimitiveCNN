import cv2
import os

import numpy as np
import matplotlib.pyplot as plt

# OpenCV ships with a face detector already so not training this part
#- need it as a tool for locating the face in the photo

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def extract_face(gray_image, img_size=64):
    faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        # fallback: no face detected, just use the whole image
        crop = gray_image
    else:
        # if multiple faces detected, take the largest one (w*h)
        x , y, w, h = max(faces, key= lambda f: f[2] * f[3])
        crop = gray_image[y:y+h, x:x+w] 

    return cv2.resize(crop, (img_size, img_size))

def load_face_dataset(root_dir, img_size=64):
    X = [] # will hold every processed face image
    y = [] # will hold the matching label (integer) for each image
    class_names = sorted(os.listdir(root_dir)) # e.g. ['luca', 'lexi', etc]

    for label, person_name in enumerate(class_names):
        person_folder = os.path.join(root_dir, person_name)
        for filename in os.listdir(person_folder):
            path = os.path.join(person_folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue # skip anything thats not a valid image file
            
            face = extract_face(img, img_size)
            X.append(face)
            y.append(label)

    X = np.array(X, dtype=np.float32) / 255.0 # normalize to 0-1
    X = X.reshape(-1, 1, img_size, img_size) # add the channel dimension: (N, 1, H, W)
    y = np.array(y, dtype=np.uint8)
    return X, y, class_names

def get_im2col_indices(x_shape, field_height, field_width, padding, stride):
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return k, i, j

def im2col(x, field_height, field_width, padding=0, stride=1):
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im(cols, x_shape, field_height, field_width, padding=0, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

class Layer_Conv2D:
    def __init__(self, n_filters, input_channels, filter_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.filter_size = filter_size
        # one small filter per output channel, same "small random start" idea as Layer_Dense
        self.weights = 0.1 * np.random.rand(n_filters, input_channels, filter_size, filter_size)
        self.biases = np.zeros((n_filters, 1))

    def forward(self, inputs):
        self.inputs = inputs
        N, C, H, W = inputs.shape
        F = self.filter_size
        n_filters = self.weights.shape[0]

        out_h = (H + 2 * self.padding - F) // self.stride + 1
        out_w = (W + 2 * self.padding - F) // self.stride + 1

        self.x_cols = im2col(inputs, F, F, self.padding, self.stride) # all patches, flattened
        w_col = self.weights.reshape(n_filters, -1) # all filters, flattened

        out = w_col @ self.x_cols + self.biases #big matrix multiplication instead of loop
        self.output = out.reshape(n_filters, out_h, out_w, N).transpose(3, 0, 1, 2)

    def backward(self, dvalues):
        n_filters, _, F, _ = self.weights.shape
        dvalues_reshaped = dvalues.transpose(1, 2, 3, 0).reshape(n_filters, -1)

        self.dbiases = np.sum(dvalues_reshaped, axis=1, keepdims=True)
        self.dweights = (dvalues_reshaped @ self.x_cols.T).reshape(self.weights.shape)

        w_col = self.weights.reshape(n_filters, -1)
        dx_cols = w_col.T @ dvalues_reshaped
        self.dinputs = col2im(dx_cols, self.inputs.shape, F, F, self.padding, self.stride)

class Layer_MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        self.inputs = inputs
        N, C, H, W = inputs.shape
        F, S = self.pool_size, self.stride
        out_h = (H - F) // S + 1
        out_w = (W - F) // S + 1

        self.out_shape = (out_h, out_w)

        # Treat each channel as if it were its own separate 1-channel image
        # so we can reuse im2col exactly as is
        x_reshaped = inputs.reshape(N*C, 1, H, W)
        cols = im2col(x_reshaped, F, F, padding=0, stride=S) #shape: (F*F, out_h*out_w*N*C)

        self.cols = cols
        self.argmax = np.argmax(cols, axis=0) # which row (which pixel) was biggest, per column
        out = cols[self.argmax, np.arange(cols.shape[1])] # pull out that max value for every column

        self.output = out.reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)

    def backward(self, dvalues):
        N, C, H, W = self.inputs.shape
        F, S = self.pool_size, self.stride
        out_h, out_w = self.out_shape

        dcols = np.zeros_like(self.cols)
        dvalues_flat = dvalues.transpose(2, 3, 0, 1).reshape(-1)
        dcols[self.argmax, np.arange(dcols.shape[1])] = dvalues_flat

        dx_reshaped = col2im(dcols, (N * C, 1, H, W), F, F, padding=0, stride=S)
        self.dinputs = dx_reshaped.reshape(N, C, H, W)

class Layer_Flatten:
    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.output = inputs.reshape(inputs.shape[0], -1)

    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.input_shape)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        '''
        Next 5 lines are for the softmax activation function for the output layer, 
        converting the values given to the neurons
        to an actually digestable list of values
        '''
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #Batch input compatible
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True) #Batch input compatible
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: #(list comes in a 1D array)
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: #for one hot encoded vectors (list comes in a 2D array)
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        #convert on hot to sparse labels if needed
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1 #This is the simplified derivative
        self.dinputs = self.dinputs / samples #normalize by this batch

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

def train_test_split(X, y, test_fraction=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    split = int(len(X) * (1 - test_fraction))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def build_model(n_classes, img_size=64):
    layers = {
        'conv1': Layer_Conv2D(n_filters=8, input_channels=1, filter_size=3, stride=1, padding=1),
        'relu1': Activation_ReLU(),
        'pool1': Layer_MaxPool2D(pool_size=2, stride=2),

        'conv2': Layer_Conv2D(n_filters=16, input_channels=8, filter_size=3, stride=1, padding=1),
        'relu2': Activation_ReLU(),
        'pool2': Layer_MaxPool2D(pool_size=2, stride=2),

        'flatten': Layer_Flatten(),
    }
    flattened_size = 16 * (img_size // 4) * (img_size // 4) #pool_size and stride both halve the size of the image so the new image is 1/4 the size
    layers['dense1'] = Layer_Dense(flattened_size, 128)
    layers['relu3'] = Activation_ReLU()
    layers['dense2'] = Layer_Dense(128, n_classes)
    layers['loss_activation'] = Activation_Softmax_Loss_CategoricalCrossentropy()
    return layers

def forward_pass(layers, X_batch, y_batch):
    layers['conv1'].forward(X_batch)
    layers['relu1'].forward(layers['conv1'].output)
    layers['pool1'].forward(layers['relu1'].output)

    layers['conv2'].forward(layers['pool1'].output)
    layers['relu2'].forward(layers['conv2'].output)
    layers['pool2'].forward(layers['relu2'].output)

    layers['flatten'].forward(layers['pool2'].output)

    layers['dense1'].forward(layers['flatten'].output)
    layers['relu3'].forward(layers['dense1'].output)
    layers['dense2'].forward(layers['relu3'].output)

    loss = layers['loss_activation'].forward(layers['dense2'].output, y_batch)
    return loss


def backward_pass(layers, y_batch):
    layers['loss_activation'].backward(layers['loss_activation'].output, y_batch)
    layers['dense2'].backward(layers['loss_activation'].dinputs)
    layers['relu3'].backward(layers['dense2'].dinputs)
    layers['dense1'].backward(layers['relu3'].dinputs)

    layers['flatten'].backward(layers['dense1'].dinputs)

    layers['pool2'].backward(layers['flatten'].dinputs)
    layers['relu2'].backward(layers['pool2'].dinputs)
    layers['conv2'].backward(layers['relu2'].dinputs)

    layers['pool1'].backward(layers['conv2'].dinputs)
    layers['relu1'].backward(layers['pool1'].dinputs)
    layers['conv1'].backward(layers['relu1'].dinputs)

def train(root_dir, img_size=64, epochs=30, batch_size=16, learning_rate=1.0):
    X, y, class_names = load_face_dataset(root_dir, img_size=img_size)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_fraction=0.2)

    layers = build_model(n_classes=len(class_names), img_size=img_size)
    optimizer = Optimizer_SGD(learning_rate=learning_rate)

    n_samples = len(X_train)

    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)     # reshuffle order every epoch
        X_train, y_train = X_train[perm], y_train[perm]

        epoch_losses = []
        for start in range(0, n_samples, batch_size):
            X_batch = X_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]

            loss = forward_pass(layers, X_batch, y_batch)
            epoch_losses.append(loss)

            backward_pass(layers, y_batch)

            for name in ('conv1', 'conv2', 'dense1', 'dense2'):
                optimizer.update_params(layers[name])

        test_loss = forward_pass(layers, X_test, y_test)
        preds = np.argmax(layers['loss_activation'].output, axis=1)
        test_acc = np.mean(preds == y_test)

        print(f"epoch {epoch+1:3d} | train loss {np.mean(epoch_losses):.3f} "
              f"| test loss {test_loss:.3f} | test acc {test_acc:.3f}")

    return layers, class_names