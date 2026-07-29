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

class Layer_Conv2D:
    def __init__(self, n_filters, input_channels, filter_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.filter_size = filter_size
        # one small filter per output channel, same "small random start" idea as Layer_Dense
        self.weights = 0.1 * np.random.rand(n_filters, input_channels, filter_size, filter_size)
        self.biases = np.zers((n_filters, 1))

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

        # Treat each channel as if it were its own separate 1-channel image
        # so we can reuse im2col exactly as is
        x_reshaped = inputs.reshape(N*C, 1, H, W)
        cols = im2col(x_reshaped, F, F, padding=0, stride=S) #shape: (F*F, out_h*out_w*N*C)

        self.cols = cols
        self.argmax = np.argmax(cols, axis=0) # which row (which pixel) was biggest, per column
        out = cols[self.argmax, np.arange(cols.shape[1])] # pull out that max value for every column

        self.output = out.reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)


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
        if len(y_true.shape == 2):
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

X,y = create_data(points=100, classes=3)

dense1 = Layer_Dense(2,64) #hidden layer
activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 3) #output layer
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_SGD(learning_rate=1.0)

for epoch in range(10001):
    #forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    #accuracy - for monitoring
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y_compare = np.argmax(y, axis=1)
    else:
        y_compare = y
    accuracy = np.mean(predictions == y_compare)

    if epoch % 1000 == 0:
        print(f'epoch: {epoch}, loss: {loss:.3f}, accuracy: {accuracy:.3f}')

    #backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
