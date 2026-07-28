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
