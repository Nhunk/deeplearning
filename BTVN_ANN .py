import numpy as np
import matplotlib.pyplot as plt
#|%%--%%| <iyDsjF4oUA|zv8CGLO1i8>

# Khoi tao 1 layer ANN

class ANN:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # khoi tao weights va bias
        self.weights_input_hidden = np.random.rand(n_inputs + 1, n_hidden) # 2x5
        self.weights_hidden_output = np.random.rand(n_hidden + 1, n_outputs) # 5x2
        
        # khoi tao activation function

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    # feedforward
    def feedforward(self, inputs):
        inputs = np.hstack((inputs, 1)) # them bias

        self.hidden_layer_input = np.einsum('ij, i -> j', self.weights_input_hidden, inputs) # 3x4, 1x3 -> 1x4
        self.hidden_layer_input_tanh = self.tanh(self.hidden_layer_input) # 1x4
        self.hidden_layer_input_tanh = np.hstack((self.hidden_layer_input_tanh, 1)) # them bias 1x5

        self.output_layer_input = np.einsum('ij, i -> j', self.weights_hidden_output, self.hidden_layer_input_tanh) # 5x2, 1x5 -> 1x2
        #self.output = self.sigmoid(self.output_layer_input)
        self.output = self.output_layer_input # neu sigma(x) = x
        return self.output


ann = ANN(2, 4, 2)
#    print(ann.weights_input_hidden)
#    print(ann.weights_hidden_output)
input = np.array([1, 0])
output = ann.feedforward(input)
print(output)

#|%%--%%| <zv8CGLO1i8|UxIU617L5N>

a = np.array([[1, 1, 2], [1, 2, 0]])
b = np.array([1, 1, 1])
print(a)
print(b)
d = np.dot(a, b.T) # 2x3 X 3x1
c = np.einsum('ij, j -> i', a, b) 
print(c)
print(d)


#|%%--%%| <UxIU617L5N|nWRG7xfYwB>

# Train
ann = ANN(2, 4, 2)

def train(self, inputs, targets, epochs, learning_rate):
    for epoch in range(epochs):
        for i in range(len(inputs)):
            input_i = inputs[i]
            target_i = targets[i]
            
            output = ann.feedforward(input_i)

            # tinh toan sai so
            error = target_i - output

            # backpropagation
            d_output = error * self.sigmoid_derivative(self.output)
            error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self.tanh_derivative(self.hidden_layer_input_tanh)

            # cap nhat weights
            self.weights_hidden_output += self.hidden_layer_input_tanh.T.dot(d_output) * learning_rate
            self.weights_input_hidden += np.hstack((input, 1)).T.dot(d_hidden_layer) * learning_rate
        
