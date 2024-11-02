from dataclasses import dataclass
import numpy as np

def sigmoid(self, z: float) -> float:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(self, z: float) -> float:
    """Derivative of the sigmoid function."""
    sig = self.sigmoid(z)
    return sig * (1 - sig)

def relu(z: float) -> float:
    """ReLU activation function."""
    return max(0, z)
        
def relu_derivative(z: float) -> float:
    """Derivative of the ReLU function."""
    return 1 if z > 0 else 0


@dataclass
class Neuron:
    w: np.array  # weights
    b: float = 0.0  # bias
    learning_rate: float = 0.01  # learning rate for gradient descent
    activation = relu  # activation function, float
    activation_derivative = relu_derivative  # derivative of activation function, float

    def forward(self, x: np.array) -> float:
        """Calculate the output of the neuron and pass through the activation function."""
        z = np.dot(self.w, x) + self.b  # weighted sum + bias
        return self.activation(z)  # activation function

    def backward(self, x: np.array, y_true: float, output: float) -> None:
        """Update the weights and bias based on the gradient of the loss."""
        # Calculate the error (loss derivative with respect to output)
        error = output - y_true
        
        # Calculate gradients
        z = np.dot(self.w, x) + self.b  # Recompute z to use in sigmoid derivative
        d_z = error * self.activation_derivative(z)  # chain rule
        
        # Update weights and bias
        self.w -= self.learning_rate * d_z * x  # gradient descent step for weights
        self.b -= self.learning_rate * d_z      # gradient descent step for bias
