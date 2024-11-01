from dataclasses import dataclass
import numpy as np
@dataclass
class Neuron:
    w: np.array #weights
    # x: np.array 
    b: float = 0.0  # bias 

    def forward(self, x: np.array) -> float: #caluclate output of neuron and pass through activation function
        z = np.dot(self.w, x) + self.b  # weighted sum + bias
        return self.sigmoid(z)  # activation function

    def sigmoid(self, z: float) -> float:
        return 1 / (1 + np.exp(-z))

    #need def for updating weights gradient descnet
