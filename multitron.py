import numpy as np

class Multitron:
    def __init__(self, inputCount: int, neuronCount: int):
        self.neuronCount = neuronCount
        self.inputCount = inputCount
        self._weights = np.random.uniform(-1, 1, (neuronCount, inputCount + 1))

    def compute(self, input: np.ndarray) -> np.ndarray:
        input2d = input if len(input.shape) > 1 else input.reshape(1, -1)
        biasedInput = np.hstack((np.ones((input2d.shape[0], 1)), input2d))
        return biasedInput @ self._weights.T
    
    def train(self, inputs: np.ndarray, expecteds: np.ndarray, learnRate: float, gradThres: float):
        dataCount = inputs.shape[0]

        while True:
            meanGradLW = np.zeros(self._weights.shape)

            for dataIdx in range(dataCount):
                gradLM = (self.compute(inputs[dataIdx]) - expecteds[dataIdx]).reshape(-1, 1)
                biasedInput = np.append(1, inputs[dataIdx]).reshape(-1, 1)
                meanGradLW += gradLM @ biasedInput.T

            if np.linalg.norm(meanGradLW) <= gradThres:
                break

            self._weights -= learnRate * meanGradLW

if __name__ == "__main__":
    # Tiny dataset: 2 inputs, 2 neurons
    inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Expected outputs: linear function for each neuron
    # neuron 0: y = 2*x0 + 3*x1
    # neuron 1: y = -x0 + 0.5*x1
    expecteds = np.array([
        [0, 0],
        [3, 0.5],
        [2, -1],
        [5, -0.5]
    ])

    # Create Multitron
    m = Multitron(inputCount=2, neuronCount=2)

    # Train
    m.train(inputs, expecteds, learnRate=0.1, gradThres=1e-4)

    # Test predictions
    for x, y_true in zip(inputs, expecteds):
        y_pred = m.compute(x)
        print(f"Input: {x}, Expected: {y_true}, Predicted: {y_pred}")
