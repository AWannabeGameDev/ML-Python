from typing import Sequence, Callable
import numpy as np
from math import sqrt

from multitron import *

Activation = Callable[[float], float]

class Megatron:
    def __init__(self, inputCount: int, tronSizes: Sequence[int], activation: Activation, dvActivation: Activation):
        self.__trons = list[Multitron]()
        self.tronCount = len(tronSizes)

        prevTronSize = inputCount
        for tronSize in tronSizes:
            self.__trons.append(Multitron(prevTronSize, tronSize))
            prevTronSize = tronSize

        self.__activate = activation
        self.__dvActivate = dvActivation

    def compute(self, input: np.ndarray) -> np.ndarray:
        output = input

        for tron in self.__trons:
            output = self.__activate(tron.compute(output))

        return output
    
    def train(self, inputs: np.ndarray, expecteds: np.ndarray, learnRate: float, gradThres: float):
        dataCount = inputs.shape[0]

        while True:
            meanGradLWs = list[np.ndarray]()

            for tron in self.__trons:
                meanGradLWs.append(np.zeros(tron._weights.shape))

            # the entire batch is processed at once using matrix multiplies

            # forward pass

            inactiveOuts = [inputs.T]
            outs = [inputs.T]
            dvActiveOuts = [inputs.T]

            for tron in self.__trons:
                inactiveOuts.append(tron.compute(outs[-1].T).T)
                outs.append(self.__activate(inactiveOuts[-1]))
                dvActiveOuts.append(self.__dvActivate(inactiveOuts[-1]))
        
            # compute gradLOs

            gradLOs = [outs[-1] - expecteds.T]

            for tronIdx in reversed(range(self.tronCount - 1)):
                gradLOsize =  gradLOs[-1].shape[0]
                term1 = (gradLOs[-1] * dvActiveOuts[(tronIdx + 1) + 1]).T[:, :, np.newaxis]
                gradLO = term1 * self.__trons[tronIdx + 1]._weights[:, 1:]
                gradLO = gradLO.sum(axis = 1).T
                gradLOs.append(gradLO)

            gradLOs.reverse()

            # compute gradLWs

            for tronIdx in range(self.tronCount):
                biasedPrevTronOut = np.vstack((np.ones((1, outs[(tronIdx + 1) - 1].shape[1])), outs[(tronIdx + 1) - 1]))
                meanGradLWs[tronIdx] += (gradLOs[tronIdx] * dvActiveOuts[tronIdx + 1]) @ biasedPrevTronOut.T / dataCount    

            # compute gradient norm and check threshold

            meanGradLWsNorm = 0

            for meanGradLWm in meanGradLWs:
                toAdd = np.linalg.norm(meanGradLWm)
                toAdd = toAdd * toAdd
                meanGradLWsNorm += toAdd

            meanGradLWsNorm = sqrt(meanGradLWsNorm)

            if meanGradLWsNorm <= gradThres:
                break

            # update weights

            for tronIdx in range(self.tronCount):
                self.__trons[tronIdx]._weights -= learnRate * meanGradLWs[tronIdx]

if __name__ == "__main__":
    from math import tanh, exp

    def dtanh(x: np.ndarray) -> np.ndarray:
        th = np.tanh(x)
        return 1 - th * th
    
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(x: np.ndarray) -> np.ndarray:
        sig = sigmoid(x)
        return x * (1 - x)

    # XOR dataset
    inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    outputs = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # Initialize Megatron: 2 input neurons, 1 hidden layer of 2 neurons, 1 output neuron
    megatron = Megatron(
        inputCount=2,
        tronSizes=[2, 1],
        activation=np.tanh,
        dvActivation=dtanh
    )

    # Training parameters
    learnRate = 0.1
    gradThres = 0.001

    # Train the network
    megatron.train(inputs, outputs, learnRate, gradThres)

    # Test predictions
    for x, y in zip(inputs, outputs):
        pred = megatron.compute(x)
        print(f"Input: {x}, Expected: {y}, Predicted: {pred}")
