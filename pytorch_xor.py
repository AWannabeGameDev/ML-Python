import torch
from math import sqrt

# dataset
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = torch.tensor([0, 1, 1, 0])

# model
layerWeights = ((2 * torch.rand((2, 3)) - 1).requires_grad_(), 
                (2 * torch.rand((1, 3)) - 1).requires_grad_())
# activation = lambda x: 1 / (1 + torch.exp(-x)) # sigmoid did not give good result!
activation = lambda x: torch.tanh(x) # tanh, sigmoid did not work!

# learning parameters
learnRate = 0.1
gradThres = 0.001
batchCount = inputs.shape[0]
maxEpochs = 10000

for _ in range(maxEpochs):
    for (input, expected) in zip(inputs, outputs):
        # forward pass

        lastOut = input.float()

        for weightMatrix in layerWeights:
            lastOut = activation(weightMatrix @ torch.cat([torch.tensor([1.0]), lastOut], dim=0))

        # compute loss and propagate gradients

        differenceNorm = (lastOut - expected).norm()
        loss = 0.5 * differenceNorm * differenceNorm / batchCount
        loss.backward()

    with torch.no_grad():
        # check convergence condition

        netGradientNorm = 0

        for weightMatrix in layerWeights:
            gradientNorm = weightMatrix.grad.norm()
            netGradientNorm += gradientNorm * gradientNorm

        netGradientNorm = sqrt(netGradientNorm)

        if netGradientNorm <= gradThres:
            break

        # update weights and reset gradients
        for (layerIdx, weightMatrix) in enumerate(layerWeights):
            weightMatrix -= learnRate * weightMatrix.grad
            weightMatrix.grad.zero_()

# test

inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
expected_outputs = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)

print("Testing XOR network:")
for inp, expected in zip(inputs, expected_outputs):
    lastOut = inp
    for weightMatrix in layerWeights:
        lastOut = activation(weightMatrix @ torch.cat([torch.tensor([1.0]), lastOut], dim=0))
    
    print(f"Input: {inp.tolist()} | Predicted: {lastOut.item():.4f} | Expected: {expected.item()}")