import torch
import torch.nn as nn
from math import sqrt

# dataset
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# model
model = nn.Sequential (
    nn.Linear(2, 2),
    nn.Tanh(),
    nn.Linear(2, 1),
    nn.Tanh()
)

# training parameters
maxEpochs = 10000
batchSize = inputs.shape[0]
gradThres = 0.001
lossFunction = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(maxEpochs):
    # process the entire batch at once: matrix magic. *could* loop through examples one by one but this is cleaner.
    predictions = model.forward(inputs) # forward pass
    loss = lossFunction(predictions, targets) / batchSize # calculate loss
    loss.backward() # calculate loss gradient

    # check early convergence (exit if gradient is "flat enough")

    with torch.no_grad():
        netGradientNorm = 0

        for weightMatrix in model.parameters():
            gradientNorm = weightMatrix.grad.norm()
            netGradientNorm += gradientNorm * gradientNorm

        netGradientNorm = sqrt(netGradientNorm)

        if netGradientNorm <=  gradThres:
            print(f"Exiting early on epoch {epoch}/{maxEpochs}.")
            break

    # update weights and reset gradients
    optimizer.step()
    optimizer.zero_grad()

else:
    print(f"Executed all {maxEpochs} epochs.")

# test
print("Testing XOR network:")
with torch.no_grad():
    for input, target in zip(inputs, targets):
        print(f"Input: {input.tolist()} | Predicted: {model.forward(input).item():.4f} | Expected: {target.item()}")