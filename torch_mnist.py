import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.cuda as cuda
from math import prod, sqrt

# dataset

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: torch.flatten(x))
])

trainData = datasets.MNIST (
    root="./data", train=True, transform=preprocess, download=True
)
testData = datasets.MNIST (
    root="./data", train=False, transform=preprocess, download=True
)

trainLoader = DataLoader(trainData, 64, True, num_workers=4, pin_memory=True)
testLoader = DataLoader(testData, 64, True, num_workers=4, pin_memory=True)
batchCount = len(trainLoader)

inputSize = prod(trainData.data[0].shape)
outputSize = 10 # 10 digits

# model

device = torch.device("cuda" if cuda.is_available() else "cpu")

hiddenNeuronCount1 = 128
hiddenNeuronCount2 = 64
model = nn.Sequential ( # outputs logits, gotta softmax manually for probabilities
    nn.Linear(inputSize, hiddenNeuronCount1),
    nn.ReLU(),
    nn.Linear(hiddenNeuronCount1, hiddenNeuronCount2),
    nn.ReLU(),
    nn.Linear(hiddenNeuronCount2, outputSize)
).to(device)

# hyperparameters

learnRate = 0.01
momentum = 0.7
gradThres = 0.001
maxEpochs = 20
lossFuncn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), learnRate, momentum)

model.train()

for epoch in range(maxEpochs):  
    epochMeanGradientRms = 0
    epochMeanLoss = 0

    for inputs, labels in trainLoader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        prediction = model.forward(inputs)
        loss = lossFuncn(prediction, labels)
        loss.backward()

        epochMeanLoss += loss.item() / batchCount

        # calculate gradient norm

        with torch.no_grad():
            netGradientSqrSum = 0
            weightCount = 0

            for weightMatrix in model.parameters():
                gradientNorm = weightMatrix.grad.norm().item()
                netGradientSqrSum += gradientNorm * gradientNorm
                weightCount += weightMatrix.grad.numel()

            epochMeanGradientRms += sqrt(netGradientSqrSum / weightCount) / batchCount

        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        print(f"Epoch {epoch}; Loss: {epochMeanLoss}; Gradient: {epochMeanGradientRms}")

        # check for early convergence (just checks if the gradient was "flat enough" in this epoch)

        if epochMeanGradientRms <= gradThres:
            print(f"Exiting early on epoch {epoch}/{maxEpochs}")
            break

# test

with torch.no_grad():
    correct = 0

    model.eval()

    for inputs, actualDigits in testLoader:
        inputs = inputs.to(device)
        actualDigits = actualDigits.to(device)

        _, predictedDigits = model.forward(inputs).max(dim=1)
        correct += ((predictedDigits - actualDigits) == 0).sum()

    accuracy = correct / testData.data.shape[0]

    print(f"Accuracy on test data: {accuracy * 100}%")