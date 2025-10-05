Running each file runs a simple test for code written in that particular file.

# multitron.py

A Multitron is just a fancy name I gave to a linear layer in a neural network.

This file contains the raw mathematical implementation of a multitron and its learning rule, using nothing but numpy. I derived the rule using multivariable calculus by myself.

It is tested by training it on a simple linear function.

# megatron.py

A Megatron is the fancy name I gave to a full neural network. Once again, I've implemented the graadient descent rule by deriving it by hand first.

It is tested on XOR.

# megatron_unbatched.py

Same as a Megatron, but the implementation doesn't leverage matrix multiplication to process an entire batch at once. It instead processes training examples one by one.

# pytorch_xor.py

A neural network written using just the autograd engine of PyTorch (doesn't use the torch.nn package). Tested on XOR.

# torch_xor_nn.py

Same as pytorch_xor.py, but uses the nn module to concisely represent a simple feedforward NN.

# torch_mnist.py

A neural network trained on MNIST digits.
