Activation Tuning (CIFAR-100).

We use one dataset: CIFAR-100. Instead of a CNN or ResNet, the model is a generic fully-connected network (ANN) with 3 hidden layers of a fixed width of 1024 units: the flattened, normalized 32x32x3 image goes through Linear(3072→1024), Linear(1024→1024), Linear(1024→1024) and a final Linear(1024→100) that outputs the class logits. The number of parameters in these connections is fixed, and the architecture may not be modified: no extra layers, skip connections, extra branches or changes to the weight shapes.

Each hidden layer computes f(Wx), where W (with bias) is that layer's fixed-shape weight, x is the layer's input and f is the activation function. The activation functions may be freely redesigned, and each of the 3 hidden layers may use a different one. The constraints are:
- the activation function must take Wx (that layer's pre-activation, and nothing else) as its input and return a vector of the same width;
- no more than 20% of the network's total parameters may be in the activation functions;
- at test time the activation must act on each example independently and deterministically.

The training procedure for the network (optimizer, schedule, loss, data augmentation, initialization, number of epochs) may be adapted to suit the proposed activation, but no other network may be trained or used (no teachers, no ensembles, no pretrained weights, no external data). Training uses only the 50,000 CIFAR-100 training images.

The goal is to maximize top-1 accuracy on the CIFAR-100 test set. Training must finish within 15 minutes on 1 NVIDIA A5000 GPU.
