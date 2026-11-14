# Data Ordering as the Only Control (CIFAR-100)

Train a CIFAR-100 classifier where everything except the data order is fixed: the architecture (a small ResNet), the initialization distribution and seed, the optimizer and learning-rate schedule, the loss, the data augmentations (random crop and horizontal flip), the batch size and the number of updates. The harness owns the training loop. The only thing you control is **which training examples go into each batch**, that is, the order in which training examples are presented and how often each one is repeated. The total number of example presentations is therefore fixed (batch size × number of updates). Any example may be shown any number of times, including zero.

To decide the order, you may use any information from the training set (images, labels) and feedback the harness provides during training (the per-example losses and outputs of each batch just trained on, and extra forward passes of the current model on training examples). You may not change the model, its weights, the optimizer or the augmentation, and you may not use the test set or any external data or pretrained models.

**Budget:** 15 minutes of wall-clock time on one A5000 GPU for the whole training process. Fixed training takes about 8.5 minutes of that. The rest is headroom, and all scoring and scheduling computation, including extra forward passes, counts toward the budget. A run that exceeds the budget is invalid.

**Metric:** top-1 accuracy on the CIFAR-100 test set (higher is better), measured on the final model. The baseline is a uniform random reshuffle of the training set every epoch.
