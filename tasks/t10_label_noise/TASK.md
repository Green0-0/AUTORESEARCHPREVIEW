# Label-Noise Survivor (CIFAR-10, 40% symmetric label noise)

Begin with CIFAR-10 where 40% of the training labels have been replaced at random: a fixed 40% of the 50,000 training images were given a wrong label, drawn uniformly from the 9 other classes (the noisy label file is fixed and provided). The clean training labels are not available. Find novel training procedures that maximize clean test top-1 accuracy.

Constraints:
- Train from scratch: using any external pretrained weights, external data, or downloads is banned.
- Only the 50,000 training images and their noisy labels may be used for training; test images and labels must not be used in any way.
- Training must take no longer than 15 minutes on 1 A5000 GPU (inference on the test set is a separate, short evaluation step that must not do further training).
