# Useful Synthetic Data (CIFAR-100)

You are given a fixed set of 10 real training examples per class from CIFAR-100 (1,000 images, 100 classes). Develop algorithms that transform this limited dataset into a synthetic training set of at most 100 examples per class (32x32 RGB images with hard class labels). No external data or pretrained models are permitted: the only information available is those 1,000 labelled images (plus whatever models you train from scratch on them during synthesis).

The synthetic set is judged by how useful it is for training classifiers from scratch. It is evaluated by transfer across three fixed classifier architectures, each trained from scratch on the synthetic set with a fixed recipe and then tested on the CIFAR-100 test set. One of the three architectures is withheld during development, so a set over-fitted to the visible architectures will not transfer.

Seek distinct approaches near the frontier of downstream accuracy, synthesis time, and storage.

**Primary metric:** mean CIFAR-100 test top-1 accuracy over the three fixed classifiers (higher is better).
**Secondary (reported, frontier):** synthesis wall time and storage size of the synthetic set.

**Budget:** 15 minutes of synthesis on one A5000 GPU (hard limit; overrunning invalidates the run). Training the three evaluation classifiers is a fixed cost of the evaluation and is not counted.

**Constraints:**
- At most 100 synthetic examples per class; labels are hard integer class labels.
- Use only the provided 1,000 real images; no other CIFAR data, external data, or pretrained weights.
- The evaluation classifiers, their training recipes and their augmentation are fixed; only the synthetic set can be changed.
