# Learning Without Forgetting (shortened)

Split CIFAR-100 into 10 sequential tasks of 10 classes each (the class order is fixed in advance). Train one model that encounters each task once and cannot revisit that task's training examples afterward. At inference, the model must classify among all 100 classes without receiving a task identifier. Persistent auxiliary state carried between tasks is limited to 1 MiB, excluding model parameters. No pretrained weights or external data.

Maximize final top-1 accuracy over all 100 classes on the CIFAR-100 test set, measured after the last task.

Budget: 15 minutes of total training time on one NVIDIA A5000 GPU for the whole 10-task sequence (5,000 training images per task). Exceeding the budget or the 1 MiB auxiliary-state limit makes the run invalid.
