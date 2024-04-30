# Fed-MS: Fault Tolerant Federated Edge Learning with Multiple Byzantine Servers
## Abstract
...
## Environment Depdendencies
python 3.9
pytorch 2.1.0
cuda 11.8
torchvision
torchaudio
nvidia
wandb
##  Data Preparation
1.To select different datasets such as MNIST, CIFAR-10, or GTSRB (German Traffic Sign Recognition Benchmark) in your configuration, you would typically modify the dataset section of your config.py. Here's how you can do it:
MNIST: Set the name parameter under the dataset_paras'name to 'mnist'.
CIFAR-10: Set the name parameter under dataset_paras'name to 'cifar10'.
GTSRB: Set the name parameter under the dataset_paras'name to 'gtsrb'.
2. After configuring the dataset parameters in your config.py file as described in the previous step, you can obtain your dataset by running the create_dataset.py script. This script will download the dataset according to the path specified in the configuration file.
