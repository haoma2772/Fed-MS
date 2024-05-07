# Fed-MS: Fault Tolerant Federated Edge Learning with Multiple Byzantine Servers

## Abstract
Due to its decentralized framework and outdoor environments, federated edge learning (FEEL) faces significant vulnerability to malicious attacks within edge networks. Prevailing FEEL approaches typically hinge on a dependable parameter server (PS) to contend with the adversarial updates from Byzantine clients. Recognizing the inherent unreliability of PSs in edge networks, this paper delves into the security challenges of FEEL, specifically addressing Byzantine PSs. We present a Byzantine fault-tolerant FEEL algorithm, named Fed-MS, in which a multi-server technique along with a newly designed trimmed-mean-based model filter is employed. This combination ensures that each client can obtain a feasible global model for its local training, closely approximating a true model aggregated by benign PSs. Furthermore, we propose a sparse uploading strategy in Fed-MS to enhance communication efficiency for model aggregation to multiple PSs. Theoretical analysis demonstrates that, when Byzantine PSs are a minority, Fed-MS achieves an expected convergence speed of $\mathcal{O}(1/T)$ with $T$ defined as the number of training rounds, akin to state-of-the-art works under non-Byzantine settings. Extensive experiments are conducted on the CIFAR-10 dataset with MobileNet V2 as the training model. The numerical results show that our Fed-MS can improve the model accuracy from 10% to at least 76% under the malicious attacks from Byzantine PSs.

## Environment Dependencies
- Python 3.9
- PyTorch 2.1.0
- CUDA 11.8
- torchvision
- torchaudio
- NVIDIA
- wandb

## Data Preparation
1. **Dataset Selection**: 
   - To select different datasets such as MNIST, CIFAR-10, or GTSRB (German Traffic Sign Recognition Benchmark) in your configuration, you would typically modify the `dataset` section of your `config.py` file. Here's how you can do it:
     - **MNIST**: Set the `name` parameter under `dataset_paras` to `'mnist'`.
     - **CIFAR-10**: Set the `name` parameter under `dataset_paras` to `'cifar10'`.
     - **GTSRB**: Set the `name` parameter under the `dataset_paras` to `'gtsrb'`.
2. **Obtaining the Dataset**: 
   - After configuring the dataset parameters in your `config.py` file as described in the previous step, you can obtain your dataset by running the `create_dataset.py` script. This script will download the dataset according to the path specified in the configuration file.

## Run Experiment

1. You can run the experiment by executing `python main.py`.
2. To adjust different parameters, modify the `config.py` file. Below is a description of each parameter:

### General Parameters

| Parameter Name | Description                                        |
|----------------|----------------------------------------------------|
| output_dir     | Directory to save the results.                     |
| random_seed    | Seed for random number generation.                 |
| defense        | Defense mechanism used ('Vanilla FL' or 'Fed-MS'). |
| attacks        | Type of attack ('Noise', 'Random', 'Safeguard', 'Backward'). |
| scale_factor   | Scaling factor used in 'Safeguard' defense.        |
| noise_std      | Standard deviation for 'Noise' attack.             |
| noise_mean     | Mean for 'Noise' attack.                           |
| random_lower   | Lower bound for 'Random' attack.                   |
| random_upper   | Upper bound for 'Random' attack.                   |

### Dataset Parameters

| Parameter Name | Description                                    |
|----------------|------------------------------------------------|
| name           | Name of the dataset ('cifar10' in this case).  |
| batch_size     | Batch size used during training.               |
| save_path      | Path to save the dataset.                      |
| num_workers    | Number of workers for data loading.            |
| num_classes    | Number of classes in the dataset.              |

### Model Parameters

| Parameter Name | Description                     |
|----------------|---------------------------------|
| name           | Name of the model ('MobileNet').|

### Federated Learning Parameters

| Parameter Name | Description                                   |
|----------------|-----------------------------------------------|
| round          | Number of communication rounds.               |
| iid            | Whether the data is independently and identically distributed (IID). |
| dirichlet_rate | Rate parameter for generating Dirichlet distribution. |
| client_number  | Number of clients participating in federated learning. |
| server_num     | Number of servers in the federated learning setting. |
| attacker_rate  | Rate of attackers in the system.              |
| split_rate     | Ratio of data split between train and test sets. |

### Training Parameters

| Parameter Name | Description                               |
|----------------|-------------------------------------------|
| learning_rate  | Learning rate for the optimizer.         |
| optimizer      | Optimization algorithm used.              |
| cuda_number    | Number of CUDA devices to use.            |
| epoch          | Number of epochs for training.            |
| criterion      | Loss function used for training.          |
