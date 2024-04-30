import torch
import numpy as np
from torch.utils.data import random_split, Subset
from torchvision import transforms
import torchvision



def split_data(dataset, train_ratio, config, random_seed=None):

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    if random_seed is not None:
        torch.manual_seed(random_seed)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset



def dirichlet_split_noniid(train_labels, alpha, n_clients, config):


    dataset_name = config['dataset_paras']['name']
    if dataset_name == 'cifar10':
        n_classes = 10
    elif dataset_name == 'gtsrb':
        n_classes = 43
    elif dataset_name == 'cifar100':
        n_classes = 100
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]


    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):

        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs


def dirichlet_distribution(dataset, config):
    total_samples = len(dataset)
    iid = config['fed_paras']['iid']
    n_clients = config['fed_paras']['client_number']
    dataset_name = config['dataset_paras']['name']

    if iid:
        samples_per_client = total_samples // n_clients
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        subsets = [indices[i*samples_per_client: (i+1)*samples_per_client] for i in range(0, n_clients)]
    else:
        DIRICHLET_ALPHA = config['fed_paras']['dirichlet_rate']
        if dataset_name == 'cifar10':
            input_sz, num_cls = dataset.data[0].shape[0], len(dataset.classes)
            train_labels = np.array(dataset.targets)
        elif dataset_name == 'gtsrb':
            num_cls = 43
            input_sz = 3 * 224 * 224
            train_labels = np.array(dataset.labels)
        elif dataset_name == 'cifar100':
            input_sz, num_cls = dataset.data[0].shape[0], len(dataset.classes)
            train_labels = np.array(dataset.targets)


        subsets = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=n_clients, config=config)

    local_dataset = []

    for subset_indices in subsets:
        subset = Subset(dataset, subset_indices)
        local_dataset.append(subset)
    return local_dataset





def load_dataset(config, trained=True):

    dataset_name = config['dataset_paras']['name']
    data_dir = config['dataset_paras']['save_path']

    resize_transform = transforms.Resize((224, 224))

    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            resize_transform,  
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = torchvision.datasets.MNIST(root=data_dir, train=trained, transform=transform, download=True)


    elif dataset_name == 'cifar10':

        transform = transforms.Compose([
            resize_transform,  
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=trained, transform=transform, download=True)
        

    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            resize_transform,  
            transforms.ToTensor(),

            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=trained, transform=transform, download=True)


    elif dataset_name == 'imagenet1k':

        transform = transforms.Compose([
            resize_transform,  
            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

    elif dataset_name == 'gtsrb':
        transform = transforms.Compose([
            resize_transform, 
            transforms.ToTensor(),

            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.GTSRB(root=data_dir, split='train',transform=transform, download=True)


    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    return train_dataset
