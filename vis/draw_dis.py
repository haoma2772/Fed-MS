import copy
import random

from utility import load_config

import math

from utility import get_dataset
from torch.utils.data import DataLoader
import numpy as np


if __name__ == '__main__':


        import matplotlib.pyplot as plt
        colors = plt.cm.tab10.colors
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.family'] = 'Times New Roman'


        print()
        path = 'config.yaml'
        config = load_config(path)

        global_round = config['fed_paras']['round']
        client_number = config['fed_paras']['client_number']
        split_rate = config['fed_paras']['split_rate']
        dirichlet_rate = config['fed_paras']['dirichlet_rate']
        num_workers = config['dataset_paras']['num_workers']
        batch_size = config['dataset_paras']['batch_size']
        dataset_name = config['dataset_paras']['name']
        model_name = config['model_paras']['name']
        attacks_name = config['general_paras']['attacks']
        defense_method = config['general_paras']['defense']
        attacker_rate = config['fed_paras']['attacker_rate']
        server_num = config['fed_paras']['server_num']
        attacker_num: int = int(attacker_rate * server_num)
        cuda_num = config['train_paras']['cuda_number']
        local_epoch = config['train_paras']['epoch']
        scale_factor = config['general_paras']['scale_factor']
        noise_mean = config['general_paras']['noise_mean']
        noise_std = config['general_paras']['noise_std']
        random_lower = config['general_paras']['random_lower']
        random_upper = config['general_paras']['random_upper']


        train_dataset_list, clean_test_dataset_list= get_dataset(
             config=config)


        dataset_train_loader_list = []

        dataset_train_loader_list = [
                DataLoader(train_dataset_list[client_id], batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                           shuffle=True) for
                client_id in range(client_number)]

        for idx in range(client_number):
             print('client idx is {}, the number of sample is {}'.format(idx, len(train_dataset_list[idx])))

        num_classes = 10

        client_labels = []
        for i, classes in enumerate(dataset_train_loader_list):
            label_counts = [0] * num_classes  
            for batch_idx, (_, lab) in enumerate(dataset_train_loader_list[i]):
                # 统计每个标签的数量
                for label in lab.tolist():
                    label_counts[label] += 1
            client_labels.append(label_counts)

        total_samples = 50000*5


        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.13, right=0.99, top=0.98, bottom=0.13)
        num_clients = 10
        for i in range(num_clients):

            drawn_length = 0

            for j, label_count in enumerate(client_labels[i]):
                x_min = drawn_length / total_samples * 100
                x_max = (drawn_length + label_count) / total_samples * 100
                plt.hlines(y=i, xmin=x_min, xmax=x_max, linewidth=15, color=colors[j],
                           alpha=0.7)
  
                drawn_length += label_count

        ax.set_xlabel('Sample Proportion(%)', color='black', fontsize=40, weight='bold')
        ax.set_ylabel('Client Index', color='black', fontsize=40,weight='bold')
        ax.set_xticks(np.arange(0, 5, 1))  # Adjust as needed
        ax.set_yticks(np.arange(num_clients))

        ax.set_xticklabels(ax.get_xticks(), color='black', fontsize=30,weight='bold')
        ax.set_yticklabels(ax.get_yticks(), color='black', fontsize=30,weight='bold')
        legend_labels = ['Class {}'.format(i) for i in range(len(client_labels[0]))]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(0.75, 1), prop={'weight': 'bold',
                                                                                   'size': 24, })

        plt.tight_layout()
        pic_name = 'distribution_alpha=' + str(dirichlet_rate) + '.pdf'
        plt.savefig(pic_name, bbox_inches='tight')
        plt.show()







