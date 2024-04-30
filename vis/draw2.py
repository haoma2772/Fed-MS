import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from utility import load_config
if __name__ == '__main__':

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
    
    local_epoch = config['train_paras']['epoch']
    num_epoch = global_round * local_epoch
    epoch = range(num_epoch)


    load_path = "multi_server_res"
    record_list = []
    defensse_name = ['Vanilla FL', 'Fed-MS', 'Fed-MS¯',]
    label_name = ['Vanilla FL','Fed-MS', 'Fed-MS¯',]

    total_acc = []
    #total_loss= []
    
    for tname in defensse_name:
        method_acc = []
        # method_loss = []
        tmp_name = dataset_name + '_' + tname + '_' + attacks_name + '_' + str(dirichlet_rate) + '_' + str(attacker_rate)
        file_path = os.path.join(load_path, tmp_name)
        record_path = os.path.join(file_path, 'record_res.pkl')
        with open(record_path, 'rb') as f:
            loaded_records = pickle.load(f)
            record_list.append(loaded_records)
        for benign_id in range(client_number):
            tmp_idx = benign_id
            tmp_acc = loaded_records[tmp_idx]['test_acc'][0:num_epoch]
            #tmp_loss = loaded_records[tmp_idx]['test_loss'][0:num_epoch]
            lowess_smooth_acc = sm.nonparametric.lowess(tmp_acc, epoch, frac=0)
            #lowess_smooth_loss = sm.nonparametric.lowess(tmp_loss, epoch, frac=0)

            method_acc.append(lowess_smooth_acc[:, 1])
            #method_loss.append(lowess_smooth_loss[:, 1])
        
        total_acc.append(method_acc)
        #total_loss.append(method_losss)
        
        

    import matplotlib.pyplot as plt
    colors = plt.cm.tab10.colors

    plt.rcParams['font.family'] = 'Times New Roman'
 
    color_ls=['#fa8c16','#52c41a','#3EECFF','#eb2f96','#1890ff','#722ed1','#FF9AE5','#f5222d',]
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.13, right=0.99, top=0.98, bottom=0.13)

    import statsmodels.api as sm
    for idx in range(len(defensse_name)):

        data = pd.DataFrame({'Epoch': np.tile(epoch, client_number), 'ASR': np.concatenate(total_acc[idx])})

        sns.lineplot(x='Epoch', y='ASR', data=data, color=color_ls[idx], label=label_name[idx],errorbar='sd', estimator='mean', linewidth=2,)

    le = ax.legend(loc='upper left', bbox_to_anchor=(0.7, 0.7), prop={'weight': 'bold',
                                                                                   'size': 24, })
    for line in le.get_lines():
        line.set_linewidth(6) 


    ax.set_xlabel('Epochs', color='black', fontsize=40, weight='bold')
    ax.set_ylabel('Accuracy (%)', color='black', fontsize=40,weight='bold')
    ax.set_xticks(np.arange(0,70,10))
    ax.set_yticks(np.arange(0,80,10))
    ax.set_xticklabels(ax.get_xticks(), color='black', fontsize=30,weight='bold')
    ax.set_yticklabels(ax.get_yticks(), color='black', fontsize=30,weight='bold')
    ax.grid(color='black', linestyle='-', linewidth=2, alpha=0.5)
    file_name1 = 'draw2_'+ attacks_name + '_ACC.pdf'
    plt.savefig(file_name1, dpi=3000)
    plt.clf()



    


