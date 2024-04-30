import torch
import copy
import numpy as np


def avg_defense(server_list, device, config, zero_model):

    num_server = len(server_list)
    local_paras = copy.deepcopy(zero_model.state_dict())
    for key in server_list[0].keys():
        tensors_list = [torch.flatten(server_list[i][key].to(device)) for i in range(num_server)]
        stacked_tensor = torch.stack(tensors_list)
           
        avg_tensor = torch.mean(stacked_tensor, dim=0)
        local_paras[key] = torch.reshape(avg_tensor, server_list[0][key].shape)
    
    return local_paras



def median_defense(server_list, device, config, zero_model):
        # server_list store the pseudo gradient
    num_server = len(server_list)
    local_paras = copy.deepcopy(zero_model.state_dict())
    for key in server_list[0].keys():
        tensors_list = [torch.flatten(server_list[i][key].to(device)) for i in range(num_server)]
        stacked_tensor = torch.stack(tensors_list)
        sort_tensor, indices = torch.sort(stacked_tensor, dim=0)    
        avg_tensor = torch.median(sort_tensor, dim=0)[0]
        local_paras[key] = torch.reshape(avg_tensor, server_list[0][key].shape)
    
    return local_paras


def compute_euclidean_distance(paras1, paras2):
    distance = 0.0

    for param1, param2 in zip(paras1.values(), paras2.values()):
        param1_float = param1.to(torch.float)
        param2_float = param2.to(torch.float)
        distance += torch.norm(param1_float - param2_float, p='fro')

    return distance.item()


def compute_scores(distances, i, n, f):


    s = [distances[j][i] for j in range(i)] + [
            distances[i][j] for j in range(i + 1, n)
        ]
        
        # 对列表 s 进行排序，并选择前 n - f - 2 个最小的距离的平方
    _s = sorted(s)[: n - f - 2]
        
        # 返回选定距离的平方之和作为节点 i 的 Krum 距离得分
    return sum(_s)


def krum_defense(server_list, device, config, zero_model, k=1):
    # for each client select n-f neighbor for compute scores and select k for avg
    # n < 2*f + 2
    # server_list 是模型列表
    rate_attacker = config['fed_paras']['attacker_rate']
    num_server = len(server_list)
    num_attacker = int(rate_attacker * num_server)
    

    distances = {}
    for i in range(num_server-1):
        distances[i] = {}
        for j in range(i + 1, num_server):
            distances[i][j] = compute_euclidean_distance(server_list[i], server_list[j])
    

            
    scores = [(i, compute_scores(distances, i, num_server, num_attacker)) for i in range(num_server)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    top_m_indices = list(map(lambda x: x[0], sorted_scores))[:k]

    # local_paras = avg_defense(server_list[top_m_indices], )
    return server_list[top_m_indices[0]]
    
 


def trimmed_defense(server_list, device, config, zero_model):
    rate_attacker = config['fed_paras']['attacker_rate']
    num_server = len(server_list)
    num_attacker = int(rate_attacker * num_server)
    local_paras = copy.deepcopy(zero_model.state_dict())
    for key in server_list[0].keys():
        tensors_list = [torch.flatten(server_list[i][key].to(device)) for i in range(num_server)]
        stacked_tensor = torch.stack(tensors_list)
        sort_tensor, indices = torch.sort(stacked_tensor, dim=0)
        sta_index = num_attacker
        end_index = num_server - num_attacker
        # 左闭右开
        avg_tensor = sort_tensor[sta_index:end_index]
        avg_tensor = torch.mean(avg_tensor, dim=0)
        local_paras[key] = torch.reshape(avg_tensor, server_list[0][key].shape)
    
    return local_paras

def trimmed_defense_minus(server_list, device, config, zero_model):
    rate_attacker = config['fed_paras']['attacker_rate']
    num_server = len(server_list)
    num_attacker = int(rate_attacker * num_server)
    local_paras = copy.deepcopy(zero_model.state_dict())
    for key in server_list[0].keys():
        tensors_list = [torch.flatten(server_list[i][key].to(device)) for i in range(num_server)]
        stacked_tensor = torch.stack(tensors_list)
        sort_tensor, indices = torch.sort(stacked_tensor, dim=0)
        sta_index = num_attacker - 1
        end_index = num_server - num_attacker + 1
        # 左闭右开
        avg_tensor = sort_tensor[sta_index:end_index]
        avg_tensor = torch.mean(avg_tensor, dim=0)
        local_paras[key] = torch.reshape(avg_tensor, server_list[0][key].shape)
    
    return local_paras


def trimmed_defense_plus(server_list, device, config, zero_model):
    rate_attacker = config['fed_paras']['attacker_rate']
    num_server = len(server_list)
    num_attacker = int(rate_attacker * num_server)
    local_paras = copy.deepcopy(zero_model.state_dict())
    for key in server_list[0].keys():
        tensors_list = [torch.flatten(server_list[i][key].to(device)) for i in range(num_server)]
        stacked_tensor = torch.stack(tensors_list)
        sort_tensor, indices = torch.sort(stacked_tensor, dim=0)
        sta_index = num_attacker +1
        end_index = num_server - num_attacker - 1
        # 左闭右开
        avg_tensor = sort_tensor[sta_index:end_index]
        avg_tensor = torch.mean(avg_tensor, dim=0)
        local_paras[key] = torch.reshape(avg_tensor, server_list[0][key].shape)
    
    return local_paras








