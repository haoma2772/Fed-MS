import torch
import copy


def noise_attacks(select_server_paras, noise_mean, noise_std, config):

  cuda_num = config['train_paras']['cuda_number']
  device = torch.device('cuda:{}'.format(cuda_num) if torch.cuda.is_available() and cuda_num != -1 else 'cpu')

  paras_select = copy.deepcopy(select_server_paras)

  for k, v in paras_select.items():

      noise = torch.normal(mean=noise_mean, std=noise_std, size=v.size()).to(device)

      paras_select[k] += noise
  
  return paras_select




def random_attacks(select_server_paras, random_lower, random_upper, config):

  cuda_num = config['train_paras']['cuda_number']
  device = torch.device('cuda:{}'.format(cuda_num) if torch.cuda.is_available() and cuda_num != -1 else 'cpu')

  paras_select = copy.deepcopy(select_server_paras)

  for k, v in paras_select.items():
      
      paras_select[k] = (random_lower + (random_upper - random_lower) * torch.rand_like(v)).to(device)
  
  return paras_select


def safeguard_attacks(select_server_paras, old_server_paras, scale_factor, config):
  paras_select = copy.deepcopy(select_server_paras)
  for k in paras_select.keys():

      paras_select[k] = (1+scale_factor)*old_server_paras[k] - scale_factor*select_server_paras[k]
  
  return paras_select


def backward_attacks(select_server_paras, old_server_paras, round, config):

  
  return old_server_paras
  