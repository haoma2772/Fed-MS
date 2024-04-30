from wandb import AlertLevel
import torch
import datetime
import wandb
from utility import load_config, train_model, test_model
from model import get_model
from utility import get_dataset
from torch.utils.data import DataLoader
import copy
import pickle
import os
import attack_types
import defense_types


if __name__ == '__main__':

        project_name = 'debug_fedms'
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

        device = torch.device(
            'cuda:{}'.format(cuda_num) if torch.cuda.is_available() and cuda_num != -1 else 'cpu')
        

        now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        runs_name = (dataset_name + '_' + model_name + '_' + attacks_name + '_' + defense_method + '({date})'.format(
                date=now_time) + 'dalpha=' +
                        str(dirichlet_rate) + 'attack_rate='+str(attacker_rate))


        wandb.init(project=project_name, name=runs_name, config=config)

        wandb.alert(
                title="{}".format(runs_name),
                text="{} Code starts running".format(runs_name),
                level=AlertLevel.WARN,
                wait_duration=1, )
        
        train_dataset_list, test_dataset = get_dataset(config=config)

        dataset_train_loader_list = [
                DataLoader(train_dataset_list[client_id], batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                           shuffle=True) for
                client_id in range(client_number)]
        dataset_test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                         shuffle=False)
        

        attacker_id:list = torch.arange(0,attacker_num)

        base_model = get_model(config=config)
        base_model.to(device)
        model_list = [copy.deepcopy(base_model) for _ in range(client_number)]
        old_model_list = copy.deepcopy(model_list)
        zeros_model = copy.deepcopy(base_model)
        zeros_model = zeros_model.to(device)
        for key in zeros_model.state_dict():
                torch.nn.init.zeros_(zeros_model.state_dict()[key])

        server_glob = base_model.state_dict()
        server_zero = zeros_model.state_dict()
        init_record_list = {'test_acc': [], 'test_loss': []}
        whole_init_record_list = [copy.deepcopy(init_record_list) for i in range(client_number)]
        server_paras = [copy.deepcopy(server_zero) for i in range(server_num)]

        for each_round in range(global_round):

                
                select_server_id: list = torch.randint(0, server_num, (client_number,))
                times = torch.zeros(server_num, dtype=torch.int)
                
                if attacks_name == 'safeguard':
                        old_server_paras = copy.deepcopy(server_paras)
                elif attacks_name == 'backward':
                        old_server_paras2 = copy.deepcopy(old_server_paras)
                        old_server_paras = copy.deepcopy(server_paras)
                server_paras = [copy.deepcopy(server_zero) for i in range(server_num)]
                
                for client_id in range(client_number):

                        # local train
                 
                        for ep in range(local_epoch):
                                train_loss_ep = ep + each_round * local_epoch
                                model_list[client_id], tmp_train_loss=train_model(model=model_list[client_id], epoch=train_loss_ep, 
                                        train_loader=dataset_train_loader_list[client_id],config=config)
                                # test
                                test_acc, test_loss = test_model(model=model_list[client_id], test_loader=dataset_test_loader, config=config)
                                wandb.log({'Testing Accuracy of client{}'.format(client_id):test_acc,
                                        'Testing Loss of client{}'.format(client_id): test_loss, 
                                        'Training loss of client {}'.format(client_id): tmp_train_loss,
                                        'Round': train_loss_ep})
                                whole_init_record_list[client_id]['test_acc'].append(test_acc)
                                whole_init_record_list[client_id]['test_loss'].append(test_loss)




                        for k in model_list[client_id].state_dict().keys():
                                server_paras[select_server_id[client_id]][k] += model_list[client_id].state_dict()[k]
                        times[select_server_id[client_id]] += 1

                # deal with the parameters 
                for ps in range(server_num):
                        for k in server_paras[ps].keys():
                                server_paras[ps][k] = torch.div(server_paras[ps][k], max(1,times[ps]))   

                for idx in attacker_id:    
                        if attacks_name == 'Noise':
                                server_paras[idx] = attack_types.noise_attacks(select_server_paras=server_paras[idx], 
                                                        noise_mean=noise_mean, noise_std=noise_std, config=config)
                        elif attacks_name == 'Random':
                                server_paras[idx] = attack_types.random_attacks(select_server_paras=server_paras[idx], random_lower=random_lower,
                                                                random_upper=random_upper, config=config)
                        elif attacks_name == 'Safeguard':
                                server_paras[idx] = attack_types.safeguard_attacks(select_server_paras=server_paras[idx], old_server_paras=old_server_paras[idx],
                                                                scale_factor=scale_factor, config=config)
                        elif attacks_name == 'Backward':
                                server_paras[idx] = attack_types.backward_attacks(select_server_paras=server_paras[idx], old_server_paras=old_server_paras2[idx],
                                                                round=each_round, config=config)
                                
                

                #local client update their model

                if defense_method == 'Vanilla FL':
                        tmp_paras = defense_types.avg_defense(server_list=server_paras, device=device, config=config, zero_model=zeros_model)
                        model_list[0].load_state_dict(tmp_paras)

                elif defense_method == 'Fed-MS':
                        tmp_paras = defense_types.trimmed_defense(server_list=server_paras, device=device, config=config, zero_model=zeros_model)
                        model_list[0].load_state_dict(tmp_paras)
    
                # fourth update self model and test
                for client_id in range(client_number):
                        if client_id == 0:
                                pass
                        else:
                                model_list[client_id].load_state_dict(model_list[0].state_dict())
                
    
        res_file = config['general_paras']['output_dir']
        tmp_file_name = dataset_name + '_' + defense_method + '_' + attacks_name + '_' + str(dirichlet_rate)+'_'+str(attacker_rate)
        save_path = os.path.join(res_file, tmp_file_name)
        if not os.path.exists(save_path):
                os.makedirs(save_path)


        file_path = os.path.join(save_path, 'record_res.pkl')
        with open(file_path, 'wb') as f:
                pickle.dump(whole_init_record_list, f)

        
        wandb.alert(
            title=runs_name,
            text="{} End of code run!".format(runs_name),
            level=AlertLevel.WARN,
            wait_duration=1, )
        
        wandb.finish()         
        
        
        
        

          
        

        




