


from utility import load_config
from plugin import dirichlet_distribution
import pickle
import os
from plugin import load_dataset
from utility import load_config


def generate_noniid_distribution(config):

    train_dataset = load_dataset(config, trained=True)
    test_dataset = load_dataset(config, trained=False)
    train_dataset_list = dirichlet_distribution(train_dataset, config)

    dataset_name = config['dataset_paras']['name']
    Dalpha = config['fed_paras']['dirichlet_rate']

    file_path = os.path.join(config['dataset_paras']['save_path'],'distribution', dataset_name)
    tmp = 'dalpha=' + str(Dalpha)
    file_path = os.path.join(file_path, tmp)

    os.makedirs(file_path, exist_ok=True)

    dataset_train_path = os.path.join(file_path, 'dataset_train_list.pkl')
    dataset_test_path = os.path.join(file_path, 'dataset_test.pkl')
    with open(dataset_train_path, 'wb') as f:
        pickle.dump(train_dataset_list, f)

    with open(dataset_test_path, 'wb') as f:
        pickle.dump(test_dataset, f)


if __name__ == '__main__':
    path = 'config.yaml'
    config_list = load_config(path)
    generate_noniid_distribution(config=config_list)




