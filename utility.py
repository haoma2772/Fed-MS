import yaml
import os
import pickle
import  torch

from tqdm import tqdm

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_dataset(config):


    dataset_name = config['dataset_paras']['name']
    Dalpha = config['fed_paras']['dirichlet_rate']

    # read data from the saved file
    file_path = os.path.join('data','distribution', dataset_name)
    tmp = 'dalpha=' + str(Dalpha)
    file_path = os.path.join(file_path, tmp)

    os.makedirs(file_path, exist_ok=True)

    dataset_train_path = os.path.join(file_path, 'dataset_train_list.pkl')
    dataset_test_path = os.path.join(file_path, 'dataset_test.pkl')


    with open(dataset_train_path, 'rb') as f:
        dataset_train_list = pickle.load(f)

    with open(dataset_test_path, 'rb') as f:
        dataset_test = pickle.load(f)


    return dataset_train_list, dataset_test


def train_model(model:torch.nn.Module, epoch, train_loader, config):
 
    cuda_num = config['train_paras']['cuda_number']
    device = torch.device('cuda:{}'.format(cuda_num) if torch.cuda.is_available() and cuda_num != -1 else 'cpu')
    lr = config['train_paras']['learning_rate']
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0  


    for data, target in tqdm(train_loader, total=len(train_loader), desc=f"Training epoch: {epoch}"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    # Calculate the average loss for the epoch
    average_loss = running_loss / len(train_loader)


    return model, average_loss


def test_model(model, test_loader, config):
    cuda_num = config['train_paras']['cuda_number']
    device = torch.device('cuda:{}'.format(cuda_num) if torch.cuda.is_available() and cuda_num != -1 else 'cpu')

    model.eval()
    model.to(device)

    test_loss = 0
    correct = 0
    total = 0
    # Your loss function, you can change this based on your task
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader), desc=f"Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            total += target.size(0)
            # Get the index of the max log-probability
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()


    test_loss /= len(test_loader)
    accuracy = 100*correct / total
    
    return accuracy, test_loss


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.zeros_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

