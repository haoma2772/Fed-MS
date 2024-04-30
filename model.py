import torchvision.models as models


def get_model(config):
    
    dataset_name = config['dataset_paras']['name']
    model_name = config['model_paras']['name']
    if dataset_name == 'mnist':
        input_size = 224 * 224 
        num_classes = 10
        number_channels = 1
    elif dataset_name in ['cifar10', 'cifar100']:
        input_size = 224 * 224 * 3
        number_channels = 3
        num_classes = 10 if dataset_name == 'cifar10' else 100
    elif dataset_name == 'imagenet1k':
        input_size = 224 * 224 * 3 
        num_classes = 1000  
        number_channels = 3
    elif dataset_name == 'gtsrb':
        input_size = 3*224*224
        num_classes = 43  
        number_channels = 3
    
    if model_name == 'MobileNet':

        return models.mobilenet_v2(weights=None, num_classes=num_classes)




