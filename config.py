CLASS_INFO = {
    'num': 2,
    'name': [
        'rectangle',
        'square',
    ]
}

TRAINING_CFG = {
    'path': './data/train/',
    'class': CLASS_INFO,
    'image_size': (224, 224),
    'modelname': 'mobilenetv3s',
    'epoch': 30, 
    'batch_size': 128,
    'optimizer': 'Adam',
    'learning_rate': 1e-5,
    'loss': 'CE',
    'model_savepath': './weights/',
    'load_checkpoint': None,
}

VALID_CFG = {
    'path': './data/valid/',
    'class': CLASS_INFO,
    'image_size': (224, 224),
    'batch_size': 16,
}

TESTING_CFG = {
    'path': './data/test/', # image path or folder path
    'class': CLASS_INFO,
    'image_size': (224, 224),
    'modelname': 'resnet18',
    'load_checkpoint': '/weights/resnet18_best.pth',
    'result_path': './result'
}

MODEL_AVAILABLE = ['custom', 'resnet18', 'resnet50', 'resnet152', 'mobilenetv3s']
OPTIMIZER_AVAILABLE = ['Adam']
LOSS_AVAILABLE = ['custom', 'CE']
