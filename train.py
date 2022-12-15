import os
import time, datetime, math
import torch
from torch.utils.data import DataLoader

from data import LoadDataset
from config import *
from model import init_model
from optimizer import init_optimizer
from loss import init_loss


def init_training_setting(train_config):
    model = init_model(train_config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = init_optimizer(model, train_config)
    loss_fn = init_loss(train_config)

    return device, model, optimizer, loss_fn



def train(train_config, valid_config):
    model_type = train_config['modelname']
    train_data = LoadDataset(train_config)
    valid_data = LoadDataset(valid_config)
    train_loader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=valid_config['batch_size'])
    

    device, model, optimizer, loss_fn = init_training_setting(train_config)
    model.to(device)

    highest_acc = 0

    if not os.path.exists(train_config['model_savepath']):
        os.mkdir(train_config['model_savepath'])

    batch_number = math.ceil(len(train_loader.dataset) / train_config['batch_size'])
    for epoch in range(train_config['epoch']):
        # Start Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            start = time.time()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            stop = time.time()
            if batch_idx % 10 == 9:
                iter_num = batch_idx * len(data)
                total_data = len(train_loader.dataset)
                iter_num = str(iter_num).zfill(len(str(total_data)))
                total_percent = 100. * batch_idx / len(train_loader)
                runtime = stop - start
                eta = int(runtime * (batch_number - batch_idx) + runtime * batch_number * (train_config['epoch'] - 1 - epoch))
                eta = str(datetime.timedelta(seconds=eta))
                print(f'Train Epoch {epoch}: [{iter_num}/{total_data} ({total_percent:2.0f}%)] |Loss: {loss.item():.6f} | ETA: {eta}')
                

        # Start Validating
        model.eval()
        correct = 0
        for (data, target) in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        print(str(correct) + '/' + str(len(valid_loader.dataset)))
        accuracy = 100. * correct / len(valid_loader.dataset)
        print('\nValid set: Accuracy: {}/{} ({:.2f}%)'.format(
            correct, len(valid_loader.dataset), accuracy))

        if accuracy >= highest_acc:
            highest_acc = accuracy
            torch.save(model.state_dict(), os.path.join(train_config['model_savepath'], f'{model_type}_best.pth'))
            print(f"Saving best model to {os.path.join(train_config['model_savepath'], f'{model_type}_best.pth')}\n")

if __name__ == '__main__':
    train(train_config = TRAINING_CFG,
          valid_config = VALID_CFG)