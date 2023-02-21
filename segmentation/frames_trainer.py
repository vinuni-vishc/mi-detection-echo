import os
import tqdm
import torch
import wandb
from easydict import EasyDict
from torch.utils.data import DataLoader
from datasets.frame_dataset import MIFrameDataset
from video_utils import *
from utils.videos import AverageMeter
# os.environ['WANDB_MODE'] = 'offline'

sweep_config = {
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'test_acc'},
    'parameters': {
        'architecture': {
            'values': [ 'r3d_18', 'mc3_18'], # 'r2plus1d_18', 
        },
        'fold': {
            'values': [0, 1, 2, 3, 4],
        },
        'epoch': {
            'values': [50],
        },
        'batch_size': {
            'values': [16],
        },
        'img_shape': {
            'values': [(224, 224)],
        },
        'out_features': {
            'values': [1],
        },
        'len_window': {
            'values': [9, 7, 5],
        }
    }
}
        
def train_one_epoch(epoch, model, optimizer, train_loader, criterion, device):
    model.train()
    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    loss_all, acc_all = AverageMeter(), AverageMeter()
    
    for step, batch in pbar:    
        data, ytrue = batch[0].to(device).float(), batch[1].to(device).float()
        ypred = torch.sigmoid(model(data)).reshape(-1)
        loss = criterion(ypred, ytrue)
        
        preds = torch.round(ypred.data)
        running_corrects = preds.eq(ytrue.data).cpu().sum() / data.shape[0]
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_all.update(loss.item(), data.shape[0])
        acc_all.update(running_corrects, data.shape[0])
        
        pbar.set_description(f"train epoch {epoch + 1} batch {step + 1} / {len(pbar)} train-loss {loss_all.avg:.4f} train-acc {acc_all.avg:.4f}") 
    # # return evaluation, metrics
    return loss_all, acc_all

def valid_one_epoch(epoch, model, val_loader, criterion, device, val_txt='val'):
    '''test the model for one epoch with progress bar'''
    model.eval()
    pbar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
    loss_all, acc_all = AverageMeter(), AverageMeter()
    
    
    for step, batch in pbar:    
        data, ytrue = batch[0].to(device).float(), batch[1].to(device).float()
        ypred = torch.sigmoid(model(data)).reshape(-1)
        loss = criterion(ypred, ytrue)
        
        preds = torch.round(ypred.data)
        running_corrects = preds.eq(ytrue.data).cpu().sum() / data.shape[0]
                
        loss_all.update(loss.item(), data.shape[0])
        acc_all.update(running_corrects, data.shape[0])
        
        pbar.set_description(f"{val_txt} epoch {epoch + 1} batch {step + 1} / {len(pbar)} {val_txt}-loss {loss_all.avg:.4f} {val_txt}-acc {acc_all.avg:.4f}")
    
    return loss_all, acc_all


def run_n_epochs(config=None):
    # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=3vqGZbxxulea
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # architecture = config.architecture
    # out_features = config.out_features
    
    # sweep_config = {
    #     'batch_size': 64,
    #     'architecture': 'r3d_18',
    #     'epoch': 50,
    #     'fold': 0,
    #     'encoder': 'mobilenet_v2',
    #     'img_shape': (224, 224),
    #     'out_features': 1,
    # }                 
    # config = EasyDict(sweep_config)
    # if True:
    with wandb.init(config=config) as run:
        config = wandb.config
    
        run.name = f'{config.architecture}__{config.epoch}__{config.fold}'
        wandb.name = f'{config.architecture}__{config.epoch}__{config.fold}'
        print("Wandb name: {}".format(wandb.name))
        
        print(f"Start wandb with config: {config}")
        # folder_exp = f'/home/vishc2/tuannm/echo/vishc-echo/models/tmp/{config.architecture}__{config.epoch:03d}__{config.fold:03d}'
        # split=t, id_fold=f, len_window=32)
        train_set = MIFrameDataset(split='train', id_fold=config.fold, img_size=config.img_shape, len_window=config.len_window)
        val_set = MIFrameDataset(split='val', id_fold=config.fold, img_size=config.img_shape, len_window=config.len_window)
        test_set = MIFrameDataset(split='test', id_fold=config.fold, img_size=config.img_shape, len_window=config.len_window)
        
        train_loader = DataLoader(train_set, config.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, config.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, config.batch_size, shuffle=False)
        
        model, optimizer, criterion = get_model_and_optim(config, device=device)
        model = model.to(device)
        
        for epoch in range(config.epoch):
            train_loss, train_acc = train_one_epoch(epoch, model, optimizer, train_loader, criterion, device)
            val_loss, val_acc = valid_one_epoch(epoch, model, val_loader, criterion, device, val_txt='val')
            test_loss, test_acc = valid_one_epoch(epoch, model, test_loader, criterion, device, val_txt='test')
            
            # if test_acc.avg >= 0.75:
            #     fn_md = f'{folder_exp}__{epoch:03d}__{train_acc.avg:.4f}__{val_acc.avg:.4f}__{test_acc.avg:.4f}.pth'
            #     os.makedirs(os.path.dirname(fn_md), exist_ok=True)
            #     torch.save(model.state_dict(), fn_md)
            
            wandb.log({'epoch': epoch, 'train_loss': train_loss.avg, 'train_acc': train_acc.avg,
                        'val_loss': val_loss.avg, 'val_acc': val_acc.avg,
                        'test_loss': test_loss.avg, 'test_acc': test_acc.avg,
                        }, step=epoch)
        

def train():
    # run_n_epochs()
    sweep_id = wandb.sweep(sweep_config, project="miframes5f", entity='mtuann')
    wandb.agent(sweep_id, run_n_epochs, count=10)
    
if __name__ == "__main__":
    train()

