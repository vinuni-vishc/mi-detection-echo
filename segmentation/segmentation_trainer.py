import os
import tqdm
import torch
import wandb
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader
from datasets.segmentation_dataset import LVWallDataset
from segmentation_utils import *
from utils.videos import AverageMeter
# os.environ['WANDB_MODE'] = 'offline'

sweep_config = {
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'val_iou'},
    'parameters': {
        'encoder': {
            'values': [ 'mobilenet_v2', 'resnet18', 'efficientnet-b0', 'resnet34']
        },
        'architecture': {
            'values': ['DeepLabV3', 'UnetPlusPlus', 'FPN', 'Linknet', 'PAN', 'Unet']
        },
        'fold': {
            'values': [0, 1, 2, 3, 4], # 0
        },
        'epoch': {
            'values': [50],
        },
        'batch_size': {
            'values': [32],
        },
        'img_shape': {
            'values': [(224, 224)],
        }
    }
}


        
def train_one_epoch(epoch, model, optimizer, train_loader, device):
    model.train()
    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    loss_all, iou_all = AverageMeter(), AverageMeter()

    for step, batch in pbar:    
        
        img, ytrue = batch[0].to(device).float(), batch[1].to(device).float()
        ypred = torch.sigmoid(model(img))

        # print("MIN-MAX", ypred.min(), ypred.max(), ytrue.min(), ytrue.max())
        loss = get_loss(ypred, ytrue, batch)
        iou = get_iou(ypred, ytrue)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # metrics.append(get_metric(loss, iou, img.shape[0]))
        # evaluation = get_mean(metrics)
        loss_all.update(loss.item(), img.shape[0])
        iou_all.update(iou.item(), img.shape[0])
        
        
        pbar.set_description(f"train epoch {epoch + 1} batch {step + 1} / {len(pbar)} train-loss {loss_all.avg:.4f} train-iou {iou_all.avg:.4f}") 
    # return evaluation, metrics
    return loss_all, iou_all

def valid_one_epoch(epoch, model, val_loader, device):
    '''test the model for one epoch with progress bar'''
    model.eval()
    pbar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
    # metrics = []
    loss_all, iou_all = AverageMeter(), AverageMeter()
    
    
    for step, batch in pbar:
        img, ytrue = batch[0].to(device).float(), batch[1].to(device).float()
        
        ypred = torch.sigmoid(model(img))
        # print(ytrue.sum())
        loss = get_loss(ypred, ytrue, batch)
        iou = get_iou(ypred, ytrue)
                
        # metrics.append(get_metric(loss, iou, img.shape[0]))
        # evaluation = get_mean(metrics)
        loss_all.update(loss.item(), img.shape[0])
        iou_all.update(iou.item(), img.shape[0])
        
        pbar.set_description(f"test epoch {epoch + 1} batch {step + 1} / {len(pbar)} test-loss {loss_all.avg:.4f} test-iou {iou_all.avg:.4f}")
    
    return loss_all, iou_all

def remove_model_redundant(list_models):
    ff = [1] * len(list_models)
    for idv1 in range(len(list_models)):
        for idv2 in range(idv1 + 1, len(list_models)):
            if list_models[idv1][1] <= list_models[idv2][1] and  list_models[idv1][2] <= list_models[idv2][2]:
                ff[idv1] = 0
    for idf1, f1 in enumerate(ff):
        if f1:
            pass
        else:
            fpa = list_models[idf1][0]
            if os.path.isfile(fpa):
                cmd = f"rm -rf '{fpa}' "
                os.system(cmd)
                print("remove model: {}".format(cmd))
                
def run_n_epochs(config=None):
    # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=3vqGZbxxulea
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # sweep_config = {
    #     'batch_size': 64,
    #     'architecture': 'DeepLabV3',
    #     'epoch': 50,
    #     'fold': 0,
    #     'encoder': 'mobilenet_v2',
    #     'img_shape': (224, 224),
    # }                 
    # config = EasyDict(sweep_config)
    # if True:
    with wandb.init(config=config) as run:
        config = wandb.config
    
        run.name = f'{config.architecture}__{config.encoder}__{config.epoch}__{config.fold}'
        wandb.name = f'{config.architecture}__{config.encoder}__{config.epoch}__{config.fold}'
        print("Wandb name: {}".format(wandb.name))
        
        print(f"Start wandb with config: {config}")
        folder_exp = f'/home/vishc2/tuannm/echo/vishc-echo/models/segment_ckpt_5folds/{config.architecture}__{config.encoder}__{config.epoch:03d}__{config.fold:03d}'
        
        train_set = LVWallDataset(split='train', id_fold=config.fold, img_size=config.img_shape)
        val_set = LVWallDataset(split='test', id_fold=config.fold, img_size=config.img_shape)
        
        train_loader = DataLoader(train_set, config.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, config.batch_size, shuffle=False)
        
        model, optimizer = get_model_and_optim(config, device=device)
        
        list_models = []
        
        for epoch in range(config.epoch):
            train_loss, train_iou = train_one_epoch(epoch, model, optimizer, train_loader, device)
            test_loss, test_iou = valid_one_epoch(epoch, model, val_loader, device)
            
            # if test_iou.avg >= 0.75:
            fn_md = f'{folder_exp}__{epoch:03d}__{train_iou.avg:.4f}__{test_iou.avg:.4f}.pth'
            os.makedirs(os.path.dirname(fn_md), exist_ok=True)
            torch.save(model.state_dict(), fn_md)
                
            list_models.append([fn_md, train_iou.avg, test_iou.avg])
                
            wandb.log({'epoch': epoch, 'train_loss': train_loss.avg, 'train_iou': train_iou.avg,
                        'val_loss': test_loss.avg, 'val_iou': test_iou.avg,
                        # 'train_val_loss': evaluation.loss, 'train_val_iou': evaluation.iou,
                        }, step=epoch)
            
        remove_model_redundant(list_models)


def train():
    # run_n_epochs()
    sweep_id = wandb.sweep(sweep_config, project="lvwseg5f", entity='mtuann')
    wandb.agent(sweep_id, run_n_epochs, count=120)
    
if __name__ == "__main__":
    train()

