import torch
import torchvision
import torch.nn as nn

def get_model_and_optim(config, device=None):
    # print(get_model_and_optim, config)
    architecture = config.architecture
    out_features = config.out_features
    
    if architecture == 'mc3_18':
        model = torchvision.models.video.mc3_18(pretrained=True)
    elif architecture == 'r3d_18':
        model = torchvision.models.video.r3d_18(pretrained=True)
    elif architecture == 'r2plus1d_18':
        model = torchvision.models.video.r2plus1d_18(pretrained=True)
        
    model.fc = torch.nn.Linear(in_features=512, out_features=out_features, bias=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=1e-4)
    criterion = nn.BCELoss()
    return model, optimizer, criterion