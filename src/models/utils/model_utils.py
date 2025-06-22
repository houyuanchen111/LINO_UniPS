import os
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import matplotlib.pyplot as plt
def pca_show(feats):
    feats = feats.cpu().detach()
    B, C, H, W = feats.shape
    n_components = 3
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)      
    feats = F.interpolate(feats, size=(128, 128), mode='bilinear', align_corners=False)  
    features = feats[0, :, :, :].squeeze().reshape(C,-1).permute(1,0).cpu()
    pca.fit(features)
    pca_features = pca.transform(features)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    pca_features = pca_features.reshape(128, 128, n_components).astype(np.uint8)
    plt.imshow(pca_features)


def loadmodel(model, filename, strict=True, remove_prefix=False):
    if os.path.exists(filename):
        params = torch.load('%s' % filename)
        if remove_prefix:
            new_params = {k.replace('module.', ''): v for k, v in params.items()}
            model.load_state_dict(new_params,strict=strict)
            print("prefix successfully removed")
        else:
            model.load_state_dict(params,strict=strict)
        print('Load %s' % filename)
    else:
        print('Model Not Found')
    return model

def loadoptimizer(optimizer, filename):
    if os.path.exists(filename):
        params = torch.load('%s' % filename)
        optimizer.load_state_dict(params)
        print('Load %s' % filename)
    return optimizer

def loadscheduler(scheduler, filename):
    if os.path.exists(filename):
        params = torch.load('%s' % filename)
        scheduler.load_state_dict(params)
        print('Load %s' % filename)
    else:
        print('Scheduler Not Found')
    return scheduler

def savemodel(model, filename):
    print('Save %s' % filename)
    torch.save(model.state_dict(), filename)

def saveoptimizer(optimizer, filename):
    print('Save %s' % filename)
    torch.save(optimizer.state_dict(), filename)

def savescheduler(scheduler, filename):
    print('Save %s' % filename)
    torch.save(scheduler.state_dict(), filename)

def optimizer_setup_Adam(net, lr = 0.0001, init=True, step_size=3, stype='step'):
    print(f'Optimizer (Adam) lr={lr}')
    if init==True:
        net.init_weights()
    net = torch.nn.DataParallel(net)
    optim_params = [{'params': net.parameters(), 'lr': lr},] # confirmed
    optimizer = torch.optim.Adam(optim_params, betas=(0.9, 0.999), weight_decay=0)
    if stype == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=0, last_epoch=-1)
        print('Cosine aneealing learning late scheduler')
    if stype == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8)
        print(f'Step late scheduler x0.8 decay every {step_size}')
    return net, optimizer, scheduler

def optimizer_setup_SGD(net, lr = 0.01, momentum= 0.9, init=True):
    print(f'Optimizer (SGD with momentum) lr={lr}')
    if init==True:
        net.init_weights()
    net = torch.nn.DataParallel(net)
    optim_params = [{'params': net.parameters(), 'lr': lr},] # confirmed
    return net, torch.optim.SGD(optim_params, momentum=momentum, weight_decay=1e-4, nesterov=True)

def optimizer_setup_AdamW(net, lr = 0.001, eps=1.0e-8, step_size = 20, init=True, stype='step', use_data_parallel=False):
    print(f'Optimizer (AdamW) lr={lr}')
    if init==True:
        net.init_weights()
    if use_data_parallel:
        net = torch.nn.DataParallel(net)
    optim_params = [{'params': net.parameters(), 'lr': lr},] # confirmed
    optimizer = torch.optim.AdamW(optim_params, betas=(0.9, 0.999), eps=eps, weight_decay=0.01)
        
    if stype=='cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=0, last_epoch=-1)
        print('Cosine aneealing learning late scheduler')
    if stype == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8)
        print(f'Step late scheduler x0.8 decay every {step_size}')
    return net, optimizer, scheduler

def mode_change(net, Training):
    if Training == True:
        for param in net.parameters():
            param.requires_grad = True
        net.train()
    if Training == False:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records,
            'args': args}
    torch.save(state, os.path.join(save_path, 'checkp_%d.pth.tar' % (epoch)))
    torch.save(records, os.path.join(save_path, 'checkp_%d_rec.pth.tar' % (epoch)))



def masking(img, mask):
    # img [B, C, H, W]
    # mask [B, 1, H, W] [0,1]
    img_masked = img * mask.expand((-1, img.shape[1], -1, -1))
    return img_masked

def print_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])


    print('# parameters: %d' % params)


def angular_error(x1, x2, mask = None): # tensor [B, 3, H, W]

    if mask is not None:
        dot = torch.sum(x1 * x2 * mask, dim=1, keepdim=True)
        dot = torch.max(torch.min(dot, torch.Tensor([1.0-1.0e-12])), torch.Tensor([-1.0+1.0e-12]))
        emap = torch.abs(180 * torch.acos(dot)/np.pi) * mask
        mae = torch.sum(emap) / torch.sum(mask)
        return mae, emap
    if mask is None:
        dot = torch.sum(x1 * x2, dim=1, keepdim=True)
        dot = torch.max(torch.min(dot, torch.Tensor([1.0-1.0e-12])), torch.Tensor([-1.0+1.0e-12]))
        error = torch.abs(180 * torch.acos(dot)/np.pi)
        return error

def write_errors(filepath, error, trainid, numimg, objname = []):
    dt_now = datetime.datetime.now()
    print(filepath)

    if len(objname) > 0:
        with open(filepath, 'a') as f:
            f.write('%s %03d %s %02d %.2f\n' % (dt_now, numimg, objname, trainid, error))
    else:
        with open(filepath, 'a') as f:
            f.write('%s %03d %02d %.2f\n' % (dt_now, numimg, trainid, error))


def save_nparray_as_hdf5(self, a, filename):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dataset_1', data=a)
    h5f.close()

def freeze_params(net):
    for param in net.parameters():
        param.requires_grad = False

def unfreeze_params(net):
    for param in net.parameters():
        param.requires_grad = True

def make_index_list(maxNumImages, numImageList):
    index = np.zeros((len(numImageList) * maxNumImages), np.int32)
    for k in range(len(numImageList)):
        index[maxNumImages*k:maxNumImages*k+numImageList[k]] = 1
    return index
    
