#%%
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import os
import sys
import argparse
from dataloader import OrientationsWithSymmDataset, SimpleDataSplitDataModule
from transformer.model import OwnModel
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from Quaternion import Quaternion

def calculate_angualr(x, y):
        # Norm = torch.norm(x, p=2, dim=2)
        # Norm = Norm.unsqueeze(-1).repeat(1,1,4).cuda()
        # x = x / Norm
        x = x.cuda()
        y = y.cuda()
        a = torch.sum(x*y, dim=1).unsqueeze(-1)
        b = torch.sum(-x*y, dim=1).unsqueeze(-1)
        c = torch.cat([a,b],dim=1)
        # c = torch.abs(torch.sum(x*y, dim=2))
        product = torch.acos(c)
        # product = torch.where(torch.isnan(product), torch.full_like(product,1000000),product)
        product, idx = torch.min(product, dim=1)
        # loss = pi - torch.abs(pi - torch.abs(product))
        loss = product
        return loss


class Quer_loss_prob_5class(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, pred_n, n):
        sym_qua_all = np.load(
            '/home/lei/Desktop/project-isas/code/Transformer_3D_Object_Tless/3D_Transformer2_mlp_schaetzung360/symsol_symmetries_tless360class.npz')
        loss_tensor = torch.zeros((y.shape[0], 1))
        label_near_tensor = torch.zeros((y.shape[0], 4))
        x_near_tensor = torch.zeros((y.shape[0], 4))
        for i in range(len(y)):
            label = y[i, :]
            pred = x[i, :]
            sym_class = n[i]
            if sym_class == 0:
                sym_sample = sym_qua_all['one_syms_qua']
            elif sym_class == 1:
                sym_sample = sym_qua_all['second_syms_qua']
            elif sym_class == 2:
                sym_sample = sym_qua_all['third_syms_qua']
            elif sym_class == 3:
                sym_sample = sym_qua_all['fourth_syms_qua']
            elif sym_class == 4:
                sym_sample = sym_qua_all['fifth_syms_qua']

            sym_sample = torch.tensor(sym_sample).cuda()  # n * 4
            label = label.repeat(sym_sample.shape[0], 1, ).cuda()  # n * 4
            pred = pred.repeat(sym_sample.shape[0], 1, ).cuda()  # n * 4
            pos_qua = (self.qua_mul(label, sym_sample)).cuda()
            loss_tensor[i, :], idx = self.calculate_angualr(pred, pos_qua)
            pos_qua = torch.cat([pos_qua,pos_qua],dim=0)
            label_near_tensor[i,:] = pos_qua[idx,:]
            x_double= torch.cat([pred,-pred],dim=0)
            x_near_tensor[i,:] = x_double[idx,:]
        return loss_tensor, x_near_tensor, label_near_tensor

    def calculate_angualr(self, x, y):
        Norm = torch.norm(x, p=2, dim=1)
        Norm = Norm.unsqueeze(-1).repeat(1, 4).cuda()
        x = x / Norm
        a = torch.sum(x * y, dim=1).unsqueeze(-1)
        b = torch.sum(-x * y, dim=1).unsqueeze(-1)
        c = torch.cat([a, b], dim=0)

        angular = torch.acos(c)
        angular, idx = torch.min(angular, dim=0)
        loss = angular
        return loss, idx

    def qua_mul(self, p, q):
        result = torch.zeros(q.shape)
        result[:, 3] = q[:, 3] * p[:, 3] - q[:, 0] * p[:, 0] - q[:, 1] * p[:, 1] - q[:, 2] * p[:, 2]
        result[:, 0] = q[:, 3] * p[:, 0] + q[:, 0] * p[:, 3] + q[:, 1] * p[:, 2] - q[:, 2] * p[:, 1]
        result[:, 1] = q[:, 3] * p[:, 1] - q[:, 0] * p[:, 2] + q[:, 1] * p[:, 3] + q[:, 2] * p[:, 0]
        result[:, 2] = q[:, 3] * p[:, 2] + q[:, 0] * p[:, 1] - q[:, 1] * p[:, 0] + q[:, 2] * p[:, 3]
        return result

def prediction(model, image, label):
    image = image.cuda()
    h = image.shape[2]
    w = image.shape[3]
    out_list = torch.zeros(image.shape[0], 4).cuda()
    n_models = torch.zeros(image.shape[0]).cuda()
    label = label.cuda()
    for n in range(len(image)):
        im = image[n, :, :, :].unsqueeze(0).cuda()
        lab = label[n,:].cuda()
        out, n_model = model(im)
        n_model = n_model.cpu().detach().numpy()
        n_model = np.argmax(n_model)
        out_list[n,:] = out
        n_models[n] = n_model
    return out_list, n_models 

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    #image = image.squeeze(0)  
    #image = unloader(image)
    plt.imshow(image)
    if title is not None:
       plt.title(title)
    plt.pause(0.001) 

performance_params = {
    "data_path": '/home/lei/Desktop/project-isas/data/Tless_360_dataset',
    "checkpoint_file": "/home/lei/Desktop/project-isas/code/Transformer_3D_Object_Tless/3D_Transformer2_mlp_schaetzung360/lightning_logs/tansformer_3D_tless360/version_2/checkpoints/epoch=158-step=24167.ckpt"
    # "checkpoint_file": '/home/prak2/Project2/Transformer_3D_Object/3D_Transformer2_mlp_schaetzung_exp3/lightning_logs/tansformer_3D_cyl/version_72/checkpoints/epoch=94-step=949.ckpt'
}

torch.autograd.set_detect_anomaly(True)

if 'init_modules' in globals(  ):
    # second or subsequent run: remove all but initially loaded modules
    for m in sys.modules.keys(  ):
        if m not in init_modules:
            del(sys.modules[m])
else:
    # first run: find out which modules were initially loaded
    init_modules = sys.modules.keys(  )

# load model
current_network = OwnModel.load_from_checkpoint(performance_params['checkpoint_file'])
current_network.cuda()
current_network.eval()

# load dataset
dataset = OrientationsWithSymmDataset(csv_file=os.path.join(performance_params['data_path'],"Tless_object.csv"),root_dir=performance_params['data_path'], transform=current_network.data_transforms())
loaderTmp = DataLoader(dataset, batch_size=dataset.__len__())
all_items, _, _= next(iter(loaderTmp))
mean = all_items.mean()
std = all_items.std()
dataset.set_mean_and_std(mean,std)

dataloader = DataLoader(dataset,
                        batch_size=10,
                        shuffle=True,
                        drop_last=True)


error1 = 0
error2 = 0
errors = []
idx1 = 10
idx2 = 1
MSE_loss = nn.MSELoss(reduction='none')
mse_loss = 0
i_batch = 0
for d, data in enumerate(dataloader):
    im, label, n_model = data
    n_model = n_model.cuda()
    batch = im.shape[0]
    label = label.cuda()
    with torch.no_grad():
        # pred, dec_enc_attn = prediction(current_network, im, label)
        pred_qua, pred_n_model = prediction(current_network, im, label)

    metrics2 = torch.zeros((pred_qua.shape[0], 1)).cuda()

    loss = Quer_loss_prob_5class()
    # loss = Quer_loss_prob_5class_without()
    metrics2, pred_qua_norm, symmetrie_label_tensor = loss(pred_qua, label, pred_n_model, n_model)
    # metrics2 = calculate_angualr(symmetrie_label_tensor, pred_qua)
    # metrics2 += 2*pi * (pred_n_model != n_model)

    error2 += torch.mean(metrics2)
    i_batch += 1


    if d < idx1:
        print("Sample")
        imshow(im[idx2])
        print('Quaternion and class of shape')
        print('Label:')
        print('Quaternion:', np.around(label[idx2,:].cpu().numpy(), 4),  'Category of Object:', int(n_model[idx2].cpu().numpy()))
        print('Prediction')
        print('Quaternion:', np.around(pred_qua[idx2,:].cpu().numpy(), 4),  'Category of Object:', int(pred_n_model[idx2].cpu().numpy()))
        print('symmetry Quaternion')
        print('Label after symmetrical rotation')
        print('Quaternion:', np.around(symmetrie_label_tensor[idx2].cpu().numpy(), 4))
        print('Prediction')
        print('Quaternion:', np.around(pred_qua_norm[idx2,:].cpu().numpy(), 4))
        # print(metrics1[idx2,:])
        print('Angle between two Quaternion:', np.around(metrics2[idx2].cpu().numpy(), 4))


error2 = error2 / i_batch
print("Evaluation result:")
print('metric2')
print( np.around(error2.cpu().numpy(), 4))

# %%
