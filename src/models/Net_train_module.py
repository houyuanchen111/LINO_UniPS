import torch
from torchmetrics import MeanMetric
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module.utils import *
from .utils import decompose_tensors
from .utils import gauss_filter
import cv2
import pytorch_lightning as pl
from src.models.utils.compute_mae import compute_mae_np,compute_mae
from datetime import datetime
from src.models.utils.utils import sobel_edge_map
from src.models.hdri_encoder.condmodel_hdri_v2 import HDRICondModel
from src.models.module.utils import *
from typing import Dict, Any

class Net(nn.Module):
    def __init__(self, pixel_samples, output,depth):
        super().__init__()
        self.target = output
        self.pixel_samples = pixel_samples
        self.depth = depth # layer numattention block
        self.glc_smoothing = True
        self.input_dim = 4 # RGB + mask   
        self.image_encoder = ScaleInvariantSpatialLightImageEncoder(self.input_dim, self.depth, use_efficient_attention=False) 
        self.mode = None #default attribute
        self.input_dim = 0 # embedding
        self.glc_upsample = GLC_Upsample(256+self.input_dim, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=True)
        self.glc_aggregation = GLC_Aggregation(256+self.input_dim, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False)
        state_dict = torch.load("/share/project/cwm/houyuan.chen/UPS_Lightning/relight_trellis_3d/decoder_ema0.9999_step0047000.pt") # load hdri encoder
        state_dict = {k[16:]:v for k,v in state_dict.items() if "hdri_cond_model." in k}
        self.img_embedding = nn.Sequential(
            nn.Linear(3,32),
            nn.LeakyReLU(),
            nn.Linear(32, 256)
        )
        self.hdri_encoder = HDRICondModel()
        self.hdri_encoder.load_state_dict(state_dict, strict=False)
        self.hdri_encoder.requires_grad_(False)
        del state_dict
        self.regressor = Regressor(384, num_enc_sab=1, use_efficient_attention=True, dim_feedforward=1024, output=self.target)
        self.criterionL2 = nn.MSELoss(reduction = 'mean')  
        self.env_feature_proj = HdriFeatureProj()
        self.env_light_head = EnvLightHead(input_feature_dim=1536, output_dim_per_item=768)
        self.point_light_align = PointLightAlign(input_feature_dim=1536, output_dim_per_item=12)
        self.area_light_align = AreaAlign(input_feature_dim=1536, output_dim_per_item=5)

    def forward(self, data, decoder_resolution, canonical_resolution):     
        
        I = data["img"].to(torch.bfloat16) # [B, 3, H, W, N]
        N = data["nml"][:, :, :, :, 0].to(torch.bfloat16)
        M = data["mask"][:, :, :, :, 0].to(torch.bfloat16)
        env_light = data["env_light"].to(torch.bfloat16)
        point_lights = data["point_lights"].to(torch.bfloat16) # b 6 12
        area_light = data["area_light"].to(torch.bfloat16) # b 6 5
        nImgArray = data["numberOfImages"].reshape(-1,1)
    
        decoder_resolution = np.int32(decoder_resolution)
        canonical_resolution = np.int32(canonical_resolution)

        """init"""
        B, C, H, W, Nmax = I.shape

        """ Image Encoder at Canonical Resolution """
        img = I.permute(0, 4, 1, 2, 3)# B Nmax C H W       
        img_index = make_index_list(Nmax, nImgArray) # Extract objects > 0
        img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4]) 
        M_enc = M.unsqueeze(1).expand(-1, Nmax, -1, -1, -1).reshape(-1, 1, H, W)
        data = img * M_enc #不concat mask, 来配合pretrain dino的三通道输入的要求
        data = data[img_index==1,:,:,:] # [B * f, 4, H, W]
        glc,light_tokens = self.image_encoder(data, nImgArray, canonical_resolution) # torch.Size([B, N, 256, H, W]) [img, mask]
        env_token = light_tokens[:,:,:,:,0,:]
        point_lights_token = light_tokens[:,:,:,:,1,:]
        area_lights_token = light_tokens[:,:,:,:,2,:]
        env_feature = torch.zeros(B,nImgArray[0],1024,768).to(torch.bfloat16).to(glc.device)
        for i in range(nImgArray[0]):
            env_feature[:,i,:,:] = self.hdri_encoder(env_light[:,i])
        env_feature_project = self.env_feature_proj(env_feature) # b f 768
        env_token_predict = self.env_light_head(env_token) # b f 768
        loss_point_lights = self.point_light_align(point_lights_token,point_lights) # b f 12
        loss_area_lights = self.area_light_align(area_lights_token,area_light) # b f 5
        loss_align_env = 1 - F.cosine_similarity(F.normalize(env_token_predict,dim=-1), F.normalize(env_feature_project,dim=-1), dim=-1).mean()
    
        """ Sample Decoder at Original Resokution"""
        img = img[img_index==1, :, :, :]
        I_dec = F.interpolate(img.float(), size=(decoder_resolution, decoder_resolution), mode='bilinear', align_corners=False).to(torch.bfloat16) # torch.Size([B, N, 3, decoder_resolution, decoder_resolution])
        N_dec = F.interpolate(N.float(), size=(decoder_resolution, decoder_resolution), mode='bilinear', align_corners=False).to(torch.bfloat16)
        M_dec = F.interpolate(M.float(), size=(decoder_resolution, decoder_resolution), mode='nearest').to(torch.bfloat16)
        Gradient_dec = sobel_edge_map(N_dec)
        del img
        del M
        _, _, H, W = I_dec.shape         
    
        if self.glc_smoothing:  
            f_scale = decoder_resolution//canonical_resolution # (2048/256)
            smoothing = gauss_filter.gauss_filter(glc.shape[1], 10 * f_scale+1, 1).to(glc.device) # channels, kernel_size, sigma
            glc = smoothing(glc) #[B*f,256,128,128]
        p = 0
        ids_batch = []

        n_true_list = [] 
        o_ids_list = []
        glc_ids_list = []
        gradient_ids_list = []

        for b in range(B): #   
            target = range(p, p+nImgArray[b])
            p = p+nImgArray[b]
            m_ = M_dec[b, :, :, :].reshape(-1, H * W).permute(1,0)        
            ids = np.nonzero(m_>0)[:,0]  
            ids = ids[np.random.permutation(len(ids))]
            idset = [ids[:self.pixel_samples]]   
            o_ = I_dec[target, :, :, :].reshape(nImgArray[b], C, H * W).permute(2,0,1)  # [N, c, h, w]]
            n_true = F.normalize(N_dec[b, :, :, :].reshape(3, H * W).permute(1,0), p=2, dim=-1).to(torch.bfloat16) 
            gradient_n = Gradient_dec[b, :, :, :].reshape(1, H * W).permute(1,0).to(torch.bfloat16) # [H*W, 1]
            for ids in idset: 
                o_ids = o_[ids, :, :]
                glc_ids = glc[target, :, :, :].permute(2,3,0,1).flatten(0,1)[ids,:,:] # 不用grid_sample 
                o_ids_list.append(o_ids)
                glc_ids_list.append(glc_ids)
                n_true_list.append(n_true[ids, :])
                gradient_ids_list.append(gradient_n[ids, :])                 
            ids_batch.append(ids)
        o_ids = torch.cat(o_ids_list, dim=0) #[B*2048,f,3]
        glc_ids = torch.cat(glc_ids_list, dim=0) #[B*2048,f,256]
        n_true = torch.stack(n_true_list, dim=0) #[B, 2048,f,3]
        gradient_ids = torch.stack(gradient_ids_list, dim=0)
        o_ids = self.img_embedding(o_ids) # [B*2048,f,256]
        x = o_ids + glc_ids
        glc_ids = self.glc_upsample(x)
        x = o_ids + glc_ids
        x = self.glc_aggregation(x)  #[B*2048, 384] #完全没有关系
        x_n, _, _, conf = self.regressor(x, len(ids_batch[0])) # [B, 2048, 3]
        x_n = F.normalize(x_n, p=2, dim=-1)
        mse = self.criterionL2(x_n, n_true)
        loss_gradient = self.criterionL2(conf.exp(), gradient_ids.exp())*3 
        loss_conf = (((x_n - n_true)**2) * (1 + conf)).mean()
        if torch.isnan(loss_conf).any():
            print("loss_conf contains NaN values")
            loss_conf = torch.zeros_like(loss_conf).to(x_n.device) + 1e-6
        if torch.isnan(loss_gradient).any():
            print("loss_gradient contains NaN values")
            loss_gradient = torch.zeros_like(loss_gradient).to(x_n.device) + 1e-6
        if torch.isnan(mse).any():
            print("mse contains NaN values")
            mse = torch.zeros_like(mse).to(x_n.device) + 1e-6
        if torch.isnan(loss_align_env).any():
            print("loss_align_env contains NaN values")
            loss_align_env = torch.zeros_like(loss_align_env).to(x_n.device) + 1e-6
        if torch.isnan(loss_point_lights).any():
            print("loss_point_lights contains NaN values")
            loss_point_lights = torch.zeros_like(loss_point_lights).to(x_n.device) + 1e-6
        if torch.isnan(loss_area_lights).any():
            print("loss_area_lights contains NaN values")
            loss_area_lights = torch.zeros_like(loss_area_lights).to(x_n.device) + 1e-6
        loss = loss_conf  + 0.1* loss_gradient/(loss_gradient / loss_conf).detach() + 0.1*loss_align_env/ (loss_align_env / loss_conf).detach() + 0.1*loss_point_lights / (loss_point_lights / loss_conf).detach() +0.1* loss_area_lights/ (loss_area_lights / loss_conf).detach()
        return {
            'mse': mse,
            'loss': loss,
            "loss_gradient": loss_gradient,
            "loss_conf": loss_conf,
            "loss_align_env": loss_align_env,
            "loss_point_lights": loss_point_lights,
            "loss_area_lights": loss_area_lights,
        }

class LINO_UniPSModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer_class,
        scheduler_class,
        canonical_resolution: int,
        sample_num: int,
        save_dir: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        max_epochs: int = 100,
        min_lr: float = 1e-6,
        step_size: int = 10,
        gamma: float = 0.8,
    ) -> None:
        super().__init__()
 
        self.strict_loading = False
        self.save_hyperparameters(logger=False)
        self.canonical_resolution = canonical_resolution
        self.net = net
        self.sample_num = sample_num
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.step_size = step_size
        self.gamma = gamma
        # metric
        self.criterion = torch.nn.MSELoss(reduction='mean') 
        self.save_dir = save_dir
        self.train_mae = MeanMetric()
        self.val_mae = MeanMetric()
        self.test_mae = MeanMetric()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, data,decoder_resolution,canonical_resolution,):
        return self.net(data,decoder_resolution,canonical_resolution)
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
    def model_step(
        self, batch
    ) :
        img = batch["img"]
        B,C,H,W,N = img.shape
        if self.net.mode !="Test":
            metric_dict= self.forward(data = batch,decoder_resolution=H,canonical_resolution=self.canonical_resolution)
            return metric_dict
        else:
            nout= self.forward(data = batch,decoder_resolution=H,canonical_resolution=self.canonical_resolution)
            return nout
    def training_step(
            self, batch, batch_idx: int
        ) -> torch.Tensor:
        
        metric_dict= self.model_step(batch)  # input is correct
        assert len(metric_dict) == 7, "metric_dict should have 7 keys"
        mse, loss = metric_dict['mse'], metric_dict['loss']
        self.loss = loss 
        self.train_loss(loss) 
        self.lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", self.lr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss(loss), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse_loss", mse, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        
        metric_dict= self.model_step(batch)  # input is correct
        assert len(metric_dict) == 7, "metric_dict should have 7 keys"
        mse, loss = metric_dict['mse'], metric_dict['loss']
        self.loss = loss
        self.log("val/loss", self.loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse_loss", mse, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
       
        optimizer = self.optimizer_class(
            self.trainer.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        if self.scheduler_class is not None:
            if self.scheduler_class.__name__ == 'StepLR':
                scheduler = self.scheduler_class(
                    optimizer=optimizer,
                    step_size=self.step_size,
                    gamma=self.gamma
                )
            else:
                scheduler = self.scheduler_class(
                    optimizer=optimizer,
                    T_max=self.max_epochs,
                    eta_min=self.min_lr
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
