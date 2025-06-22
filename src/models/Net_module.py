import torchvision
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
from src.models.utils.compute_mae import compute_mae_np
from datetime import datetime

class LiNo_UniPS(pl.LightningModule):
    def __init__(self, 
                 pixel_samples: int = 2048,
                 task_name :str = None):
        super().__init__()
        self.pixel_samples = pixel_samples
        self.task_name = task_name
        self.input_dim = 4 
        self.image_encoder = ScaleInvariantSpatialLightImageEncoder(self.input_dim, use_efficient_attention=False) 
        self.input_dim = 0 
        self.glc_upsample = GLC_Upsample(256+self.input_dim, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=True)
        self.glc_aggregation = GLC_Aggregation(256+self.input_dim, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False)
        self.img_embedding = nn.Sequential(
            nn.Linear(3,32),
            nn.LeakyReLU(),
            nn.Linear(32, 256)
        )
        self.regressor = Regressor(384, num_enc_sab=1, use_efficient_attention=True, dim_feedforward=1024)
        self.test_mae = MeanMetric()
        self.test_loss = MeanMetric()
    def on_test_start(self):
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_save_dir = f'output/{timestamp}/{self.task_name}/results/'
        os.makedirs(self.run_save_dir, exist_ok=True)
        
    def from_pretrained(self,pth_path):
        pretrain_weight = torch.load(pth_path,weights_only=False)
        self.load_state_dict(pretrain_weight, strict=False)
    def _prepare_test_inputs(self, batch):
        img = batch["imgs"].to(torch.bfloat16)
        self.numberofImages = img.shape[-1]
        print("number of test images", self.numberofImages)
        nml = batch["nml"].to(torch.bfloat16)
        directlist = batch["directlist"]
        roi = batch.get("roi",None)
        roi = roi[0].cpu().numpy()
        return img, nml,directlist,roi 
    def _postprocess_prediction(self, nml_predict_raw, nml_gt_raw, roi):
       
        h_orig, w_orig, r_s, r_e, c_s, c_e = roi
        nml_predict = nml_predict_raw.squeeze().permute(1, 2, 0).cpu().numpy()
        nml_predict = cv2.resize(nml_predict, dsize=(c_e - c_s, r_e - r_s), interpolation=cv2.INTER_AREA)
        mask = np.float32(np.abs(1 - np.sqrt(np.sum(nml_predict * nml_predict, axis=2))) < 0.5)
        nml_predict = np.divide(nml_predict, np.linalg.norm(nml_predict, axis=2, keepdims=True) + 1e-12)
        nml_predict = nml_predict * mask[:, :, np.newaxis]
        nout = np.zeros((h_orig, w_orig, 3), np.float32)
        nout[r_s:r_e, c_s:c_e, :] = nml_predict

        nml_gt = nml_gt_raw.squeeze().permute(1, 2, 0).float().cpu().numpy()
        mask_gt = np.float32(np.abs(1 - np.sqrt(np.sum(nml_gt * nml_gt, axis=2))) < 0.5)
        
        return nout, nml_gt, mask_gt

    def _calculate_and_log_metrics(self, nout, nml_gt, mask_gt):
        mse = torch.nn.MSELoss()(torch.tensor(nout).to(self.device), torch.tensor(nml_gt).to(self.device))
        
        mae, emap = compute_mae_np(nout, nml_gt, mask_gt)

        self.test_loss(mse)
        self.test_mae(mae)
        self.log("test/mse", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)
        
        return mse, mae, emap

    def _save_test_results(self, nout, nml_gt, emap, img, loss, mae, directlist, save_dir):
       
        obj_name_parts = os.path.dirname(directlist[0][0]).split('/')
        obj_name = obj_name_parts[-1]
       
     
        save_path = os.path.join(save_dir,f'{self.numberofImages}',f'{obj_name}')
        os.makedirs(save_path, exist_ok=True)
        print(f"save to: {save_path}")
        if ("DiLiGenT_100" not in self.task_name) and ("Real" not in self.task_name):
            nout_to_save = (nout + 1) / 2
            nml_gt_to_save = (nml_gt + 1) / 2
            
            emap_to_save = emap.astype(np.float32).squeeze()
            thresh = 90
            emap_to_save[emap_to_save >= thresh] = thresh
            emap_to_save = emap_to_save / thresh

            plt.imsave(save_path + '/nml_predict.png', np.clip(nout_to_save, 0, 1))
            plt.imsave(save_path + '/nml_gt.png', np.clip(nml_gt_to_save, 0, 1))
            plt.imsave(save_path + '/error_map.png', emap_to_save, cmap='jet')
            torchvision.utils.save_image(img.squeeze(0).permute(3,0,1,2), save_path + '/tiled.png')

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(np.clip(nout_to_save, 0, 1))
            axes[0].set_title('Prediction'); axes[0].axis('off')
            axes[1].imshow(np.clip(nml_gt_to_save, 0, 1))
            axes[1].set_title('Ground Truth'); axes[1].axis('off')
            axes[2].imshow(emap, cmap='jet')
            axes[2].set_title('Error Map'); axes[2].axis('off')
            plt.figtext(0.5, 0.02, f'Loss: {loss:.4f} | MAE: {mae:.4f}', ha='center', fontsize=12)
            plt.tight_layout()
            plt.savefig(save_path + '/combined.png', dpi=300)
            plt.close(fig) 

        
            with open(save_path + '/result.txt', 'w') as f:
                f.write(f"loss: {loss.item()}\n")
                f.write(f"mae: {mae}\n")
                
            print(f"Done for {obj_name}")
        else:
            if "DiLiGenT_100" in self.task_name:
                from scipy.io import savemat
                mat_save_path = os.path.join(os.path.dirname(save_path),"submit")
                os.makedirs(mat_save_path,exist_ok=True)
                normal_map = nout
                savemat(mat_save_path + "/" + obj_name + '.mat',  {'Normal_est': normal_map})
            torchvision.utils.save_image(img.squeeze(0).permute(3,0,1,2), save_path + '/tiled.png')
            nout = (nout + 1) / 2 
            plt.imsave(save_path + '/nml_predict.png', nout)

    def test_step(self, batch, batch_idx):
        
        img, nml_gt, directlist, roi = self._prepare_test_inputs(batch)
        
        nml_predict = self.model_step(batch)

      
        nout, nml_gt, mask_gt = self._postprocess_prediction(nml_predict, nml_gt, roi)
        if ("DiLiGenT_100" not in self.task_name) and ("Real" not in self.task_name):
            loss, mae, emap = self._calculate_and_log_metrics(nout, nml_gt, mask_gt)
            print(f"{os.path.basename(os.path.dirname(directlist[0][0]))} | MAE: {mae:.4f}")
            self._save_test_results(nout, nml_gt, emap, img, loss, mae, directlist, self.run_save_dir)
        else:
            emap,loss,mae = None,None,None
            self._save_test_results(nout, nml_gt, emap, img, loss, mae, directlist, self.run_save_dir)

    def predict_step(self,batch):
        roi = batch.get("roi",None)
        nml_predict = self.model_step(batch)
        roi = roi[0].cpu().numpy()
        h_ = roi[0] 
        w_ = roi[1] 
        r_s = roi[2]
        r_e = roi[3]
        c_s = roi[4]
        c_e = roi[5]
        nml_predict = nml_predict.squeeze().permute(1,2,0).cpu().numpy()
        nml_predict = cv2.resize(nml_predict, dsize=(c_e-c_s, r_e-r_s), interpolation=cv2.INTER_AREA)
        nml_predict = np.divide(nml_predict, np.linalg.norm(nml_predict, axis=2, keepdims=True) + 1.0e-12)
        mask = np.float32(np.abs(1 - np.sqrt(np.sum(nml_predict * nml_predict, axis=2))) < 0.5)
        nml_predict = nml_predict * mask[:, :, np.newaxis] 
        nout = np.zeros((h_, w_, 3), np.float32)
        nout[r_s:r_e, c_s:c_e,:] = nml_predict
        mask = batch["mask_original"].squeeze().cpu().numpy()[:,:,None]

        return nout*mask
    def model_step(self,batch):
        I = batch.get("imgs",None)
        M = batch.get("mask",None)
        # roi = batch.get("roi",None)
        B, C, H, W, Nmax = I.shape

        patch_size = 512               
        patches_I = decompose_tensors.divide_tensor_spatial(I.permute(0,4,1,2,3).reshape(-1, C, H, W), block_size=patch_size, method='tile_stride')
        patches_I = patches_I.reshape(B, Nmax, -1, C, patch_size, patch_size).permute(0, 2, 3, 4, 5, 1)
        sliding_blocks = patches_I.shape[1]
        patches_M = decompose_tensors.divide_tensor_spatial(M, block_size=patch_size, method='tile_stride')
        patches_nml = []

        nImgArray = np.array([Nmax])
        canonical_resolution = 256
        for k in range(sliding_blocks):
            """ Image Encoder at Canonical Resolution """
            print("please wait for a moment, it may take a while")
            I = patches_I[:, k, :, :, :, :] 
            M = patches_M[:, k, :, :, :] 
            B, C, H, W, Nmax = I.shape
            decoder_resolution = H
            I_enc = I.permute(0, 4, 1, 2, 3)
            M_enc = M 
            img_index = make_index_list(Nmax, nImgArray) 
            I_enc = I_enc.reshape(-1, I_enc.shape[2], I_enc.shape[3], I_enc.shape[4]) 
            M_enc = M_enc.unsqueeze(1).expand(-1, Nmax, -1, -1, -1).reshape(-1, 1, H, W) 
            data = I_enc * M_enc 
            data = data[img_index==1,:,:,:] 
            glc,_= self.image_encoder(data, nImgArray, canonical_resolution)
            I_dec = []
            M_dec = []
            img = I.permute(0, 4, 1, 2, 3)            
            """ Sample Decoder at Original Resokution"""
            img = img.squeeze()
            I_dec = F.interpolate(img.float(), size=(decoder_resolution, decoder_resolution), mode='bilinear', align_corners=False).to(torch.bfloat16) 
            M_dec = F.interpolate(M.float(), size=(decoder_resolution, decoder_resolution), mode='nearest').to(torch.bfloat16)
            decoder_imgsize = (decoder_resolution, decoder_resolution)
            C = img.shape[1]
            H = decoder_imgsize[0]
            W = decoder_imgsize[1]     
            nout = torch.zeros(B, H * W, 3).to(I.device)
            f_scale = decoder_resolution//canonical_resolution 
            smoothing = gauss_filter.gauss_filter(glc.shape[1], 10 * f_scale+1, 1).to(glc.device)
            chunk_size = 16
            processed_chunks = []
            for glc_chunk in torch.split(glc, chunk_size, dim=0):
                smoothed_chunk = smoothing(glc_chunk)
                processed_chunks.append(smoothed_chunk)
            glc = torch.cat(processed_chunks, dim=0) 
            del M
            _, _, H, W = I_dec.shape         
            p = 0
            nout = torch.zeros(B, H * W, 3).to(I.device)
            conf_out = torch.zeros(B, H * W, 1).to(I.device)
            for b in range(B):
                target = range(p, p+nImgArray[b])
                p = p+nImgArray[b]
                m_ = M_dec[b, :, :, :].reshape(-1, H * W).permute(1,0)        
                ids = np.nonzero(m_>0)[:,0]  
                ids = ids[np.random.permutation(len(ids))]
                ids_shuffle = ids[np.random.permutation(len(ids))]  
                num_split = len(ids) // self.pixel_samples + 1
                idset = np.array_split(ids_shuffle, num_split) 
                o_ = I_dec[target, :, :, :].reshape(nImgArray[b], C, H * W).permute(2,0,1)  
                for ids in idset: 
                    o_ids = o_[ids, :, :]
                    glc_ids = glc[target, :, :, :].permute(2,3,0,1).flatten(0,1)[ids,:,:] 
                    o_ids = self.img_embedding(o_ids) 
                    x = o_ids + glc_ids
                    glc_ids = self.glc_upsample(x)
                    x = o_ids + glc_ids
                    x = self.glc_aggregation(x)  
                    x_n, _, _, conf = self.regressor(x, len(ids)) 
                    x_n = F.normalize(x_n, p=2, dim=-1)
                    nout[b, ids, :] = x_n[b,:,:]
                    conf_out[b, ids, :] = conf[b,:,:].to(torch.float32)
                nout = nout.reshape(B,H,W,3).permute(0,3,1,2)
                conf_out = conf_out.reshape(B,H,W,1).permute(0,3,1,2)
                patches_nml.append(nout)

        patches_nml = torch.stack(patches_nml, dim=1)
        merged_tensor_nml = decompose_tensors.merge_tensor_spatial(patches_nml.permute(1,0,2,3,4), method='tile_stride')
        return merged_tensor_nml
