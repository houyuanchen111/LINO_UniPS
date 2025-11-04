import glob
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from lightning import LightningDataModule
import json
import pyexr
from src.utils.hdri_processor import HDRI_Preprocessor
import torch        
import random

class TrainData(Dataset):
    def __init__(
            self, 
            mode: str, # option = ['Train', 'Test', 'Validation'] 
            debug_or_not: bool = False,
            data_root: list = None,
            meta: bool = False,
            low_normal: bool = False, 
            numMaxImages: int = 6, 
            numMinImages: int = 3, 
            imgBufferSize: int = 6, 
            ratio_train: float = 0.999, 
            ratio_val: float = 0.001,
        ):
        self.hdri_root = None
        self.mode = mode
        self.image_size = 512
        self.data_root = data_root 
        self.numMaxImages = numMaxImages
        self.numMinImages = numMinImages
        self.imgBufferSize = imgBufferSize
        self.low_normal = low_normal
        self.hdri_preprocessor = HDRI_Preprocessor(envmap_h=256, envmap_w=512)
        self.objlist = []
        with os.scandir(self.data_root) as entries:
            self.objlist += [entry.path for entry in entries if entry.is_dir()]
        print(f"[Dataset]  => {len(self.objlist)} items selected.")             
        random.shuffle(self.objlist)
        objlist = self.objlist
        total = len(objlist)
        n_train = int(total * ratio_train)
        n_val   = int(total * ratio_val)
        indices = list(range(total))
        random.shuffle(indices)
        train_indices = indices[:n_train]
        val_indices   = indices[n_train:n_train+n_val]
        test_indices  = indices[n_train+n_val:]
        if mode == "Train":
            chosen_indices = train_indices
        elif mode in ["Val", "Validation"]:
            chosen_indices = val_indices
        elif mode in ["Test"]:
           chosen_indices = test_indices 
        else:
            raise ValueError(f"Unknown mode: {mode}, must be 'train','val','test'")
        self.objlist = [objlist[i] for i in chosen_indices]
        print(f"NewData mode={mode}, => {len(self.objlist)} items selected.")

    def horizontal_flip(self,I, N, M): 
        I = I[:, ::-1, :, :]
        N = N[:, ::-1, :]
        N[:, :, 0] *= -1
        M = M[:, ::-1, :]
        return I.copy(), N.copy(), M.copy()
        
    def vertical_flip(self,I, N, M):
        I = I[::-1, :, :, :]
        N = N[::-1, :, :]
        N[:, :, 1] *= -1
        M = M[::-1, :, :]
        return I.copy(), N.copy(), M.copy()
    def rotate(self,I, N, M):
       
        I = I.transpose(1, 0, 2, 3)
        N = N.transpose(1, 0, 2)
        N = N[:, :, [1,0,2]]
        N[:, :, 0] *= -1
        N[:, :, 1] *= -1
        M = M.transpose(1, 0, 2)
        return I.copy(), N.copy(), M.copy()
    

    def color_swap(self,I):
        for k in range(I.shape[3]):
            ids = np.random.permutation(3)
            I[:, :, :, k] = I[:, :, ids, k]
        return I.copy()

    def blend_augumentation(self,I):
            # blending
            k = 0.3
            alpha = k + (1-k) * np.random.rand()
            mean_img = np.mean(I, axis=0, keepdims=True)
            I = alpha * I + (1 - alpha) * mean_img
            return I.copy()

    def quantize_augumentation(self,I):
            for k in range(I.shape[3]):
                temp = 255.0 * (I[:, :, :, k] / np.max(I[:, :, :, k]))
                temp = temp.astype(np.uint8)
                I[:, :, :, k] = temp/255.0
            return I.copy()
    
    def get_point_lights(self, json_file_path,indexofimage):
        with open(json_file_path, "r") as f:
            data = json.load(f)
        point_lights_info_dict = data["frames"][indexofimage]["points_light_info"]
        point_lights_info = np.array([0,0,1,0,0,0,1,0,0,0,1,0]).astype(np.float32) 
        if point_lights_info_dict is not None:
            location = point_lights_info_dict["location"] 
            energy = point_lights_info_dict["energy"]  
            for i in range(len(location)):
                _ = np.append(location[i] / np.linalg.norm(location[i]), energy[i] / 170 ).astype(np.float32) # normalize
                point_lights_info[i*4:i*4+4] = _
        return np.array(point_lights_info)

    def get_hdri(self, json_file_path,indexofimage):
        with open(json_file_path, "r") as f:
            data = json.load(f)
        img_info = data["frames"][indexofimage]
        if "hdri" not in img_info["file_path"]:
            return np.zeros((9, 256, 256)) + 1e-6 
        hdri_file_path_basename = os.path.basename(img_info["hdri_file_path"])
        hdri_file_path = os.path.join(self.hdri_root, hdri_file_path_basename)
        if os.path.exists(hdri_file_path):
            hdri_rot = img_info["rotation_euler"]
            hdri = torch.from_numpy(pyexr.read(hdri_file_path)[..., :3]) # tensor
            envir_map_ldr, envir_map_hdr, envir_map_hdr_raw, view_dirs_world = self.hdri_preprocessor.preprcess_envir_map(hdri, hdri_rot)
            return  np.array(torch.cat([envir_map_ldr, envir_map_hdr, view_dirs_world], dim=0).float())
        return None

    def get_area(self, json_file_path,indexofimage):
        with open(json_file_path, "r") as f:
            data = json.load(f)
        area_light_info_dict = data["frames"][indexofimage]["area_light_info"]
        if area_light_info_dict is not None:
            location = area_light_info_dict["location"] 
            energy = area_light_info_dict["energy"] 
            scale = area_light_info_dict["scale"]  
            return np.append(location[0] / np.linalg.norm(location[0]), [energy[0] / 170, scale[0] / 3]) 
            
        else:
            return np.array([0.,0.,1.,0.,0.]).astype(np.float32)
        
    def get_background(self, json_file_path,indexofimage):
        with open(json_file_path, "r") as f:
            data = json.load(f)
        img_info = data["frames"][indexofimage]
        if "background" not in img_info["file_path"]:
            return np.zeros((9, 256, 256)) + 1e-6 # tensor
        background_energy = img_info["background_energy"] 
        background_light = torch.ones((256, 512, 3)) * background_energy # tensor
        background_light_rot =  np.array([0.,0.,0.])
        envir_map_ldr, envir_map_hdr, envir_map_hdr_raw, view_dirs_world = self.hdri_preprocessor.preprcess_envir_map(background_light, background_light_rot)
        return np.array(torch.cat([envir_map_ldr, envir_map_hdr, view_dirs_world], dim=0).float())
        
    def get_transform_matrix(self, json_file_path, target_file_prefix):
        with open(json_file_path, "r") as f:
            data = json.load(f)
        for frame_list in data["frames"]:
            return np.array(frame_list.get('transform_matrix')) 
            
    def blender_world_normal_2_opengl_camera(self,normals_world: np.ndarray, c2w: np.ndarray, visualization = False) -> np.ndarray:    
        H, W, C = normals_world.shape
        if C == 4:
            normals_world = normals_world[..., :3]
        R_c2w = c2w[:3, :3]
        R_opencv = R_c2w.T

        transformed_normals = normals_world.reshape(-1, 3).T  
        transformed_normals = R_opencv @ transformed_normals
        transformed_normals = transformed_normals.T
        transformed_normals = transformed_normals.reshape(H, W, 3)
        return transformed_normals
    
    def load(self, objlist, objid):
        dirid = objid
        self.objname = objlist[dirid].split('/')[-1]
        directlist = []
        if self.low_normal:
            directlist = glob.glob(os.path.join(objlist[dirid] + '/low_normal_image', f"0*"))
        else:
            directlist = glob.glob(os.path.join(objlist[dirid], f"0*"))
        directlist = sorted(directlist)
        if len(directlist) != 20:
            return 0
        if os.name == 'posix':
            temp = directlist[0].split("/")
        if os.name == 'nt':
            temp = directlist[0].split("\\")
        img_dir = "/".join(temp[:-1])
        self.numberOfImages = self.numMaxImages 
        indexset = np.random.permutation(len(directlist))[:self.numberOfImages] 
        for i, indexofimage in enumerate(indexset):
            img_path = directlist[indexofimage]
            if i == 0:
                # Defensive code
                _ = cv2.imread(img_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if _ is None:
                    return 0
                img = cv2.cvtColor(_, cv2.COLOR_BGR2RGB)
                h = img.shape[0]
                w = img.shape[1]
            else:
                _ = cv2.imread(img_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if _ is None:
                    return 0
                img = cv2.cvtColor( _ , cv2.COLOR_BGR2RGB)
            if img.dtype == 'uint16':
                bit_depth = 65535.0
            img = np.float32(img) / bit_depth

            if i == 0:
                mask = []
                I = np.zeros((len(indexset), h, w, 3), np.float32)
                Env_Light = np.zeros((len(indexset),9,256,256), np.float32)
                Point_Light = np.zeros((len(indexset),12), np.float32)
                Area_Light = np.zeros((len(indexset),5), np.float32)
                
            I[i, :, :, :] = img
            if self.low_normal:
                nml_path = os.path.dirname(img_dir) + '/low_normal/000_000_low_normal.exr'
                json_path = os.path.dirname(img_dir) + '/transforms_low.json'
                c2w = self.get_transform_matrix(json_path, "000_000_low_normal.exr")
                hdri = self.get_hdri(json_path, indexofimage).astype(np.float32)
                if hdri is None:
                    return 0
                point_lights_infos = self.get_point_lights(json_path, indexofimage).astype(np.float32) 
                area_light_info = self.get_area(json_path, indexofimage).astype(np.float32) 
                background_light = self.get_background(json_path, indexofimage).astype(np.float32)
                if hdri.mean() !=0:
                    Env_Light[i, :, :, :] = hdri
                else: 
                    Env_Light[i, :, :, :] = background_light
                Point_Light[i, :] = point_lights_infos
                Area_Light[i, :] = area_light_info
            else:
                nml_path = img_dir + '/normal/000_000_normal.exr'
                json_path = img_dir + '/transforms.json'
                if not os.path.isfile(json_path):
                    return 0
                c2w = self.get_transform_matrix(json_path, "000_000_normal.exr")
            if os.path.exists(json_path) == False or isinstance(c2w,bool): # defensive code
                return 0
            if os.path.isfile(nml_path) == False:
                return 0
            if os.path.isfile(nml_path) and i == 0:
                N_4 = np.array(cv2.resize(pyexr.open(nml_path).get(),(h,w),interpolation=cv2.INTER_NEAREST))
                N = self.blender_world_normal_2_opengl_camera(N_4, c2w, visualization = False).astype(np.float32) # [-1,1], unit
                mask = np.abs(1 - np.sqrt(np.sum(N * N, axis=2))) < 0.5 
                mask = (mask.reshape(h, w, 1)).astype(np.float32) # h, w, 1
                N = N * mask
        imgs_ = I.copy()
        I = np.reshape(I, (-1, h * w, 3))

        """Data Normalization"""
        temp = np.mean(I[:, mask.flatten()==1,:], axis=2) 
        mean = np.mean(temp, axis=1) # the spatially mean value of the mask region
        mx = np.max(temp, axis=1) #the max value of the mutil-light input(almost always 1)
        scale = np.random.rand(self.numberOfImages,) # nparray n float numbers between range(0,1)
        temp = (1-scale) * mean + scale * mx # weighted interpolation between mean and max 
        imgs_ /= (temp.reshape(-1,1,1,1) + 1.0e-6)
        I = imgs_
        I = np.transpose(I, (1, 2, 3, 0)) # h, w, 3, N
        prob = 0.5 # aug
        if self.mode == 'Train':
            if np.random.rand() > prob:
                I, N, mask = self.horizontal_flip(I, N, mask)
            if np.random.rand() > prob:
                I, N, mask = self.vertical_flip(I, N, mask)
            if np.random.rand() > prob:
                I, N, mask = self.rotate(I, N, mask)
        self.I = I
        self.N = N[..., np.newaxis]
        self.mask = mask[..., np.newaxis]
        self.directlist =directlist
        self.env_light = Env_Light
        self.point_lights = Point_Light
        self.area_light = Area_Light
        if np.isnan(self.I).any() or np.isnan(self.N).any() or np.isnan(self.mask).any() or np.isnan(self.env_light).any() or np.isnan(self.point_lights).any() or np.isnan(self.area_light).any():
            print("nan in I, N, mask, env_light, point_lights, area_light")
            return 0
        if np.isinf(self.I).any() or np.isinf(self.N).any() or np.isinf(self.mask).any() or np.isinf(self.env_light).any() or np.isinf(self.point_lights).any() or np.isinf(self.area_light).any():
            print("inf in I, N, mask, env_light, point_lights, area_light")
            return 0
        if np.any(self.I == None) or np.any(self.N == None) or np.any(self.mask == None) or np.any(self.env_light == None) or np.any(self.point_lights == None) or np.any(self.area_light == None):
            print("None in I, N, mask, env_light, point_lights, area_light")
            return 0
        return 1 

    def __getitem__(self, index_):
        objid = index_
        while 1:
            success = self.load(self.objlist, objid)
            if success:
                break
            else:
                objid = np.random.randint(0, len(self.objlist))
        img = self.I.transpose(2,0,1,3) # 3 h w Nmax
        nml = self.N.transpose(2,0,1,3) # 3 h w 1
        mask = self.mask.transpose(2,0,1,3) # 1 h w 1
        objname = os.path.basename(os.path.basename(self.objlist[objid]))
        numberOfImages = self.numberOfImages
        try:
            output = {
                    'img': img,
                    'nml': nml,
                    'mask': mask,
                    'directlist': self.directlist,
                    'objname': objname,
                    'numberOfImages': numberOfImages
                }
            if hasattr(self, 'roi'):
                output['roi'] = self.roi
            if hasattr(self, 'env_light'):
                output['env_light'] = self.env_light
            if hasattr(self, 'point_lights'):
                output['point_lights'] = self.point_lights
            if hasattr(self, 'area_light'):
                output['area_light'] = self.area_light
            if hasattr(self, "I_full"):
                output['img_full'] = self.I_full.transpose(2,0,1,3)
            if hasattr(self, "N_full"):
                output["N_full"] = self.N_full.transpose(2,0,1,3)
            if hasattr(self, "mask_full"):
                output["mask_full"] = self.mask_full.transpose(2,0,1,3)
            return output
        except:
            output = {
                'img': img,
                'nml': nml,
                'mask': mask,
                'directlist': self.directlist,
                'objname': objname,
                'numberOfImages': numberOfImages
            }
            if hasattr(self, 'roi'):
                output['roi'] = self.roi
            return output

    def __len__(self):
        return len(self.objlist)


