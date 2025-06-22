
import numpy as np
from torch.utils.data import Dataset
import cv2
import glob
import os

def get_roi(mask, margin=8):
    """
    """
    h0, w0 = mask.shape[:2]
    
    if  mask is not None:
        rows, cols = np.nonzero(mask)
        rowmin, rowmax = np.min(rows), np.max(rows)
        colmin, colmax = np.min(cols), np.max(cols)
        row, col = rowmax - rowmin, colmax - colmin
        
        flag = not (rowmin - margin <= 0 or rowmax + margin > h0 or 
                    colmin - margin <= 0 or colmax + margin > w0)
        
        if row > col and flag:
            r_s, r_e = rowmin - margin, rowmax + margin
            c_s, c_e = max(colmin - int(0.5 * (row - col)) - margin, 0), \
                       min(colmax + int(0.5 * (row - col)) + margin, w0)
        elif col >= row and flag:
            r_s, r_e = max(rowmin - int(0.5 * (col - row)) - margin, 0), \
                       min(rowmax + int(0.5 * (col - row)) + margin, h0)
            c_s, c_e = colmin - margin, colmax + margin
        else:
            r_s, r_e, c_s, c_e = 0, h0, 0, w0
    else:
        r_s, r_e, c_s, c_e = 0, h0, 0, w0
    
    return np.array([h0, w0, r_s, r_e, c_s, c_e])

def crop_and_resize_img(img, roi, max_image_resolution=6000):
    
   
    h0, w0, r_s, r_e, c_s, c_e = roi
    
    img = img[r_s:r_e, c_s:c_e, :]
 
    
    h = max(512, min(max_image_resolution, (max(img.shape[:2]) // 512) * 512))
    w = h
    
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    
    bit_depth = 255.0 if img.dtype == np.uint8 else 65535.0 if img.dtype == np.uint16 else 1.0
    img = np.float32(img) / bit_depth
    
    return img

def crop_and_resize_mask(mask, roi, max_image_resolution=6000):
    
    
    h0, w0, r_s, r_e, c_s, c_e = roi
    

    mask = mask[r_s:r_e, c_s:c_e]
    
    h = max(512, min(max_image_resolution, (max(mask.shape[:2]) // 512) * 512))
    w = h
    
   
    mask = np.float32(cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC) > 0.5)
    
    return mask

class DemoData(Dataset):
    def __init__(self,input_imgs_list,input_mask):
         self.input_imgs_list = input_imgs_list
         self.input_mask = input_mask
    def __len__(self):
        return 1
    def load(self,input_images_list,mask):
        if mask is None:
            mask = np.ones_like(input_images_list[0][0])
        else:
            mask = np.array(mask)
        mask = mask[:,:,0]
        if mask.max() <= 1.0:
            self.mask_original = mask[:,:,None]
        else:
            self.mask_original = mask[:,:,None] / 255.0
        self.roi = get_roi(mask)
        for i in range(len(input_images_list)):
            img = input_images_list[i]
            input_images_list[i]= crop_and_resize_img(img[0], self.roi)
        I = np.array(input_images_list)
        numberofimages,h,w,_ = I.shape
        mask = crop_and_resize_mask(mask, self.roi)
        I = np.reshape(I, (-1, h * w, 3))
        temp = np.mean(I[:, mask.flatten()==1,:], axis=2)
        mx = np.max(temp, axis=1)
        temp = mx      
        I /= (temp.reshape(-1,1,1) + 1.0e-6)
        I = np.transpose(I, (1, 2, 0))
        I = I.reshape(h, w, 3, numberofimages)
        mask = (mask.reshape(h, w, 1)).astype(np.float32) 
        h = mask.shape[0]
        w = mask.shape[1]
        self.h = h
        self.w = w
        self.I = I 
        self.N = np.ones((h, w, 3), np.float32)
        self.mask = mask
        return 1
    def __getitem__(self, idx):
        self.load(self.input_imgs_list,self.input_mask)
        return {
            "imgs":self.I.transpose(2,0,1,3),
            "mask":self.mask.transpose(2,0,1),
            "mask_original":self.mask_original.transpose(2,0,1),
            "roi":self.roi
        }
    
class TestData(Dataset):
    def __init__(
            self, 
            data_root: list = None, 
            numofimages: int = 16
          
        ):
        self.data_root = data_root 
        self.numberOfImages = numofimages
        self.objlist = []
        for i in range(len(self.data_root)):
             with os.scandir(self.data_root[i]) as entries:
                self.objlist += [entry.path for entry in entries if entry.is_dir()]
             print(f"[Dataset]  => {len(self.objlist)} items selected.")
        objlist = self.objlist
        total = len(objlist)
        indices = list(range(total))
        self.objlist = [objlist[i] for i in indices]
        print(f"Test, => {len(self.objlist)} items selected.")
    def load(self, objlist, dirid):
        obj_path = objlist[dirid]
        if "DiLiGenT" in obj_path:
            nml_path = os.path.join(obj_path, "Normal_gt.png")
            if "10" not in obj_path: # diligent
                directlist = sorted(glob.glob(os.path.join(obj_path, f"0*")))
            else: # diligent100
                directlist = sorted([
                    path for path in glob.glob(os.path.join(obj_path, "*.png"))
                    if not os.path.basename(path).lower() == "mask.png"
                ])           
        elif "LUCES" in obj_path:
            nml_path = os.path.join(obj_path, "normals.png")
            directlist = sorted([
                f for i in range(1, 52) for f in glob.glob(os.path.join(obj_path, f"{i:02d}*"))
            ])
        elif "Real" in obj_path:
            nml_path = os.path.join(obj_path, "Normal_gt.png")
            directlist = sorted(glob.glob(os.path.join(obj_path, f"L*")))
        else:
            print(f"error:unknown dataset{obj_path}")
            return 0

       
        num_images_to_sample = self.numberOfImages 
        if num_images_to_sample is not None and num_images_to_sample < len(directlist):
            indexset = np.random.permutation(len(directlist))[:num_images_to_sample]
        else:
            indexset = range(len(directlist))
        
        I = None
        mask = None
        N = None
        n_true = None
        
        for i, indexofimage in enumerate(indexset):
            img_path = directlist[indexofimage]
            read_img = cv2.imread(img_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if read_img is None:
                print(f"warning: can not read {img_path}")
                return 0 
            
            img = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)
            if i == 0:
                         
                mask_path = os.path.join(obj_path, "mask.png")
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
                else:
                    mask = np.ones_like(read_img)[:,:,0]
                
                if os.path.exists(nml_path):
                    bit_depth = 65535.0 if "LUCES" in obj_path else 255.0
                    N = cv2.cvtColor(cv2.imread(nml_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB) / bit_depth
                    N = 2 * N - 1
                    N = N / np.linalg.norm(N, axis=2, keepdims=True)
                    N = N * mask[:, :, np.newaxis]
                    n_true = N


                self.roi = get_roi(mask)
                mask = crop_and_resize_mask(mask, self.roi)
              
                
            img= crop_and_resize_img(img, self.roi)
            h, w = img.shape[:2]
            if i == 0:
                I = np.zeros((len(indexset), h, w, 3), np.float32)
            I[i, :, :, :] = img

      
        imgs_ = I.copy()
        I = np.reshape(I, (-1, h * w, 3))

        """Data Normalization"""
        temp = np.mean(I[:, mask.flatten()==1,:], axis=2)
        mean = np.mean(temp, axis=1) 
        mx = np.max(temp, axis=1)
        scale = np.random.rand(I.shape[0],) 
        temp = (1-scale) * mean + scale * mx 
        imgs_ /= (temp.reshape(-1,1,1,1) + 1.0e-6)
        I = imgs_
        I = np.transpose(I, (1, 2, 3, 0)) 
        mask = (mask.reshape(h, w, 1)).astype(np.float32) 
        h = mask.shape[0]
        w = mask.shape[1]
        self.h = h
        self.w = w
        self.I = I #
        if ("DiLiGenT" in obj_path and "10" in obj_path) or "Real" in obj_path: # diligent100
            self.N = np.ones((h,w,3,1)) 
        else:
            self.N = n_true[:,:,:,np.newaxis] 
        self.mask = mask
        self.directlist = directlist

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
        objname = os.path.basename(os.path.basename(self.objlist[objid]))
        numberOfImages = self.numberOfImages
        try:
            output = {
                    'imgs': img,
                    'nml': nml,
                    "mask":self.mask.transpose(2,0,1),
                    'directlist': self.directlist,
                    'objname': objname,
                    'numberOfImages': numberOfImages,
                    "roi":self.roi
                }
            return output
        except:
            raise KeyError

    def __len__(self):
        return len(self.objlist)