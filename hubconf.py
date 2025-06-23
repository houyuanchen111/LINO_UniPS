import os
import numpy as np
from typing import Optional
import torch
from src.models import LiNo_UniPS 
from src.data import TestData
from src.data import DemoData
dependencies = ['torch', 'pytorch_lightning', 'numpy']

DEFAULT_MODEL_URL = "https://huggingface.co/houyuanchen/lino/resolve/main/lino.pth"  

def lino_unips(pretrained=True, task_name="DiLiGenT", **kwargs):
    model = LiNo_UniPS(task_name=task_name, **kwargs)
    if pretrained:
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                DEFAULT_MODEL_URL,
                progress=True
            )
            model.load_state_dict(state_dict) #
            model.eval()
            print("load lino_unips successfully")
        except Exception as e:
            print(f"error{e}")
            
    return model

def load_test_data(data_root: list, numofimages: int):
    return TestData(data_root,numofimages)

def load_data(input_imgs_list, input_mask):
    return DemoData(input_imgs_list,input_mask)

def LINO(local_file_path: Optional[str] = None):
    """
    Load the LINO model with optional local file path for state_dict.
    
    Args:
        local_file_path (str, optional): Path to the local state_dict file. If None, uses the default URL.
        
    Returns:
        Predictor: An instance of the Predictor class with the loaded model.
    """
    state_dict = _load_state_dict(local_file_path)
    model = LiNo_UniPS()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return Predictor(model)


def _load_state_dict(local_file_path: Optional[str] = None):
    if local_file_path is not None and os.path.exists(local_file_path):
        # Load state_dict from local file
        state_dict = torch.load(local_file_path, weights_only=False, map_location=torch.device("cpu"))
    else:
        # Load state_dict from the default URL
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True, map_location=torch.device("cpu"))

    return state_dict


class Predictor:
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        self.model.to(self.device, dtype=self.dtype)
    
    def predict(self, input_imgs_list, input_mask):
        demodata = load_data(input_imgs_list, input_mask)
        data = demodata[0]
        # 将数据 batch 化
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = torch.tensor(data[key], device=self.device, dtype=self.dtype)[None, ...]  # Add None to keep the batch dimension
            elif isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(self.device, dtype=self.dtype)[None, ...]
            elif data[key] is None:
                data[key] = None
            else:
                raise TypeError(f"Unsupported data type: {type(data[key])}")

        with torch.no_grad():
            output = self.model(data)
        return output
    

def _test_run():
    # Example usage
    from PIL import Image
    input_imgs_paths_list = [
            "/share/project/cwm/hong.li/code/unips/LINO_UniPS/demo/basket/L_1.png", 
            "/share/project/cwm/hong.li/code/unips/LINO_UniPS/demo/basket/L_2.png",
            "/share/project/cwm/hong.li/code/unips/LINO_UniPS/demo/basket/L_3.png",
            "/share/project/cwm/hong.li/code/unips/LINO_UniPS/demo/basket/L_4.png",
            "/share/project/cwm/hong.li/code/unips/LINO_UniPS/demo/basket/L_5.png",
            "/share/project/cwm/hong.li/code/unips/LINO_UniPS/demo/basket/L_6.png",
            "/share/project/cwm/hong.li/code/unips/LINO_UniPS/demo/basket/L_7.png",
            "/share/project/cwm/hong.li/code/unips/LINO_UniPS/demo/basket/L_8.png",
        ]
    
    input_imgs_list = [(np.array(Image.open(img_path)), None) for img_path in input_imgs_paths_list]
    
    input_mask = None
    
    predictor = LINO(local_file_path="/share/project/cwm/hong.li/code/unips/LINO_UniPS/weights/lino/lino.pth")
    result = predictor.predict(input_imgs_list, input_mask)
    
    print("Prediction result:", result)

if __name__ == "__main__":
    _test_run()