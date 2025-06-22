import torch
from src.models import LiNo_UniPS 
dependencies = ['torch', 'pytorch_lightning']
DEFAULT_MODEL_URL = "https://huggingface.co/houyuanchen/lino/blob/main/lino.pth" 

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

