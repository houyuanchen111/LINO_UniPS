from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything 
import argparse
from src.data import TestData
from src.models import LiNo_UniPS_PBR
import torch
def predict():
    test_loader = DataLoader(testdata, batch_size=1)
    trainer = pl.Trainer(accelerator="auto", devices=1,precision="bf16-mixed")
    trainer.test(model=lino, dataloaders=test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", 
        type=str, 
        default="DiLiGenT", 
        help="Name of the task"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="data/LUCES/data/",
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=32,
        help="Number of images to process"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
    )

    
    args = parser.parse_args()
    testdata = TestData(args.data_root, args.num_images)
    lino = LiNo_UniPS_PBR(task_name=args.task_name, brdf=True)
    lino.from_pretrained("/share/project/cwm/houyuan.chen/UPS_Lightning/ckpt/Net2_v15_demo_v1_pbr/epoch_008.pth")
    predict()



