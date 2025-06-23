from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything 
import argparse
import torch
def predict_normal():
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
        default="data/DiLiGenT/",
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=16,
        help="Number of images to process"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
    )

    
    args = parser.parse_args()
    seed_everything(seed=args.seed, workers=True)
    lino = torch.hub.load(
            "houyuanchen111/LINO_UniPS",
            "lino_unips",
            pretrained=True,
            task_name=args.task_name 
        )
    testdata = torch.hub.load(
            "houyuanchen111/LINO_UniPS",
            "load_test_data",
            data_root=[args.data_root],
            numofimages=args.num_images
        )
    predict_normal()