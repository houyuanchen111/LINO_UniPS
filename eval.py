from src.data import TestData
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.models import LiNo_UniPS
from pytorch_lightning import seed_everything 
import argparse

def predict_normal(data_root: list,numberofinput: int):
    test_dataset = TestData([data_root],numberofinput)
    test_loader = DataLoader(test_dataset, batch_size=1)
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
        "--ckpt_path", 
        type=str, 
        default="weights/lino/lino.pth",
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
    lino = LiNo_UniPS(task_name=args.task_name)
    lino.from_pretrained(args.ckpt_path)
    predict_normal(args.data_root, args.num_images)