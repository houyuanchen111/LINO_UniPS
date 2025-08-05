from torch.utils.data import DataLoader
from src.data.data_train_module import TrainData
import pytorch_lightning as pl
from pytorch_lightning import seed_everything 
import argparse
from src.models.Net_train_module import Net, LINO_UniPSModule
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def train_model(args):
    train_data = TrainData(
        mode='Train',    
        data_root=args.data_root,
        low_normal=args.low_normal
    )
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_data = TrainData(
        mode='Val',    
        data_root=args.data_root,
        low_normal=args.low_normal
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    # 初始化模型
    net = Net(
        pixel_samples=args.pixel_samples,
        output="normal",  
        depth=args.depth
    )
    

    model = LINO_UniPSModule(
        net=net,
        optimizer_class=optim.AdamW,
        scheduler_class=StepLR,
        canonical_resolution=args.canonical_resolution,
        sample_num=args.pixel_samples,
        save_dir=args.save_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        min_lr=args.min_lr,
        step_size=args.step_size,
        gamma=args.gamma
    )
    
    # 配置训练器
    trainer = pl.Trainer(
        accelerator="auto",
        devices=args.devices,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val/loss",
                dirpath=args.save_dir,
                filename="best_model_{epoch:02d}_{val_loss:.4f}",
                save_top_k=3,
                mode="min"
            ),
            pl.callbacks.EarlyStopping(
                monitor="val/loss",
                patience=args.patience,
                mode="min"
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        ],
        logger=pl.loggers.TensorBoardLogger(
            save_dir=args.save_dir,
            name="lightning_logs"
        )
    )
    
    # 开始训练
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiNO UniPS Training Script")
    
    parser.add_argument(
        "--low_normal",
        type=bool,
        default=True,
        help="Low normal mode or high normal mode"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=6,
        help="Number of images to process"
    )
    parser.add_argument(
        "--pixel_samples", 
        type=int, 
        default=2048,
        help="Number of pixel samples for training"
    )
    
    parser.add_argument(
        "--depth", 
        type=int, 
        default=4,
        help="Depth of the network, default is 4"
    )
    parser.add_argument(
        "--canonical_resolution", 
        type=int, 
        default=256,
        help="Canonical resolution for processing"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.05,
        help="Weight decay"
    )
    parser.add_argument(
        "--max_epochs", 
        type=int, 
        default=100,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--min_lr", 
        type=float, 
        default=1e-6,
        help="Minimum learning rate"
    )
    parser.add_argument(
        "--step_size", 
        type=int, 
        default=10,
        help="Step size for StepLR scheduler"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.8,
        help="Gamma for StepLR scheduler"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=10,
        help="Early stopping patience"
    )
    
    parser.add_argument(
        "--devices", 
        type=int, 
        default=1,
        help="Number of devices to use"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="checkpoints/",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--val_check_interval", 
        type=float, 
        default=1.0,
        help="Validation check interval (epochs)"
    )
    parser.add_argument(
        "--log_every_n_steps", 
        type=int, 
        default=50,
        help="Log every n steps"
    )

    args = parser.parse_args()

    seed_everything(seed=args.seed, workers=True)
    train_model(args)
   