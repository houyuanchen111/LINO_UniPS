import argparse
import json
import os

def save_args(args, dst):
    os.makedirs(dst, exist_ok=True)
    if dst is None:
        return
    with open(f"{dst}args.json", 'w+') as f:
        json.dump(vars(args), f)

def load_args(args, src):
    args_copy = args
    if src is not None and os.path.exists(src):
        with open(src, 'r') as f:
            ns = json.load(f)
        args_loaded = argparse.Namespace(**ns)
        for arg in vars(args_copy):
            if arg in ('training_dir', 'test_dir',  'pretrained', 
                        't_check_point', 't_test_dir', 't_out_dir', 't_num_img', 't_test_suffix', 't_test_ext', 't_test_data_type', 't_loader_imgsize', 't_normalize'):
                setattr(args_loaded, arg, getattr(args_copy, arg))    
        return args_loaded
    else:
        print('no arg file found! args was not updated')
        return args




        

