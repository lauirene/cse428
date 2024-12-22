import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import datetime
import json
import numpy as np
import torchvision.transforms as transforms

from ops.distribute_utils import init_distributed_mode,get_world_size,get_rank,is_main_process
from ops.Logger import print_important_info,print_warning_info
from data_processing.pretrain_dataset import Pretrain_Dataset
from data_processing.finetune_collate_fn import collate_fn

def parse_text(config_file, data_dir):
    train_list=[]
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            line = line.replace('\n', '')
            if len(line) == 0:
                continue
            current_path = os.path.join(data_dir, line)
            if not os.path.exists(current_path):
                print("The sub-directory {} does not exist in the data directory".format(current_path))
                print("Please check the sub-directory name in the {} file".format(config_file))
                continue
            train_list.append(current_path)
    return train_list

def configure_data_loader(args):
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_dir=os.path.abspath(args.data_path)
    train_config= os.path.abspath(args.train_config)
    train_list = parse_text(train_config, data_dir)
    val_config= os.path.abspath(args.valid_config)
    val_list = parse_text(val_config, data_dir)
    input_row_size = args.input_row_size
    input_col_size = args.input_col_size
    sparsity_filter = float(args.sparsity_ratio)
    patch_size = args.patch_size

    dataset_train = Pretrain_Dataset(train_list,transform=transform_train,
                                     sparsity_filter=sparsity_filter,patch_size=patch_size,
                                     window_height=input_row_size,window_width=input_col_size)
    dataset_val = Pretrain_Dataset(val_list,transform=transform_train,
                                   sparsity_filter=sparsity_filter, patch_size=patch_size,
                                        window_height=input_row_size,window_width=input_col_size)

    if  args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)
        global_rank = -1
    sample_batch_size = args.batch_size
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=sample_batch_size, sampler=sampler_train, 
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
        collate_fn=collate_fn
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=sample_batch_size, sampler=sampler_val, 
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
        collate_fn=collate_fn
    )
    return data_loader_train, data_loader_val

def config_writer(output_dir,tensorboard_log):
    tensorboard_dir = os.path.join(output_dir,'tensorboard')
    os.makedirs(tensorboard_dir,exist_ok=True)
    if tensorboard_log:
        from torch.utils.tensorboard import SummaryWriter
        log_writer = SummaryWriter(tensorboard_dir)
    else:
        log_writer = None
    return log_writer
def main_worker(gpu, ngpus_per_node,args):
    if ngpus_per_node>1:
        init_distributed_mode(gpu,ngpus_per_node,args)
    else:
        args.distributed=False
        print_warning_info("The distributed mode is disabled.\n For pre-training, one GPU may take very long to train!")
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    if  args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
    else:
        global_rank = -1
        num_tasks = 1
    output_dir = os.path.abspath(args.output)
    if global_rank==0:
        os.makedirs(output_dir,exist_ok=True)
        log_writer =config_writer(output_dir,args.tensorboard)
    elif args.distributed:
        log_writer = None
    else:
        os.makedirs(output_dir,exist_ok=True)
        log_writer = config_writer(output_dir,args.tensorboard)

    cudnn.benchmark = True
    device = torch.device(args.device)

    # Data loading code
    data_loader_train, data_loader_val = configure_data_loader(args)
    print("Data loader is configured!")

    # Configure the model

    model = models_hicfoundation.__dict__[args.model]()