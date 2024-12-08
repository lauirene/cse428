import torch
import os
import torch.nn as nn

from utils.hic_coverage import calculate_coverage
from data_processing.inference_dataset import Inference_Dataset

def configure_dataset(args,input_pkl):
    resolution = args.resolution
    import torchvision.transforms as transforms
    transform_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if args.task==3:
        #resolution enhancement
        fill_diagonal_zero=True
        
    else:
        fill_diagonal_zero=False
    if args.task==3:
        #judge if it is a very deep sequencing data, if it is, set max_cutoff to None
        coverage_perresolution = calculate_coverage(input_pkl)/resolution
        if coverage_perresolution>1:
            max_cutoff = None
        else:
            max_cutoff = 100
    elif args.task==2:
        #loop calling
        max_cutoff = 1000
    else:
        max_cutoff = None
    
    if args.task==4:
        #epigenomic assay prediction
        locus_embedding = True
    else:
        locus_embedding = False
        
    bounding = args.bound
    stride = args.stride
    input_row_size = args.input_row_size
    input_col_size = args.input_col_size
    dataset = Inference_Dataset(data_path=input_pkl,   
                            transform=transform_input,
                            stride=stride,
                            window_height= input_row_size,
                            window_width = input_col_size,
                            max_cutoff=max_cutoff,
                            fill_diagonal_zero=fill_diagonal_zero,
                            bounding=bounding,
                            locus_embedding=locus_embedding)
    sample_batch_size = args.batch_size
    data_loader_test = torch.utils.data.DataLoader(
        dataset,
        batch_size=sample_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)
    return data_loader_test

def main_worker(args, input_pkl):
    output_dir = os.path.abspath(args.output)
    dataloader = configure_dataset(args, input_pkl)
    import model.Vision_Transformer_count as Vision_Transformer
    vit_backbone = Vision_Transformer.__dict__[args.model]()
    patch_wise_size = (args.input_row_size//args.patch_size,args.input_col_size//args.patch_size)
    from model.Finetune_Model_Head import Finetune_Model_Head
    model = Finetune_Model_Head(vit_backbone, task=args.task,
                            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                        mlp_ratio=4., norm_layer=nn.LayerNorm,pos_embed_size=patch_wise_size)
    
    #check model_path exists
    model_path = os.path.abspath(args.model_path)
    assert os.path.exists(model_path), "model_path does not exist"
    #load model weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if "model" in checkpoint:
        checkpoint_model = checkpoint["model"]
    elif "state_dict" in checkpoint:
        checkpoint_model = checkpoint["state_dict"]
    else:
        checkpoint_model = checkpoint
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print("Loading pre-train model decoder message:",msg)

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    from inference.inference_worker import inference_worker
    return_dict= inference_worker(model,dataloader,
                                  log_dir=output_dir,
                                  args=args)