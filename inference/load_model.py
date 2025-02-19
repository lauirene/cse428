

def load_model(model_path,input_row_size,input_col_size):
    """
    Load a model from a file.

    Args:
        model_path (str): The path to the model file.
        input_row_size (int): The number of rows in the input matrix.
        input_col_size (int): The number of columns in the input matrix.

    Returns:
        model: The loaded model.
    """
    import torch
    import model.Vision_Transformer_count as Vision_Transformer
    from model.pos_embed import interpolate_pos_embed_inputsize
    import torch.nn as nn


    model_name="vit_large_patch16"
    patch_size=16
    task = 6

    patch_wise_size = (input_row_size//patch_size, input_col_size//patch_size)
    vit_backbone = Vision_Transformer.__dict__[model_name](img_size=(input_row_size,input_col_size))
    checkpoint = torch.load(model_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = vit_backbone.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    interpolate_pos_embed_inputsize(vit_backbone, checkpoint_model,input_size=patch_wise_size,
                                            use_decoder=False)
    # load pre-trained model
    msg = vit_backbone.load_state_dict(checkpoint_model, strict=False)
    print("Loading pre-train encoder message:",msg)

    from model.Finetune_Model_Head import Finetune_Model_Head
    model = Finetune_Model_Head(vit_backbone, task=task,
                            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                        mlp_ratio=4., norm_layer=nn.LayerNorm,pos_embed_size=patch_wise_size)
    checkpoint = torch.load(model_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    #loading pre-trained decoder
    interpolate_pos_embed_inputsize(model, checkpoint['model'],
                                    input_size=patch_wise_size,use_decoder=True)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print("Loading pre-train model decoder message:",msg)
    return model # return the loaded model

def to_cuda(x):
    """
    Move a tensor to the GPU.

    Args:
        x (torch.Tensor): The tensor to move to the GPU.

    Returns:
        torch.Tensor: The tensor on the GPU.
    """
    import torch
    if x is not None:
        return x.cuda()
    else:
        return None
    
def to_float(x):
    """
    Convert a tensor to float.

    Args:
        x (torch.Tensor): The tensor to convert to float.

    Returns:
        torch.Tensor: The tensor as float.
    """
    import torch
    if x is not None:
        return x.float()
    else:
        return None

def convert_rgb(data_log,max_value):
    import torch
    if len(data_log.shape)==2:
        data_log = data_log[None,:,:]
    data_red = torch.ones(data_log.shape)
    data_log1 = (max_value-data_log)/max_value
    data_rgb = torch.cat([data_red,data_log1,data_log1],dim=0)
    data_rgb = data_rgb.permute(1,2,0)#transform only accept channel last case
    return data_rgb

def format_input(input):
    """
    Format the input for the model.

    Args:
        input (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The formatted input tensor.
    """
    import torch
    import torchvision.transforms as transforms
    transform_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    input = torch.nan_to_num(input)
    max_value = torch.max(input)
    input = torch.log10(input+1)
    max_value = torch.log10(max_value+1)
    input = convert_rgb(input,max_value)
    
    input = transform_input(input)
    return input