import os
import timm
assert timm.__version__ == "0.3.2" # version check for timm
from ops.argparser import  argparser_infer
from ops.file_format_convert import convert_to_pkl

def main(args):
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("local ip: ",local_ip)
    #format processing, convert different formats to .pkl format for further processing
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir,exist_ok=True)
    input_file = os.path.abspath(args.input)
    config_resolution = args.resolution
    input_pkl=convert_to_pkl(input_file, output_dir,config_resolution)
    #for reproducibility analysis, we need to smooth the matrix to generate embeddings.
    if args.task==1:
        from ops.smooth_matrix import smooth_pkl
        smooth_pkl_file = os.path.join(output_dir,"input_smoothed.pkl")
        input_pkl = smooth_pkl(input_pkl,smooth_pkl_file)
        print("Reproducibility analysis smoothed input matrix saved to ",input_pkl)
    from inference.main_worker import main_worker
    main_worker(args, input_pkl)


if __name__ == '__main__':
    print("HiCFoundation inference started!")
    parser = argparser_infer()
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #print mode based on --task
    if args.task==1:
        print("Reproducibility analysis")
    elif args.task==2:
        print("Loop calling")
    elif args.task==3:
        print("Resolution enhancement")
    elif args.task==4:
        print("Epigenomic assay prediction")
    elif args.task==5:
        print("scHi-C enhancement")
    else:
        print("Unknown task specified ",args.task)
        print("Please specify the task using --task with 1,2,3,4,5")
        exit(1)
    #check the specied input size, must be a multiple of args.patch_size
    if args.input_row_size%args.patch_size!=0 or args.input_col_size%args.patch_size!=0:
        print("args configuration error: input_row_size and input_col_size must be a multiple of patch_size")
        exit(1)
    #output the args in a beautiful format
    main(args)

"""
```
python3 inference.py --input [input_file] --batch_size [infer_batch_size] --resolution [hic_resolution] --task 1 --input_row_size [input_submatrix_length] --input_col_size [input_submatrix_width] --stride [stride] --bound [scan_boundary] --model_path [trained_model_path] --output [output_dir] --gpu [gpu]
```
- input_file: a .hic/.cool/.pkl/.txt/.pairs/.npy file records Hi-C matrix.
- infer_batch_size: batch size of the input during inference, recommended: 4 for small GPU.
- hic_resolution: resolution of the input matrix, default: 25000 (25 kb for reproducibility task).
- input_submatrix_length: input submatrix row size, default: 224.
- input_submatrix_width: input submatrix column size, default: 224.
- stride: scanning stride for the input Hi-C matrix, default: 20.
- scan_boundary: off-diagonal bound for the scanning, default: 0.
- trained_model_path: load fine-tuned model for inference.
- output_dir: output directory to save the results, default: hicfoundation_inference.
- gpu: which gpu to use, default: None (will use all GPU).

Example command:

"""