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
        input_pkl = smooth_pkl(input_pkl,os.path.abspath(args.output))
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
