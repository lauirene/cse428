import argparse

def argparser_infer():
    parser = argparse.ArgumentParser('HiCFoundation inference', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size of the input')
    # Dataset parameters
    parser.add_argument('--input', type=str, help='a .hic/.cool/.pkl/.txt/.pairs/.npy file records Hi-C/scHi-C matrix')
    parser.add_argument('--resolution', default=10000, type=int,help='resolution of the input matrix')
    parser.add_argument("--task",default=0,type=int,help="1: Reproducibility analysis; \n 2: Loop calling; \n \
                        3: Resolution enhancement; \n 4: Epigenomic assay prediction; \n 5: scHi-C enhancement")
    
    parser.add_argument('--input_row_size', default=224, type=int,
                        help='images input size')
    parser.add_argument("--input_col_size",default=4000,type=int,help="span size for the input matrix")
    parser.add_argument("--patch_size",default=16,type=int,help="patch size for the input matrix")

    parser.add_argument('--stride', default=20, type=int,
                        help='images input size')
    parser.add_argument("--bound",default=200,type=int,help="off-diagonal bound for the scanning")
    parser.add_argument('--num_workers', default=8, type=int,help="data loading workers per GPU")
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument("--model_path",default='',help='load fine-tuned model for inference')
    parser.add_argument('--output', default='hicfoundation_inference',help='output directory to save the results')
    parser.add_argument("--print_freq",default=1,type=int,
                        help="log frequency for output log during inference")
    parser.add_argument("--gpu",default=None,type=str,help="which gpu to use")
    parser.add_argument("--genome_id",type=str,default="hg38", help="genome id for generating .hic file. \n \
                        Must be one of hg18, hg19, hg38, dMel, mm9, mm10, anasPlat1, bTaurus3, canFam3, equCab2, \
                        galGal4, Pf3D7, sacCer3, sCerS288c, susScr3, or TAIR10; \n \
                         alternatively, this can be the path of the chrom.sizes file that lists on each line the name and size of the chromosomes.")
    return parser