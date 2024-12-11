import math
import sys
import numpy as np
from typing import Iterable
import torch
import torch.nn as nn
import time
from ops.Logger import MetricLogger,SmoothedValue
import os
from collections import defaultdict
from ops.sparse_ops import array_to_coo
from scipy.sparse import coo_matrix,triu
def inference_worker(model,data_loader,log_dir=None,args=None):
    """
    model: model for inference
    data_loader: data loader for inference
    log_dir: log directory for inference
    args: arguments for inference
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Inference: '
    print_freq = args.print_freq
    print("number of iterations: ",len(data_loader))
    num_iter = len(data_loader)
    dataset_shape_dict = data_loader.dataset.dataset_shape
    infer_task = args.task
    if infer_task==1:
        output_dict=defaultdict(list)
    elif infer_task==2 or infer_task==3 or infer_task==5:
        output_dict={}
        for chrom in dataset_shape_dict:
            output_dict[chrom] = {"row_record":[],"col_record":[],"value_record":[],"count_record":[]}
    elif infer_task==4:
        #epigenomic assay prediction
        num_track = 6
        output_dict={}
        for chrom in dataset_shape_dict:
            current_shape = dataset_shape_dict[chrom]
            current_length = current_shape[0]
            mean_array = np.zeros([num_track,current_length])
            count_array = np.zeros([num_track,current_length])
            output_dict[chrom] = {"mean":mean_array,"count":count_array}

    if infer_task==3:
        #resolution enhancement
        cutoff= 1000
        cutoff = torch.tensor(cutoff).float().cuda()
        log_cutoff = torch.log10(cutoff+1).cuda()
    if infer_task==5:
        #scHi-C enhancement
        cutoff= 1000
        cutoff = torch.tensor(cutoff).float().cuda()
        log_cutoff = torch.log10(cutoff+1).cuda()
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        input,total_count,indexes = data
        input = input.cuda()
        input = input.float()
        total_count = total_count.cuda()
        total_count = total_count.float()
        with torch.no_grad():
            output = model(input,total_count) 
            # fixme: loop, and epigenomic assay prediction did not take count in benchmark, I think this will not impact performance, will check later. If yes, will revise it to model(input)
        if infer_task==1:
            #reproducibility analysis
            pass
        elif infer_task==2:
            #loop calling
            output= torch.sigmoid(output)
        elif infer_task==3 or infer_task==5:
            #resolution enhancement and scHi-C enhancement
            output = output*log_cutoff
            output = torch.pow(10,output)-1
            output = torch.clamp(output,min=0)
        output = output.detach().cpu().numpy()
        input = input.detach().cpu().numpy()
        chrs, row_starts, col_starts = indexes
        for i in range(len(output)):
            chr = chrs[i]
            row_start = row_starts[i]
            col_start = col_starts[i]
            row_start = int(row_start)
            col_start = int(col_start)
            current_shape = dataset_shape_dict[chr]
            row_end = min(row_start+args.input_row_size,current_shape[0])
            col_end = min(col_start+args.input_col_size,current_shape[1])
            current_input = input[i]
            input_count = np.sum(current_input)
            #ignore empty matrix
            if input_count==0:
                print("empty matrix:",chr,row_start,col_start)
                continue

            # # may be not necessary, will check if error happens
            # if input_count<=len(current_input):
            #     #skip super low read count matrix
            #     #that's to say, <1 read per 10 kb, samller than 0.3M total read for human
            #     continue
            cur_output = output[i]
            if infer_task==1:
                match_key = f"{chr}:{row_start},{col_start}"
                output_dict[match_key] = cur_output
            elif infer_task==2 or infer_task==3 or infer_task==5:
                #loop calling, resolution enhancement, scHi-C enhancement
                cur_output = cur_output[:row_end-row_start,:col_end-col_start]
                cur_output = array_to_coo(cur_output)
                output_dict[chr]["row_record"].extend(cur_output.row+row_start)
                output_dict[chr]["col_record"].extend(cur_output.col+col_start)
                output_dict[chr]["value_record"].extend(cur_output.data)
                output_dict[chr]["count_record"].extend([1]*len(cur_output.data))
            elif infer_task==4:
                #epigenomic assay prediction
                cur_output = cur_output[:, :row_end-row_start]
                output_dict[chrom]['mean'][:, row_start:row_end] += cur_output
                output_dict[chrom]['count'][:, row_start:row_end] += 1
                



    
    if infer_task==1:
        return output_dict
    elif infer_task==2 or infer_task==3 or infer_task==5:
        final_dict=output_dict
        output_dict={}
        for chrom in output_dict:
            row_record = np.concatenate(final_dict[chrom]["row_record"])
            col_record = np.concatenate(final_dict[chrom]["col_record"])
            value_record = np.concatenate(final_dict[chrom]["value_record"])
            count_record = np.concatenate(final_dict[chrom]["count_record"])
            combine_row=np.concatenate([row_record,col_record])
            combine_col=np.concatenate([col_record,row_record])
            combine_value=np.concatenate([value_record,value_record])
            combine_count=np.concatenate([count_record,count_record])
            prediction_sym = coo_matrix((combine_value, (combine_row, combine_col)), shape=dataset_shape_dict[chrom])
            count_sym = coo_matrix((combine_count, (combine_row, combine_col)), shape=dataset_shape_dict[chrom])
            
            prediction_sym.sum_duplicates()
            count_sym.sum_duplicates()
            prediction_sym.data = prediction_sym.data/count_sym.data
            #remove very small prediction to save time
            select_index = prediction_sym.data>0.01
            prediction_sym.data = prediction_sym.data[select_index]
            prediction_sym.row = prediction_sym.row[select_index]
            prediction_sym.col = prediction_sym.col[select_index]
            print("finish summarize %s prediction"%chrom,prediction_sym.nnz)
            output_dict[chrom] = triu(prediction_sym,0)
        return output_dict
    elif infer_task==4:
        return_dict={}
        for chrom in dataset_shape_dict:
            count_array=output_dict[chrom]['count']
            mean_array=output_dict[chrom]['mean']
            count_array =np.maximum(count_array,1)
            mean_array = mean_array/count_array
            return_dict[chrom] = mean_array
        return return_dict