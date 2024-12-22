import os
import numpy as np
import torch
import torch.utils.data
import random
from collections import defaultdict
from scipy.sparse import coo_matrix
import pickle 
from ops.sparse_ops import array_to_coo
from ops.io_utils import load_pickle
from data_processing.finetune_dataset import to_tensor, list_to_tensor

def validate_input_size(input_matrix, window_height, window_width):
    """
    Validate the input size is larger than the window size
    Args:
        input_matrix: the input matrix
        window_height: the height of the window
        window_width: the width of the window
    """
    if isinstance(input_matrix, coo_matrix):
        input_matrix = input_matrix.toarray()
    input_height, input_width = input_matrix.shape
    if input_height>=window_height and input_width>=window_width:
        #this validation is different from fine-tuning since we can crop the input to self-supervise
        return True
    return False 

def sample_index(matrix_size,window_size):
    """
    Sample the index of the window
    Args:
        matrix_size: the size of the matrix
        window_size: the size of the window
    """
    if matrix_size==window_size:
        return 0
    start = random.randint(0, matrix_size-window_size-1)
    return start

def sample_index_patch(matrix_size,window_size,patch_size):
    """
    Please choose this version if you want to use hi-c processed data pipeline
    The generated patch make sure the diagonal region only starts at the multiple of patch_size
    Sample the index of the window only in patch separation
    Args:
        matrix_size: the size of the matrix
        window_size: the size of the window
        patch_size: the size of the patch
    """
    if matrix_size==window_size:
        return 0
    patch_list=[]
    for i in range(0,matrix_size-window_size,patch_size):
        patch_list.append(i)
    start = random.choice(patch_list)
    return start
class Pretrain_Dataset(torch.utils.data.Dataset):
    def __init__(self,data_list,   
                transform=None,
                sparsity_filter=0.05,
                patch_size=16,
                window_height= 224,
                window_width = 224):
        """
        Args:
            data_list: the list of data
            transform: the transformation function
            sparsity_filter: the sparsity ratio to filter too sparse data for pre-training
            window_height: the height of the window
            window_width: the width of the window
        """
        self.data_list = data_list
        self.transform = transform
        self.window_height = window_height
        self.window_width = window_width
        self.sparsity_filter = sparsity_filter
        self.patch_size = patch_size
        self.train_dict=defaultdict(list)
        self.train_list=[]
        for data_index, data_dir in enumerate(data_list):
            cur_dir = data_dir
            dataset_name = os.path.basename(cur_dir)
            listfiles = os.listdir(cur_dir)
            for file_index,file in enumerate(listfiles):
                cur_path = os.path.join(cur_dir, file)
                if file.endswith('.pkl'):
                    if file_index==0:
                        #verify the input pkl file includes the input key
                        data= load_pickle(cur_path)
                        data_keys = list(data.keys())
                        if 'input' not in data:
                            print("The input key is not included in the pkl file. The directory is skipped.")
                            print("The dir is {}".format(cur_dir))
                            continue
                        #check input_count key
                        # if 'input_count' not in data:
                        #     print("The input_count key is not included in the pkl file. The directory is skipped.")
                        #     print("The dir is {}".format(cur_dir))
                        #     continue
                        
                        #validate the input size
                        input_matrix = data['input']
                        if not validate_input_size(input_matrix, window_height, window_width):
                            print("The input size is not matched with the window size. The directory is skipped.")
                            print("The dir is {}".format(cur_dir))
                            print("The input size is {}".format(input_matrix.shape))
                            print("The specified window size is {} x {}".format(window_height, window_width))
                            print("Please adjust --input_row_size and --input_col_size to match your input.")
                            continue
                    self.train_dict[dataset_name].append(cur_path)
                    self.train_list.append(cur_path)
                else:
                    print("The file {} is not a .pkl file.".format(file),"It is skipped.")
                    continue    
        print("The number of samples used in the dataset is {}".format(len(self.train_list)))
    #you can either select the train_list or train_dict to do training based on your exprience
    def __len__(self):
        return len(self.train_list)
    
    def convert_rgb(self,data_log,max_value):
        if len(data_log.shape)==2:
            data_log = data_log[np.newaxis,:]
        data_red = np.ones(data_log.shape)
        data_log1 = (max_value-data_log)/max_value
        data_rgb = np.concatenate([data_red,data_log1,data_log1],axis=0,dtype=np.float32)#transform only accept channel last case
        data_rgb = data_rgb.transpose(1,2,0)
        return data_rgb
    
    def __getitem__(self, idx):
        """
        Args:
            idx: the index of the data
        """
        data_path = self.train_list[idx]
        data= load_pickle(data_path)
        input_matrix = data['input']
        region_size =self.window_height*self.window_width
        if isinstance(input_matrix, coo_matrix):
            cur_sparsity = input_matrix.nnz/region_size
        else:
            cur_sparsity = np.count_nonzero(input_matrix)/region_size
        #we suggest you processed the submatrix to make sure they pass the threshold, otherwise it may take much longer to iteratively sampling until passing the threshold
        if cur_sparsity<self.sparsity_filter:
            random_index = random.randint(0, len(self.train_list)-1)
            return self.__getitem__(random_index)
        
        if isinstance(input_matrix, coo_matrix):
            input_matrix = input_matrix.toarray()
            #make sure you save the down-diagonal regions if you use the coo_matrix
            #to support off-diagonal submatrix, we did not any automatic symmetrical conversion for your input array.

        input_matrix = np.nan_to_num(input_matrix)
        
        if 'input_count' in data:
            matrix_count = np.sum(input_matrix)
            hic_count = data['input_count']
        else:
            matrix_count = None
            hic_count = None
        

        submat = np.zeros([1,self.window_height,self.window_width])

        
        row_start = sample_index(input_matrix.shape[0],self.window_height)
        col_start = sample_index(input_matrix.shape[1],self.window_width)
            
        row_end = min(row_start+self.window_height,input_matrix.shape[0])
        col_end = min(col_start+self.window_width,input_matrix.shape[1])
        submat[0,0:row_end-row_start,0:col_end-col_start] = input_matrix[row_start:row_end,col_start:col_end] 
        submat = submat.astype(np.float32)
        mask_array = np.ones(submat.shape,dtype=np.float32)
        mask_array[submat==0]=0
        mask_array = mask_array[np.newaxis,:,:]
        input = submat
        max_value = np.max(input)
        input = np.log(input+1)
        max_value = np.log(max_value+1)
        input = self.convert_rgb(input,max_value)
        if self.transform is not None:
            input = self.transform(input)
        return list_to_tensor([input, mask_array, hic_count, matrix_count])