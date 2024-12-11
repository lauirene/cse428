# HiCFoundation

<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/HiCFoundation-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
</a>  

HiCFoundation is a generalizable Hi-C foundation model for chromatin architecture, single-cell and multi-omics analysis across species.

Copyright (C) 2024 Xiao Wang, Yuanyuan Zhang, Suhita Ray, Anupama Jha, Tangqi Fang, Shengqi Hang, Sergei Doulatov, William Stafford Noble, and Sheng Wang

License: Apache License 2.0

Contact:  Sergei Doulatov (doulatov@uw.edu) & William Stafford Noble (wnoble@uw.edu) & Sheng Wang (swang@cs.washington.edu)

For technical problems or questions, please reach to Xiao Wang (wang3702@uw.edu) and Yuanyuan Zhang (zhang038@purdue.edu).

## Citation:
Xiao Wang, Yuanyuan Zhang, Suhita Ray, Anupama Jha, Tangqi Fang, Shengqi Hang, Sergei Doulatov, William Stafford Noble, & Sheng Wang. A generalizable Hi-C foundation model for chromatin architecture, single-cell and multi-omics analysis across species. bioRxiv, 2024. [Paper]()
<br>
```
@article{wang2024hicfoundation,   
  title={A generalizable Hi-C foundation model for chromatin architecture, single-cell and multi-omics analysis across species},   
  author={Xiao Wang, Yuanyuan Zhang, Suhita Ray, Anupama Jha, Tangqi Fang, Shengqi Hang, Sergei Doulatov, William Stafford Noble, and Sheng Wang},    
  journal={bioRxiv},    
  year={2024}    
}   
```

## Introduction

<details>
   <summary>HiCFoundation is a generalizable Hi-C foundation model for chromatin architecture, single-cell and multi-omics analysis across species. </summary>
The genetic information within nuclear DNA is organized into a compact three-dimensional (3D) structure that impacts critical cellular processes.
High-throughput chromosome conformation capture (Hi-C) stands as the most widely used method for measuring 3D genome architecture, while linear epigenomic assays, such as ATAC-seq, DNase-seq, and ChIP-seq, are extensively employed to characterize genome regulatory activities.
However, the integrative analysis of chromatin interactions and associated gene regulatory mechanisms remains challenging due to the mismatched resolution between Hi-C and epigenomic assays, as well as inconsistencies among analysis tools.
Here we propose HiCFoundation, a Hi-C-based foundation model for genome architecture and regulatory functions analysis. 
HiCFoundation is trained from hundreds of Hi-C assays encompassing 118 million contact matrix patches. 
The model achieves state-of-the-art performance in multiple types of 3D genome analysis, including reproducibility analysis, resolution enhancement, and loop detection, offering high efficiency and broad applicability. 
We further demonstrate the model's generalizability to genome architecture analysis of 316 species.
Notably, by enabling analysis of low-coverage experimental data, HiCFoundation reveals genome-wide loop loss during differentiation of HSPCs to neutrophil. 
Additionally, HiCFoundation is able to predict multiple gene regulatory activities from Hi-C input by generating epigenomic assays, and further offers interpretable analysis to reveal the relationship between chromatin conformation and genome function. 
Finally, HiCFoundation can analyze single cell Hi-C data, shedding light on genome structure at single-cell resolution.
HiCFoundation thus provides a unified, efficient, generalizable, and interpretable foundation for integrative, multi-species, single-cell, and multi-omics analyses, paving the path for systematically studying genome 3D architecture and its regulatory mechanisms.

</details>

## Overall Protocol 
<details>
<br>
1) Pre-training stage: the model is trained in a self-supervised fashion on massive quantities of unlabeled Hi-C data. 
The model takes masked Hi-C submatrices as input, optimizing for the reconstruction of the full submatrix.
<br>
2) Fine-tuning stage: the model is fine-tuned and tested for diverse downstream tasks, including integrative Hi-C analysis, multi-omics analysis, and single-cell analysis.

<p align="center">
  <img src="imgs/github_v2.png" alt="HiCFoundation framework" width="80%">
</p>
</details>

## Installation

<details>

### System Requirements
- **CPU**: 4 cores or higher
- **Memory**: 12GB RAM or higher
- **GPU**: CUDA-compatible with minimum 12GB memory
- **Note**: GPU is mandatory as HiCFoundation

## Installation  
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```bash
git clone https://github.com/Noble-Lab/HiCFoundation.git && cd HiCFoundation
```

### 3. Configure environment for HiCFoundation.
##### 3.1 Install anaconda
Install anaconda from https://www.anaconda.com/download#downloads.
##### 3.2 Install environment via yml file
```bash
conda env create -f environment.yml
```
##### 3.3 Activate environment for running
Each time when you want to run HiCFoundation, simply activate the environment by
```bash
conda activate HiCFoundation
# To exit
conda deactivate
```

### 4. Download the trained HiCFoundation model
You can download our pre-trained and fine-tuned model to ``hicfoundation_model`` for inference, embedding generation and fine-tuning purposes. <br>
HiCFoundation model weights: [hicfoundation_model]() <br>

You can also use command line to do this
```commandline
cd hicfoundation_model
wget 
cd ..
```

If the link failed, you can also download our model files via our [lab server]() to ``hicfoundation_model`` directory. 

### 5. (Optional) Visualization software
Juicebox: https://aidenlab.org/juicebox/

HiGlass: https://higlass.io/

</details>

# Usage

<details>
<summary>Inference of fine-tuned HiCFoundation</summary>

## Overview
This include five different fine-tuned model for 
- Reproducibility analysis: HiCFoundation will generate embeddings of the input Hi-C, and the submatrix embeddings can be used to compare across biological replicates and non-replicates.
- Chromatin loop detection: HiCFoudation will generate the loop detection of the input Hi-C in .bedpe format.
- Resolution enhancement: HiCFoundation will generate enhanced Hi-C map given the input Hi-C.
- Epigenomic assay profiling: HiCFoundation will generate corressponding epigenomic assays in .bigWig format given the input Hi-C.
- Single-cell Hi-C enhancement: HiCFoundation will generate the enhanced scHi-C given the input siHi-C.

## Input format
HiCFoundation supports the .hic/.cool/.pkl/.txt/.pairs/.npy format.
- .hic/.cool: the common Hi-C format that stores the final matrix of Hi-C experiment
- .pkl: the pickle file that stores a dict of all Hi-C matrices, with the chrom name as key, and scipy.sparse/numpy array as the value. [chrom_name]:[matrix].
- .txt/.pairs: the pairs format text that records pairwise interactions in pairs format "#readID\tchr1\tpos1\tchr2\tpos2" that records the chr1:pos1 interactions with chr2:pos2.
- .npy format: a numpy array that records the contact map of a specific chromosome.

## Example
Please download the following files to the example folder for example testing purposes.<br>
- Low coverage Hi-C example: https://www.encodeproject.org/files/ENCFF689CUX/@@download/ENCFF689CUX.hic.
- High coverage Hi-C example: https://data.4dnucleome.org/files-processed/4DNFITUOMFUQ/. (4DN requires authentication in for downloading, so please download in the webpage)
- Single-cell Hi-C example: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7006527 
(For single-cell Hi-C example, it is already kept in ``example`` directory, so you do not need to downlaod again.)

### Other format examples
- .cool: https://data.4dnucleome.org/files-processed/4DNFI18UHVRO/ (4DN requires authentication in for downloading, so please download in the webpage)
- .txt/.pairs: [example/input.pairs](example/GSM7006527_ValaB8w4191.pairs) 
- .pkl: You can run [utils/hic2array.py](utils/hic2array.py) to convert .hic files to .pkl files to see .pkl format.
- .npy: You can use [numpy](https://numpy.org/) to save any 2D numpy file to .npy file to run our inference. 


## Inference for different tasks
### 1. Inference embeddings for reproducibility analysis
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
- gpu: which gpu to use, default: None (will use all GPU). You can specify --gpu="0" to only use GPU 0, you can also specify --gpu="0,1" to use GPU0 and GPU1.
<br>
The output is saved in the ``output_dir``, where the embedding is saved in "HiCFoundation_reproducibility_embedding.pkl" in a dict format. <br>
The key of the dict is "chrom:row_index,col_index", and the value is the corresponding embedding. <br>
This embedding corresponds to the submatrix of [row_index:row_index+input_row_size, col_index:col_index+input_col_size] at chromsome ``chrom``.

Example command:
```
python3 inference.py --input example/ENCFF689CUX.hic --batch_size 4 --resolution 25000 --task 1 --input_row_size 224 --input_col_size 224 --stride 20 --bound 0 --model_path hicfoundation_model/hicfoundation_reproducibility.pth.tar --output hicfoundation_inference/reproducibility_analysis/ --gpu "0"
```
This uses the low-coverage example ``ENCFF689CUX.hic`` to run the inference. <br>
The output embedding is saved in ``hicfoundation_inference/reproducibility_analysis/HiCFoundation_reproducibility_embedding.pkl``.

<details>
