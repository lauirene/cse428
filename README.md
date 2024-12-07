# HiCFoundation

<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/HiCFoundation-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
</a>  

HiCFoundation is a generalizable Hi-C foundation model for chromatin architecture, multi-species, single-cell and multi-omics analysis.

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

