import math
import sys
import numpy as np
from typing import Iterable
import torch
import torch.nn as nn
import time
from ops.Logger import MetricLogger,SmoothedValue
import os

def inference_worker(model,data_loader,log_dir=None,args=None):
    """
    model: model for inference
    data_loader: data loader for inference
    log_dir: log directory for inference
    args: arguments for inference
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Eval: '
    print_freq = args.print_freq
    print("number of iterations: ",len(data_loader))
    num_iter = len(data_loader)