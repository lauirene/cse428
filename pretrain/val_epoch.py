import math
import sys
import numpy as np
from typing import Iterable
import torch
import torch.nn.functional as F
import time

from ops.Logger import MetricLogger,SmoothedValue
import model.lr_sched as lr_sched

from ops.train_utils import list_to_device, to_value, create_image, torch_to_nparray

def val_epoch(model, data_loader,device, epoch,
                    log_writer=None,
                    args=None,flag='val'):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Val Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    print("number of iterations: ",len(data_loader))
    num_iter = len(data_loader)
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        input_matrix, mask_matrix, hic_count, return_diag,matrix_count = list_to_device(data,device=device)
        with torch.no_grad(): #to enable mixed precision training
            ssim_loss,contrastive_loss, count_pred, pred_image, mask \
              = model(input_matrix, mask_matrix, total_count=hic_count,  \
                diag=return_diag,mask_ratio=args.mask_ratio)
            matrix_count = torch.log10(matrix_count+1)
            count_pred = count_pred.flatten()
            count_loss = torch.nn.functional.mse_loss(count_pred, matrix_count)
            loss = args.loss_alpha*(ssim_loss+count_loss) + contrastive_loss
        metric_logger.update(loss=to_value(loss))
        metric_logger.update(ssim_loss=to_value(ssim_loss))
        metric_logger.update(count_loss=to_value(count_loss))
        metric_logger.update(contrastive_loss=to_value(contrastive_loss))
        torch.cuda.synchronize() # Make sure all gradients are finished computing before moving on

        if log_writer is not None and ((data_iter_step + 1) % accum_iter == 0 or data_iter_step==0):
            """ 
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalars('Loss/loss', {'%s_loss'%flag: to_value(loss)}, epoch_1000x)
            log_writer.add_scalars('Loss/ssim_loss', {'%s_loss'%flag: to_value(ssim_loss)}, epoch_1000x)
            log_writer.add_scalars('Loss/count_loss', {'%s_loss'%flag: to_value(count_loss)}, epoch_1000x)
            log_writer.add_scalars('Loss/contrastive_loss', {'%s_loss'%flag: to_value(contrastive_loss)}, epoch_1000x)
            #add visualization
            if ((data_iter_step+1)//accum_iter)%50==0 or data_iter_step==0:
                new_samples = create_image(input_matrix)
                mask_image = new_samples*(1-mask)
                pred_image = create_image(pred_image) 

                select_num = min(8,len(new_samples))
                new_samples = torch_to_nparray(new_samples.clone().detach()[:select_num])
                mask_image = torch_to_nparray(mask_image.clone().detach()[:select_num])
                pred_image = torch_to_nparray(pred_image.clone().detach()[:select_num])
                log_writer.add_images('Target_%s'%flag, new_samples, epoch_1000x)
                log_writer.add_images('Input_%s'%flag, mask_image, epoch_1000x)
                log_writer.add_images('Pred_%s'%flag, pred_image, epoch_1000x)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}