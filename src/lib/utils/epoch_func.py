import os
import sys
import pathlib
import time
import collections
import itertools
import shutil
import pickle
import inspect
import json
import subprocess
import logging
import numpy as np
import torch
import torchvision
import lib.utils.average_meter as average_meter
import lib.utils.simclr_utils as simclr_utils
import lib.evaluation.few_shot_eval as few_shot_eval
from tqdm import tqdm
from sklearn import linear_model
try:
    from apex import amp
except ImportError:
    pass


def make_directory(path):
    if os.path.exists(str(path)) is False:
        os.makedirs(str(path))
        

def iter_func(wrappered_model, data, args, meter, since, optimizer=None):
    is_train = optimizer is not None
    if is_train:
        optimizer.zero_grad()

    meter.add_value("time_data", time.time() - since)
    input_x = data["data"]
    since = time.time()
    output = wrappered_model(input_x)

    loss, output = output
    if is_train:
        meter.add_value("time_f", time.time()-since)
        since = time.time()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        # loss.backward()
        else:
            loss.backward()
        optimizer.step()

        meter.add_value("time_b", time.time()-since)

    meter.add_value("loss_total", loss)


def train_epoch(wrappered_model, train_loader, optimizer, epoch, args, logger=None):
    wrappered_model.train()
    meter = average_meter.AverageMeter()

    iter_num = len(train_loader)
    iter_since = time.time()
    since = iter_since

    # all_logits = []
    for batch_idx, data in enumerate(train_loader):
        iter_func(wrappered_model, data, args, meter, since, optimizer=optimizer)
        if args.LOG.train_print and (batch_idx+1) % args.LOG.train_print_iter == 0:
            # current training accuracy
            time_cur = (time.time() - iter_since)
            meter.add_value("time_iter", time_cur)
            iter_since = time.time()

            msg = f"Epoch [{epoch}] [{batch_idx+1}]/[{iter_num}]\t"
            summary = meter.get_summary()
            for name, value in summary.items():
                msg += " {}:{:.6f} ".format(name, value)
            logger.info(msg)

        if args.debug:
            if batch_idx >= 5:
                break
        since = time.time()

    # all_logits = np.concatenate(all_logits)
    train_info = meter.get_summary()

    return train_info


def valid_epoch(wrappered_model, train_loader, epoch, args, logger=None):
    wrappered_model.eval()

    meter = average_meter.AverageMeter()
    iter_num = len(train_loader)
    iter_since = time.time()
    since = iter_since

    with torch.no_grad():
        # all_logits = []
        for batch_idx, data in tqdm(enumerate(train_loader), total=iter_num, desc="validation"):
            iter_func(wrappered_model, data, args, meter, since)
            if args.debug:
                if batch_idx >= 5:
                    break
            since = time.time()
        # infer_data = torch.cat(infer_data, dim=0)[:save_num]
        # all_logits = np.concatenate(all_logits)
    train_info = meter.get_summary()
    # inference(wrappered_model.model, infer_data, args, epoch)

    return train_info
