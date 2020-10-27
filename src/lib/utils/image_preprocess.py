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
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from tqdm import tqdm


def preprocess_with_info(img, image_wh):
    h, w, c = img.shape
    w_r = image_wh[0] / w
    h_r = image_wh[1] / h

    if h_r > w_r:
        resize_ratio = w_r
        resize_other_length = int(h * resize_ratio)
        resize_wh = (image_wh[0], resize_other_length)
        diff = image_wh[1] - resize_other_length
        if diff % 2 == 0:
            pad_length = diff // 2
            pad = [[pad_length, pad_length], [0, 0], [0, 0]]
            resize_wh_range = [[0, image_wh[0]], [
                pad_length, image_wh[1]-pad_length]]

        else:
            pad_length = diff // 2
            pad = [[pad_length+1, pad_length], [0, 0], [0, 0]]
            resize_wh_range = [[0, image_wh[0]], [
                pad_length+1, image_wh[1]-pad_length]]

    else:
        resize_ratio = h_r
        resize_other_length = int(w*resize_ratio)
        resize_wh = (resize_other_length, image_wh[1])
        diff = image_wh[0] - resize_other_length
        if diff % 2 == 0:
            pad_length = diff // 2
            pad = [[0, 0], [pad_length, pad_length], [0, 0]]
            resize_wh_range = [
                [pad_length, image_wh[0]-pad_length], [0, image_wh[1]]]

        else:
            pad_length = diff // 2
            pad = [[0, 0], [pad_length+1, pad_length], [0, 0]]
            resize_wh_range = [
                [pad_length+1, image_wh[0]-pad_length], [0, image_wh[1]]]

    img = cv2.resize(img, resize_wh)
    img = np.pad(img, pad, mode="constant", constant_values=0)
    meta = {
        "pad": pad,
        "resize_ratio": resize_ratio,
        "resize_wh": resize_wh,
        "org_wh": (w, h),
        "resize_wh_range": resize_wh_range
    }

    return img, meta
