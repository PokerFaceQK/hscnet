from __future__ import division

import sys
import os
import random
import argparse
from pathlib import Path
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from models import get_model
from datasets import get_dataset, Rio10Dataset, RIOScenes
from loss import *
from utils import *

data_path = "data/rio10/scene01"

dataset = Rio10Dataset(
            data_path=data_path,
            split='validation',
            aug=False
        )


def run_generate_label(start_idx, end_idx):
    for idx in range(start_idx, end_idx):
        dataset.generate_label(idx)


# processes = mp.cpu_count() - 2
processes = 5
per_len = len(dataset) // processes
with mp.Pool(processes=processes) as pool:
    for i in range(processes):
        if i < processes - 1:
            pool.apply_async(run_generate_label, args=(i * per_len, (i+1) * per_len))
        else:
            pool.apply_async(run_generate_label, args=(i * per_len, len(dataset)))
    pool.close()
    pool.join()
print("done!")
