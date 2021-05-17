import os
import pickle
import sys
from datetime import datetime

from scipy import sparse as sp
from sklearn.model_selection import ParameterGrid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.abspath('../../'))
from ease import EASE
from conf import UN_SEEDS, TRAITS, UN_LOG_VAL_STR, UN_LOG_TE_STR, DATA_PATH, PERS_PATH, UN_OUT_DIR
from utils.data_splitter import DataSplitter
from utils.eval import eval_proced

print('STARTING UNCONTROLLED EXPERIMENTS WITH EASE')
print('SEEDS ARE: {}'.format(UN_SEEDS))

grid = {
    'lam': [1, 1e1, 1e2, 5e2, 1e3, 1e4, 1e5, 1e6, 1e7]
}
pg = ParameterGrid(grid)

now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

for seed in tqdm(UN_SEEDS, desc='seeds'):