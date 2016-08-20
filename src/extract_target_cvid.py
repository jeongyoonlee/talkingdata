#!/usr/bin/env python
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
import argparse
import numpy as np
import pandas as pd

from const import SEED


def extract_target_cvid(train_file, target_file, cvid_file):
    trn = pd.read_csv(train_file, index_col='device_id')

    lbe = LabelEncoder()
    y = lbe.fit_transform(trn.group)
    np.savetxt(target_file, y, fmt='%d')
    
   
    cv = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=SEED)
    cv_id = np.zeros_like(y, dtype=int)
    for i, (i_trn, i_val) in enumerate(cv, 1):
        cv_id[i_val] = i
    
    np.savetxt(cvid_file, cv_id, fmt='%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--target-file', required=True, dest='target_file')
    parser.add_argument('--cvid-file', required=True, dest='cvid_file')
    args = parser.parse_args()

    extract_target_cvid(args.train_file,
                        args.target_file,
                        args.cvid_file)
