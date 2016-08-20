#!/usr/bin/env python
from __future__ import division
from scipy import sparse
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time


def generate_feature(train_file, test_file, app_event_file, app_label_file, event_file, label_file,
                     phone_file, train_feature_file, test_feature_file):
    """Generate features based on Dune Dweller's script."""

    logging.info('loading raw data files')
    trn = pd.read_csv(train_file, index_col='device_id', usecols=['device_id', 'group'])
    tst = pd.read_csv(test_file, index_col='device_id')
    app_event = pd.read_csv(app_event_file, usecols=['event_id', 'app_id', 'is_active'],
                            dtype={'is_active': bool})
    app_label = pd.read_csv(app_label_file)
    event = pd.read_csv(event_file, parse_dates=['timestamp'], index_col='event_id')
    label = pd.read_csv(label_file, index_col='label_id')
    phone = pd.read_csv(phone_file)

    logging.info('removeing gender and age from training, set groups to 0 for test')
    tst['group'] = 0

    logging.info('label-encoding group in training data')
    trn.group = LabelEncoder().fit_transform(trn.group)

    logging.info('adding row ids to training and test data')
    trn['row_id_trn'] = np.arange(trn.shape[0])
    tst['row_id_tst'] = np.arange(tst.shape[0])
    logging.info('training data frame:\n{}'.format(trn.head()))
    logging.info('test data frame:\n{}'.format(trn.head()))

    logging.info('combining training and test data')
    df = pd.concat([trn, tst], axis=0)


    logging.info('removing duplicates from phone data')
    phone = phone.drop_duplicates('device_id', keep='first').set_index('device_id')

    logging.info('label-encodeing phone brand and device model')
    phone.ix[:, 'phone_brand'] = LabelEncoder().fit_transform(phone.phone_brand)
    logging.debug('# phone_brand: {}'.format(phone.phone_brand.nunique()))

    phone.ix[:, 'device_model'] = LabelEncoder().fit_transform(phone.device_model)
    logging.debug('# device_model: {}'.format(phone.device_model.nunique()))

    logging.info('phone data frame:\n{}'.format(phone.head()))

    logging.info('joining with phone data')
    trn = pd.merge(trn, phone, left_index=True, right_index=True, how='left')
    tst = pd.merge(tst, phone, left_index=True, right_index=True, how='left')

    X_brand_trn = sparse.csr_matrix((np.ones(trn.shape[0]),
                                     (trn.row_id_trn, trn.phone_brand)))
    X_brand_tst = sparse.csr_matrix((np.ones(tst.shape[0]),
                                     (tst.row_id_tst, tst.phone_brand)))
    logging.debug('phone brand data: train: {}, test: {}'.format(X_brand_trn.shape, X_brand_tst.shape))

    X_model_trn = sparse.csr_matrix((np.ones(trn.shape[0]),
                                     (trn.row_id_trn, trn.device_model)))
    X_model_tst = sparse.csr_matrix((np.ones(tst.shape[0]),
                                     (tst.row_id_tst, tst.device_model)))
    logging.debug('device model data: train: {}, test: {}'.format(X_model_trn.shape, X_model_tst.shape))


    logging.info('removing app labels not associated with app ids in app_event')
    app_label = app_label.loc[app_label.app_id.isin(app_event.app_id.unique())]

    logging.info('label-encoding app_id in app_event')
    lbe_app_id = LabelEncoder()
    app_event.ix[:, 'app_id'] = lbe_app_id.fit_transform(app_event.app_id)
    logging.debug('# app_id: {}'.format(app_event.app_id.nunique()))

    logging.info('joining app event data with event data to get device ids')
    app_event = pd.merge(app_event, event[['device_id']], left_on='event_id', right_index=True, how='left')

    logging.info('joining app event data with training and test row ids')
    device_app_event = (app_event.groupby(['device_id', 'app_id'])['app_id']
                                 .agg(['size'])
                                 .merge(trn[['row_id_trn']], how='left', left_index=True, right_index=True)
                                 .merge(tst[['row_id_tst']], how='left', left_index=True, right_index=True)
                                 .reset_index())
    logging.debug('device_app_event:\n{}'.format(device_app_event.head()))
    device_app_event.columns = ['device_id', 'app_id', 'n_app_event', 'row_id_trn', 'row_id_tst']
    device_app_event_trn = device_app_event.dropna(subset=['row_id_trn'])
    device_app_event_tst = device_app_event.dropna(subset=['row_id_tst'])

    n_app = len(lbe_app_id.classes_)
    X_app_event_trn = sparse.csr_matrix((np.log2(1 + device_app_event_trn.n_app_event),
                                         (device_app_event_trn.row_id_trn,
                                          device_app_event_trn.app_id)),
                                        shape=(trn.shape[0], n_app))
    X_app_event_tst = sparse.csr_matrix((np.log2(1 + device_app_event_tst.n_app_event),
                                         (device_app_event_tst.row_id_tst,
                                          device_app_event_tst.app_id)),
                                        shape=(tst.shape[0], n_app))
    logging.debug('app event data: train: {}, test: {}'.format(X_app_event_trn.shape, X_app_event_tst.shape))


    logging.info('label-encoding app_id and label_id in app_label')
    app_label.ix[:, 'app_id'] = lbe_app_id.transform(app_label.app_id)
    lbe_label = LabelEncoder()
    app_label.ix[:, 'label_id'] = lbe_label.fit_transform(app_label.label_id)

    logging.info('joining app_label with app event data above to get device ids')
    device_app_label = pd.merge(device_app_event[['device_id', 'app_id']], app_label[['app_id', 'label_id']])
    logging.debug('device_app_label:\n{}'.format(device_app_label.head()))

    device_label = (device_app_label.groupby(['device_id', 'label_id'])['label_id'].agg(['size'])
                                    .merge(trn[['row_id_trn']], how='left', left_index=True, right_index=True)
                                    .merge(tst[['row_id_tst']], how='left', left_index=True, right_index=True)
                                    .reset_index())
    logging.debug('device_label:\n{}'.format(device_label.head()))
    device_label.columns = ['device_id', 'label_id', 'n_app_label', 'row_id_trn', 'row_id_tst']
    device_label_trn = device_label.dropna(subset=['row_id_trn'])
    device_label_tst = device_label.dropna(subset=['row_id_tst'])

    n_label = len(lbe_label.classes_)
    X_app_label_trn = sparse.csr_matrix((np.log2(1 + device_label_trn.n_app_label),
                                         (device_label_trn.row_id_trn,
                                          device_label_trn.label_id)),
                                        shape=(trn.shape[0], n_label))
    X_app_label_tst = sparse.csr_matrix((np.log2(1 + device_label_tst.n_app_label),
                                         (device_label_tst.row_id_tst,
                                          device_label_tst.label_id)),
                                        shape=(tst.shape[0], n_label))
    logging.debug('app label data: train: {}, test: {}'.format(X_app_label_trn.shape, X_app_label_tst.shape))


    logging.info('combining all features - phone brand, device model, app_event, app_label')
    X_trn = sparse.hstack((X_brand_trn, X_model_trn, X_app_event_trn, X_app_label_trn), format='csr')
    X_tst = sparse.hstack((X_brand_tst, X_model_tst, X_app_event_tst, X_app_label_tst), format='csr')
    logging.debug('all features: train: {}, test: {}'.format(X_trn.shape, X_tst.shape))

    logging.info('saving as libSVM format')
    dump_svmlight_file(X_trn, trn.group, train_feature_file, zero_based=False)
    dump_svmlight_file(X_tst, tst.group, test_feature_file, zero_based=False)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--app-event-file', required=True, dest='app_event_file')
    parser.add_argument('--app-label-file', required=True, dest='app_label_file')
    parser.add_argument('--event-file', required=True, dest='event_file')
    parser.add_argument('--label-file', required=True, dest='label_file')
    parser.add_argument('--phone-file', required=True, dest='phone_file')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')

    args = parser.parse_args()

    start = time.time()
    generate_feature(args.train_file,
                     args.test_file,
                     args.app_event_file,
                     args.app_label_file,
                     args.event_file,
                     args.label_file,
                     args.phone_file,
                     args.train_feature_file,
                     args.test_feature_file)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))

