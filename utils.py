import sys
import os
import imageio
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import datetime
from glob import glob
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

from PyQt5.QtWidgets import (QFileDialog, QAbstractItemView, QListView,
                             QTreeView, QApplication, QDialog)

# File directory dialog for selecting multiple directories.
# See usage in plotting functions.

class getExistingDirectories(QFileDialog):
    def __init__(self, *args):
        super(getExistingDirectories, self).__init__(*args)
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.Directory)
        self.setOption(self.ShowDirsOnly, True)
        self.findChildren(QListView)[0].setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.findChildren(QTreeView)[0].setSelectionMode(QAbstractItemView.ExtendedSelection)


# Tensorboard image/scalar extraction

def tb_extract(model_path=None, keys=['Loss'], anim=False):

    if model_path is None:
        qapp = QApplication(sys.argv)
        dirs = getExistingDirectories()
        if dirs.exec_() == QDialog.Accepted:
            model_path = dirs.selectedFiles()
            # print(model_path)
    elif 'Pycharm' not in model_path:
        model_path = os.path.join('/home/omar/PycharmProjects/mlmOK_Pytorch/models', model_path)

    for model in model_path:
        event_acc = event_accumulator.EventAccumulator(
            model, size_guidance={'images': 0})
        event_acc.Reload()

        outdir = pathlib.Path(model+'/out')
        outdir.mkdir(exist_ok=True, parents=True)

        for key in keys:
            if key == 'Loss':
                for tag in event_acc.Tags()['scalars']:
                    tag_name = tag.replace('/', '_')
                    pd.DataFrame(event_acc.Scalars(tag)).to_csv(model+'/out/'+tag_name+'.csv')
                    f = plt.figure(figsize=(12, 6))
                    # print(pd.DataFrame(event_acc.Scalars(tag)))
                    plt.plot(pd.DataFrame(event_acc.Scalars(tag)).values[:,1],
                             pd.DataFrame(event_acc.Scalars(tag)).values[:,2])
                    plt.xlabel('Epoch')
                    plt.ylabel(tag_name)
                    plt.yscale('log')
                    plt.savefig(model+'/out/'+tag_name+'.png')
            for tag in event_acc.Tags()['images']:

                if key in tag:
                    events = event_acc.Images(tag)

                    tag_name = tag.replace('/', '_')
                    dirpath = outdir / tag_name
                    dirpath.mkdir(exist_ok=True, parents=True)

                    for index, event in enumerate(events):
                        s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
                        image = cv2.imdecode(s, cv2.IMREAD_COLOR)
                        outpath = dirpath / '{:04}.jpg'.format(index)
                        cv2.imwrite(outpath.as_posix(), image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

                    if anim:
                        jpg_dir = dirpath
                        images = []
                        for file_name in os.listdir(jpg_dir):
                            if file_name.endswith('.jpg'):
                                file_path = os.path.join(jpg_dir, file_name)
                                images.append(imageio.imread(file_path))
                        imageio.mimsave(os.path.join(jpg_dir, tag_name + '.gif'), images, fps=2)


def compare_truth_pred(pred_file, truth_file):
    """
    Read truth and pred from csv files, compute their mean-absolute-error and the mean-squared-error
    :param pred_file: full path to pred file
    :param truth_file: full path to truth file
    :return: mae and mse
    """
    pred = np.loadtxt(pred_file, delimiter=' ')
    truth = np.loadtxt(truth_file, delimiter=' ')
    print("in compare truth pred function in eval_help package, your shape of pred file is", np.shape(pred))
    # This is for the edge case of ballistic, where y value is 1 dimensional which cause dimension problem
    if len(np.shape(pred)) == 1:
        pred = np.reshape(pred, [-1, 1])
        truth = np.reshape(truth, [-1, 1])
    mae = np.mean(np.abs(pred - truth), axis=1)
    mse = np.mean(np.square(pred - truth), axis=1)

    return mae, mse

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches





