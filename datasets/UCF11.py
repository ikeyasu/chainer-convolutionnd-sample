import chainer
import os
import re
import numpy as np


def get_dir_list(path):
    return [os.path.join(path, i) for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]


LABEL_PATTERN = re.compile(r"v_([a-zA-Z_]+)_[0-9]")


def get_label(path):
    dir = os.path.split(path)[-1]
    matched = LABEL_PATTERN.match(dir)
    return matched.group(1)


class UCF11Dataset(chainer.dataset.DatasetMixin):
    """
    example:
    videos/v_biking_01_01/00001.jpg
    videos/v_biking_01_01/00002.jpg
    """

    def __init__(self, path):
        self.dir_list = get_dir_list(path)
        labels = [get_label(d) for d in self.dir_list]
        self.labels = list(set(labels))

    def __len__(self):
        return len(self.dir_list)

    def get_example(self, i):
        base = chainer.datasets.ImageDataset(os.listdir(self.dir_list[i]), root=self.dir_list[i])
        label = get_label(self.dir_list[i])

        frames = 6 #len(base)
        images = np.array([base[i] for i in range(frames)]).transpose(1, 0, 2, 3)

        # TODO: crop
        # TODO: mean
        # image -= self.mean[:, top:bottom, left:right]
        images *= (1.0 / 255.0)  # Scale to [0, 1]
        return images, self.labels.index(label)
