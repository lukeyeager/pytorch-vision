"""Loads a dataset from a tarfile.

Behavior should be identical to the ImageFolder loader,
except that it loads from a tarfile instead of from a directory.
"""
from __future__ import absolute_import

from multiprocessing import Lock
import os
import tarfile
import warnings

import PIL.Image

from . import folder as folder_dataset
from torch.utils.data import Dataset

class TarFile(Dataset):
    def __init__(self, filename, transform=None, target_transform=None):
        self.tarfile = tarfile.open(filename)
        self.transform = transform
        self.target_transform = target_transform

        # Read entries from the tarfile (skip reading the actual data)
        self.entries = []
        labels_set = set()
        for member in self.tarfile.getmembers():
            if member.isfile():
                if not folder_dataset.is_image_file(member.name):
                    warnings.warn('Skipping non-image file "%s"' % member.name, Warning)
                    continue

                # Read class label
                filename_sections = member.name.split(os.path.sep)
                if len(filename_sections) == 1:
                    warnings.warn('Skipping top-level file "%s"' % member.name, Warning)
                    continue

                label = filename_sections[0]
                labels_set.add(label)
                self.entries.append( (member, label) )

        self.labels_dict = dict([(name, i) for i, name in enumerate(labels_set)])

        # Since tarfile is not thread-safe (https://bugs.python.org/issue23649)
        self.lock = Lock()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        member, label_str = self.entries[index]

        # Need to acquire a lock when calling extractfile()
        # And also when actually reading the resulting fileobj
        self.lock.acquire()
        img = PIL.Image.open(self.tarfile.extractfile(member))
        img.load()
        self.lock.release()

        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        target = self.labels_dict[label_str]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
