from .lsun import LSUN, LSUNClass
from .folder import ImageFolder
from .coco import CocoCaptions, CocoDetection
from .cifar import CIFAR10, CIFAR100
from .mnist import MNIST
from .tarfile import TarFile

__all__ = ('LSUN', 'LSUNClass',
           'ImageFolder',
           'CocoCaptions', 'CocoDetection',
           'CIFAR10', 'CIFAR100',
           'MNIST',
           'TarFile')
