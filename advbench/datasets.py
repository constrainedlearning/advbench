import os
import torch
import sys
import glob
import h5py
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10 as CIFAR10_
from torchvision.datasets import CIFAR100 as CIFAR100_
from torchvision.datasets import MNIST as MNIST_
from torchvision.datasets import STL10 as STL10_
from torchvision.datasets import ImageFolder
from advbench.lib.transformations import Cutout
from sklearn.model_selection import train_test_split
from copy import deepcopy
import numpy as np
try:
    FFCV_AVAILABLE = os.get_env('FFCV_AVAILABLE')
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
    from ffcv.transforms.common import Squeeze
    from ffcv.writer import DatasetWriter
    print("*"*80)
    print('FFCV available. Also using Low precision operations. May result in numerical instability.')
    print("*"*80)
    FFCV_AVAILABLE=True
except:
    FFCV_AVAILABLE=False

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from advbench.lib.AutoAugment.autoaugment import CIFAR10Policy

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SPLITS = ['train', 'val', 'test']
DATASETS = ['CIFAR10', 'MNIST']
MEAN = {
    'CIFAR10': (0.4914, 0.4822, 0.4465),
    'CIFAR100': (0.5071, 0.4867, 0.4408),
    'STL10': (0.485, 0.456, 0.406),
    'IMNET': IMAGENET_DEFAULT_MEAN,
}

STD = {
    'CIFAR10': (0.2023, 0.1994, 0.2010),
    'CIFAR100': (0.2675, 0.2565, 0.2761),
    'STL10': (0.229, 0.224, 0.225),
    'IMNET': IMAGENET_DEFAULT_STD,
}

def to_loaders(all_datasets, hparams, device):
    if not all_datasets.ffcv:    
        def _to_loader(split, dataset):
            if split == 'train':
                batch_size = hparams['batch_size'] 
            elif all_datasets.TEST_BATCH is None:
                batch_size = hparams['batch_size'] 
            else:
                batch_size = all_datasets.TEST_BATCH 
            return DataLoader(
                dataset=dataset, 
                batch_size=batch_size,
                num_workers=all_datasets.N_WORKERS,
                shuffle=(split == 'train'))
    
        return [_to_loader(s, d) for (s, d) in all_datasets.splits.items()]
    else:
        loaders = []

        for split, path  in all_datasets.splits.items():           
        
            ordering = OrderOption.RANDOM if split == 'train' else OrderOption.SEQUENTIAL
            
            if split == 'train':
                batch_size = hparams['batch_size'] 
            elif all_datasets.TEST_BATCH is None:
                batch_size = hparams['batch_size'] 
            else:
                batch_size = all_datasets.TEST_BATCH

            label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
            
            loaders.append(Loader(path, batch_size=batch_size, num_workers=all_datasets.N_WORKERS,
                                order=ordering, drop_last=(split == 'train'),
                                pipelines={'image':all_datasets.transforms[split], 'label': label_pipeline}))

    return loaders

def sample_idxs(data, val_frac=0.3):
    perm = torch.randperm(len(data))
    k = int(val_frac*len(data))
    train_idx = perm[:k]
    val_idx = perm[k:]
    return train_idx, val_idx   

class AdvRobDataset(Dataset):

    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    NUM_CLASSES = None       # Subclasses should override
    N_EPOCHS = None          # Subclasses should override
    CHECKPOINT_FREQ = None   # Subclasses should override
    LOG_INTERVAL = None      # Subclasses should override
    LOSS_LANDSCAPE_INTERVAL = None # Subclasses should override
    ATTACK_INTERVAL = 200     # Default, subclass may override
    ANGLE_GSIZE = 100     # Default, subclass may override
    LOSS_LANDSCAPE_BATCHES = None # Subclasses should override
    HAS_LR_SCHEDULE = False  # Default, subclass may override
    HAS_LR_SCHEDULE_DUAL = False # Default, subclass may override
    TRANSLATIONS = [-3, 0, 3] # Default, for plotting purposes only, subclass may override
    TEST_INTERVAL = 1 # Default, subclass may override
    TEST_BATCH = None

    def __init__(self):
        self.splits = dict.fromkeys(SPLITS)

if FFCV_AVAILABLE:
    class CIFAR10(AdvRobDataset):
        ATTACK_INTERVAL = 200
        INPUT_SHAPE = (3, 32, 32)
        NUM_CLASSES = 10
        N_EPOCHS = 200
        CHECKPOINT_FREQ = 10
        LOG_INTERVAL = 100
        LOSS_LANDSCAPE_INTERVAL = 10
        LOSS_LANDSCAPE_BATCHES = 40
        ANGLE_GSIZE = 100
        HAS_LR_SCHEDULE = True

        def __init__(self, root, augmentation=True):
            super(CIFAR10, self).__init__()
            CIFAR_MEAN = [125.307, 122.961, 113.8575]
            CIFAR_STD = [51.5865, 50.847, 51.255]
            self.ffcv = True
            self.transforms = {}
            for split in ["train", "val", "test"]:
                image_pipeline = [SimpleRGBImageDecoder()]
                if split == 'train' and augmentation:
                    image_pipeline.extend([
                        RandomHorizontalFlip(),
                    ])
                image_pipeline.extend([
                    ToTensor(),
                    ToDevice('cuda:0', non_blocking=True),
                    ToTorchImage(),
                    Convert(torch.float16),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ])
                self.transforms[split] = image_pipeline

            train_data = CIFAR10_(root, train=True, download=True)
            self.splits['train'] = train_data
            # self.splits['train'] = Subset(train_data, range(5000))

            train_data = CIFAR10_(root, train=True)
            self.splits['val'] = Subset(train_data, range(45000, 50000))

            self.splits['test'] = CIFAR10_(root, train=False)
            self.write()

        @staticmethod
        def adjust_lr(optimizer, epoch, hparams):
            lr = hparams['learning_rate']
            if epoch >= 150:
                lr = hparams['learning_rate'] * 0.1
            if epoch >= 175:
                lr = hparams['learning_rate'] * 0.01
            if epoch >= 190:
                lr = hparams['learning_rate'] * 0.001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        def write(self):
            folder = os.path.join('data','ffcv', 'CIFAR')
            for (name, ds) in self.splits.items():
                fields = {
                    'image': RGBImageField(),
                    'label': IntField(),
                }
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(folder, name+'.beton')
                writer = DatasetWriter(path, fields)
                writer.from_indexed_dataset(ds)
                self.splits[name] = path
    
    class CIFAR100(AdvRobDataset):
        INPUT_SHAPE = (3, 32, 32)
        NUM_CLASSES = 100
        N_EPOCHS = 200
        CHECKPOINT_FREQ = 10
        LOG_INTERVAL = 100
        LOSS_LANDSCAPE_INTERVAL = 200
        LOSS_LANDSCAPE_GSIZE = 1000
        ANGLE_GSIZE = 100
        LOSS_LANDSCAPE_BATCHES = 10
        HAS_LR_SCHEDULE = True

        def __init__(self, root, augmentation=True):
            super(CIFAR100, self).__init__()

            self.ffcv = True
            self.transforms = {}
            for split in ["train", "val", "test"]:
                image_pipeline = [SimpleRGBImageDecoder()]
                if split == 'train' and augmentation:
                    image_pipeline.extend([
                        RandomHorizontalFlip(),
                        #Cutout(4, tuple(map(int, MEAN['CIFAR100']))),
                    ])
                image_pipeline.extend([
                    ToTensor(),
                    ToDevice('cuda:0', non_blocking=True),
                    ToTorchImage(),
                    Convert(torch.float16),
                    transforms.Normalize(MEAN['CIFAR100'], STD['CIFAR100']),
                ])
                self.transforms[split] = image_pipeline

            train_data = CIFAR100_(root, train=True, download=True)
            self.splits['train'] = train_data
            # self.splits['train'] = Subset(train_data, range(5000))

            train_data = CIFAR100_(root, train=True)
            _, val_idx = sample_idxs(train_data, val_frac=0.1)
            self.splits['val'] =  Subset(train_data, val_idx)
            self.splits['test'] = CIFAR100_(root, train=False)
            self.write()

        @staticmethod
        def adjust_lr(optimizer, epoch, hparams):
            """
            Decay initial learning rate exponentially starting after epoch_start epochs
            The learning rate is multiplied with base_factor every lr_decay_epoch epochs
            """
            lr = hparams['learning_rate']
            if epoch > hparams['lr_decay_start']:
                lr = hparams['learning_rate'] * (hparams['lr_decay_factor'] ** ((epoch - hparams['lr_decay_start']) // hparams['lr_decay_epoch']))
            print('learning rate = {:6f}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return

        def write(self):
                folder = os.path.join('data','ffcv', 'CIFAR')
                for (name, ds) in self.splits.items():
                    fields = {
                        'image': RGBImageField(),
                        'label': IntField(),
                    }
                    os.makedirs(folder, exist_ok=True)
                    path = os.path.join(folder, name+'.beton')
                    writer = DatasetWriter(path, fields)
                    writer.from_indexed_dataset(ds)
                    self.splits[name] = path

else:
    class CIFAR10(AdvRobDataset):
        INPUT_SHAPE = (3, 32, 32)
        NUM_CLASSES = 10
        N_EPOCHS = 200
        CHECKPOINT_FREQ = 10
        LOG_INTERVAL = 50
        LOSS_LANDSCAPE_INTERVAL = 100
        LOSS_LANDSCAPE_GSIZE = 1000
        ANGLE_GSIZE = 100
        LOSS_LANDSCAPE_BATCHES = 10
        HAS_LR_SCHEDULE = True

        def __init__(self, root, augmentation=True):
            super(CIFAR10, self).__init__()

            self.ffcv=False
            if augmentation:
                train_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
            else:
                train_transforms = transforms.Compose([
                    transforms.ToTensor()])
            test_transforms = transforms.ToTensor()

            train_data = CIFAR10_(root, train=True, transform=train_transforms, download=True)
            self.splits['train'] = train_data

            train_data = CIFAR10_(root, train=True, transform=train_transforms)
            _, val_idx = sample_idxs(train_data, val_frac=0.1)
            self.splits['val'] =  Subset(train_data, val_idx)

            self.splits['test'] = CIFAR10_(root, train=False, transform=test_transforms)

        @staticmethod
        def adjust_lr(optimizer, epoch, hparams):
            lr = hparams['learning_rate']
            if epoch >= 60:
                lr = hparams['learning_rate'] * 0.2
            if epoch >= 120:
                lr = hparams['learning_rate'] * 0.04
            if epoch >= 160:
                lr = hparams['learning_rate'] * 0.008
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    class CIFAR100(AdvRobDataset):
        INPUT_SHAPE = (3, 32, 32)
        NUM_CLASSES = 100
        N_EPOCHS = 200
        CHECKPOINT_FREQ = 10
        LOG_INTERVAL = 50
        LOSS_LANDSCAPE_INTERVAL = 100
        LOSS_LANDSCAPE_GSIZE = 1000
        ANGLE_GSIZE = 100
        LOSS_LANDSCAPE_BATCHES = 10
        HAS_LR_SCHEDULE = True

        def __init__(self, root, augmentation=True, auto_augment=False, exclude_translations=False, cutout=False):
            super(CIFAR100, self).__init__()

            self.ffcv=False
            tfs = [transforms.RandomHorizontalFlip()]

            if augmentation and not exclude_translations:
                tfs+= [transforms.RandomCrop(32, padding=4)]
            
            if auto_augment:
                tfs += [CIFAR10Policy(exclude_translations = exclude_translations)]
            
            tfs += [transforms.ToTensor(),
                    transforms.Normalize(MEAN['CIFAR100'], STD['CIFAR100'])]

            if auto_augment or cutout:
                tfs += [transforms.RandomErasing(p=0.5, scale=(0.5, 0.5), ratio=(1, 1))]

            train_transforms = transforms.Compose(tfs)
            test_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(MEAN['CIFAR100'], STD['CIFAR100'])])

            train_data = CIFAR100_(root, train=True, transform=train_transforms, download=True)
            self.splits['train'] = train_data
            _, val_idx = sample_idxs(train_data, val_frac=0.1)
            self.splits['val'] =  Subset(train_data, val_idx)

            self.splits['test'] = CIFAR100_(root, train=False, transform=test_transforms)

        @staticmethod
        def adjust_lr(optimizer, epoch, hparams):
            """
            Decay initial learning rate exponentially starting after epoch_start epochs
            The learning rate is multiplied with base_factor every lr_decay_epoch epochs
            """
            lr = hparams['learning_rate']
            if epoch > hparams['lr_decay_start']:
                lr = hparams['learning_rate'] * (hparams['lr_decay_factor'] ** ((epoch - hparams['lr_decay_start']) // hparams['lr_decay_epoch']))
            print('learning rate = {:6f}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return

    class STL10(AdvRobDataset):
        INPUT_SHAPE = (3, 96, 96)
        NUM_CLASSES = 10
        N_EPOCHS = 1000
        CHECKPOINT_FREQ = 100
        TEST_INTERVAL = 50
        LOG_INTERVAL = 100
        LOSS_LANDSCAPE_INTERVAL = 1000
        LOSS_LANDSCAPE_GSIZE = 500
        ANGLE_GSIZE = 100
        LOSS_LANDSCAPE_BATCHES = 5
        HAS_LR_SCHEDULE = True
        ATTACK_INTERVAL = 1000

        def __init__(self, root, augmentation=True, auto_augment=False, exclude_translations=False, cutout=False):
            super(STL10, self).__init__()

            self.ffcv=False
            if augmentation:
                tfs = [transforms.RandomHorizontalFlip()]
            else:
                tfs = []

            tfs += [transforms.ToTensor()]
            if augmentation:        
                tfs += [Cutout(60)]
            tfs += [transforms.Normalize(MEAN['CIFAR10'], STD['CIFAR10'])]

            train_transforms = transforms.Compose(tfs)
            test_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(MEAN['CIFAR10'], STD['CIFAR10'])])

            train_data = STL10_(root, split="train", transform=train_transforms, download=True)
            self.splits['train'] = train_data
            
            _, val_idx = sample_idxs(train_data, val_frac=0.1)
            self.splits['val'] =  Subset(train_data, val_idx)

            self.splits['test'] = STL10_(root, split="test", transform=test_transforms)

        @staticmethod
        def adjust_lr(optimizer, epoch, hparams):
            """
            Decay initial learning rate exponentially starting after epoch_start epochs
            The learning rate is multiplied with base_factor every lr_decay_epoch epochs
            """
            lr = hparams['learning_rate']
            if epoch >= 300:
                lr = hparams['learning_rate'] * 0.2
            if epoch >= 400:
                lr = hparams['learning_rate'] * 0.04
            if epoch >= 600:
                lr = hparams['learning_rate'] * 0.008
            if epoch >= 800:
                lr = hparams['learning_rate'] * 0.0016
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

class MNIST(AdvRobDataset):
    INPUT_SHAPE = (1, 28, 28)
    NUM_CLASSES = 10
    N_EPOCHS = 100
    CHECKPOINT_FREQ = 99
    LOG_INTERVAL = 100
    ATTACK_INTERVAL = 100
    LOSS_LANDSCAPE_INTERVAL = 100
    LOSS_LANDSCAPE_BATCHES = 20
    HAS_LR_SCHEDULE = True
    LOSS_LANDSCAPE_GSIZE = 1000
    ANGLE_GSIZE = 100
    LOSS_LANDSCAPE_BATCHES = 20

    def __init__(self, root, augmentation=False):
        super(MNIST, self).__init__()
        self.ffcv = False
        
        xforms = transforms.ToTensor()

        train_data = MNIST_(root, train=True, transform=xforms,  download=True)
        self.splits['train'] = train_data
        # self.splits['train'] = Subset(train_data, range(60000))

        train_data = MNIST_(root, train=True, transform=xforms)
        _, val_idx = sample_idxs(train_data, val_frac=0.1)
        self.splits['val'] =  Subset(train_data, val_idx)

        self.splits['test'] = MNIST_(root, train=False, transform=xforms)

    @staticmethod
    def adjust_lr(optimizer, epoch, hparams):
        lr = hparams['learning_rate']
        if epoch >= 55:
            lr = hparams['learning_rate'] * 0.1
        if epoch >= 75:
            lr = hparams['learning_rate'] * 0.01
        if epoch >= 90:
            lr = hparams['learning_rate'] * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class MNISTe2(AdvRobDataset):
    INPUT_SHAPE = (1, 28, 28)
    NUM_CLASSES = 10
    N_EPOCHS = 40
    CHECKPOINT_FREQ = 50
    LOG_INTERVAL = 100
    ATTACK_INTERVAL = 40
    LOSS_LANDSCAPE_INTERVAL = 40
    LOSS_LANDSCAPE_BATCHES = 50
    HAS_LR_SCHEDULE = False
    LOSS_LANDSCAPE_GSIZE = 1000#28000
    ANGLE_GSIZE = 100
    LOSS_LANDSCAPE_BATCHES = 20
    HAS_LR_SCHEDULE_DUAL = False

    # test adversary parameters
    ADV_STEP_SIZE = 0.1
    N_ADV_STEPS = 10

    def __init__(self, root, augmentation=False):
        super(MNIST, self).__init__()
        self.ffcv = False
        
        xforms = transforms.ToTensor()

        train_data = MNIST_(root, train=True, transform=xforms,  download=True)
        self.splits['train'] = train_data
        # self.splits['train'] = Subset(train_data, range(60000))

        train_data = MNIST_(root, train=True, transform=xforms)
        self.splits['val'] = Subset(train_data, range(54000, 60000))

        self.splits['test'] = MNIST_(root, train=False, transform=xforms)

    @staticmethod
    def adjust_lr(optimizer, epoch, hparams):
        """
        Decay initial learning rate exponentially starting after epoch_start epochs
        The learning rate is multiplied with base_factor every lr_decay_epoch epochs
        """
        lr = hparams['learning_rate']
        if epoch > hparams['lr_decay_start']:
            lr = hparams['learning_rate'] * (hparams['lr_decay_factor'] ** ((epoch - hparams['lr_decay_start']) // hparams['lr_decay_epoch']))
        print('learning rate = {:6f}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return

class IMNET(AdvRobDataset):
        INPUT_SHAPE = (3, 224, 224)
        NUM_CLASSES = 1000
        N_EPOCHS = 20
        CHECKPOINT_FREQ = 10
        LOG_INTERVAL = 50
        LOSS_LANDSCAPE_INTERVAL = 100
        LOSS_LANDSCAPE_GSIZE = 1000
        ANGLE_GSIZE = 100
        LOSS_LANDSCAPE_BATCHES = 10
        HAS_LR_SCHEDULE = True

        # test adversary parameters
        ADV_STEP_SIZE = 2/255.
        N_ADV_STEPS = 10

        def __init__(self, root, augmentation=True):
            super(IMNET, self).__init__()
            self.data_path = "~/chiche/imagenet1k/ILSVRC/Data/CLS-LOC/"
            self.ffcv=False
            train_transforms = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation="bicubic",
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=MEAN['IMNET'],
            std=STD['IMNET'])
            if augmentation:
                train_transforms.transforms[0] = transforms.RandomCrop(224, padding=4)
            
            test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN['IMNET'], STD['IMNET'])])
            train_root = os.path.join(self.data_path, 'train' )
            val_root = os.path.join(self.data_path, 'val' )
            test_root = os.path.join(self.data_path, 'val' )
            train_data = ImageFolder(train_root, transform=test_transforms)
            for i in range(len(train_data)):
                img, label = train_data[i]
                print(img.shape)
            self.splits['train'] = train_data
            self.splits['val'] = ImageFolder(val_root, transform=test_transforms)
            self.splits['test'] = ImageFolder(test_root, transform=test_transforms)

        @staticmethod
        def adjust_lr(optimizer, epoch, hparams):
            """
            Decay initial learning rate exponentially starting after epoch_start epochs
            The learning rate is multiplied with base_factor every lr_decay_epoch epochs
            """
            lr = hparams['learning_rate']
            if epoch > hparams['lr_decay_start']:
                lr = hparams['learning_rate'] * (hparams['lr_decay_factor'] ** ((epoch - hparams['lr_decay_start']) // hparams['lr_decay_epoch']))
            print('learning rate = {:6f}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return

class modelnet40(AdvRobDataset):
    INPUT_SHAPE = (3, 1024)
    NUM_CLASSES = 40
    N_EPOCHS = 250
    NUM_POINTS = 1024
    CHECKPOINT_FREQ = 100
    LOG_INTERVAL = 100
    ATTACK_INTERVAL = 250
    LOSS_LANDSCAPE_INTERVAL = 300
    LOSS_LANDSCAPE_GSIZE = 100
    ANGLE_GSIZE = 100
    LOSS_LANDSCAPE_BATCHES = 10
    HAS_LR_SCHEDULE = True
    MIN_LR = 0.005
    START_EPOCH = 0
    TEST_BATCH = 10

    # test adversary parameters
    ADV_STEP_SIZE = 2/255.
    N_ADV_STEPS = 10

    def __init__(self, root, augmentation=True):
        super(modelnet40, self).__init__()
        self.ffcv=False
        self.splits['train'] = _ModelNet40(partition='train', num_points=self.NUM_POINTS,random_translate=False, validation=True)
        self.splits['val'] = get_val(self.splits['train'])
        self.splits['test'] =  _ModelNet40(partition='test', num_points=self.NUM_POINTS)

class _ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', random_translate=True, validation=False):
        self.validation = validation
        if self.validation:
            self.all_data, self.all_label, self.train_idx, self.val_idx = self.load_data(partition, validation=validation)
            if partition=="train":
                self.data, self.label =  self.all_data[self.train_idx], self.all_label[self.train_idx]
            elif partition=="val":
                self.data, self.label =  self.all_data[self.val_idx], self.all_label[self.val_idx]
        else:
            self.data, self.label = self.load_data(partition, validation=self.validation)
        self.num_points = num_points
        self.partition = partition
        self.random_translate = random_translate

    def set_validation(self):
        if self.validation:
            self.data, self.label = self.all_data[self.val_idx], self.all_label[self.val_idx]
        else:
            print("Dataset has no validation set")

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            if self.random_translate:
                pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud.T, label

    def __len__(self):
        return self.data.shape[0]

    def download(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
            www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
            zipfile = os.path.basename(www)
            os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
            os.system('rm %s' % (zipfile))


    def load_data(self, partition, validation=False):
        self.download()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
            # print(f"h5_name: {h5_name}")
            f = h5py.File(h5_name,'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        if validation:
            idx = np.arange(len(all_data))
            train_idxs, val_idxs = train_test_split(idx, test_size=0.1)
            return all_data, all_label, train_idxs, val_idxs
        else:
            return all_data, all_label
    

def get_val(dataset):
    val_set = deepcopy(dataset)
    val_set.set_validation()
    return(val_set)