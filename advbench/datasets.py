import os
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10 as CIFAR10_
from torchvision.datasets import CIFAR100 as CIFAR100_
from torchvision.datasets import MNIST as MNIST_
try:
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
    from ffcv.transforms.common import Squeeze
    from ffcv.writer import DatasetWriter
    FFCV_AVAILABLE=True
except ImportError:
    FFCV_AVAILABLE=False

SPLITS = ['train', 'val', 'test']
DATASETS = ['CIFAR10', 'MNIST']
MEAN = {
    'CIFAR10': (0.4914, 0.4822, 0.4465),
    'CIFAR100': (0.5071, 0.4867, 0.4408),
}

STD = {
    'CIFAR10': (0.2023, 0.1994, 0.2010),
    'CIFAR100': (0.2675, 0.2565, 0.2761),
}

def to_loaders(all_datasets, hparams):
    if not all_datasets.ffcv:    
        def _to_loader(split, dataset):
            batch_size = hparams['batch_size'] if split == 'train' else 10
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
            
            batch_size = hparams['batch_size'] if split == 'train' else 48

            label_pipeline = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
            
            loaders.append(Loader(path, batch_size=batch_size, num_workers=all_datasets.N_WORKERS,
                                order=ordering, drop_last=(split == 'train'),
                                pipelines={'image':all_datasets.transforms[split], 'label': label_pipeline}))

    return loaders
        


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
    TRANSLATIONS = [-3, 0, 3]

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
        HAS_LR_SCHEDULE_DUAL = False
        HAS_LR_SCHEDULE = True
        HAS_LR_SCHEDULE_DUAL = True
        MAX_DUAL_LR_FACTOR = 8
        MAX_DUAL_LR_EPOCH = 120
        MIN_DUAL_LR_EPOCH = 180
        MIN_DUAL_LR_FACTOR = 0.01


        # test adversary parameters
        ADV_STEP_SIZE = 2/255.
        N_ADV_STEPS = 20

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
        def adjust_lr_dual(pd_optimizer, epoch):
            lr = pd_optimizer.eta
            if epoch <= self.MAX_DUAL_LR_EPOCH:
                lr = lr*self.MAX_LR_FACTOR*(epoch-self.MAX_DUAL_LR_EPOCH)
            elif epoch <= self.MIN_LR_EPOCH:
                lr = lr*self.MIN_LR_FACTOR*(epoch-self.MAX_DUAL_LR_EPOCH)/(self.MIN_DUAL_LR_EPOCH-self.MAX_DUAL_LR_EPOCH)


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
        LOG_INTERVAL = 50
        LOSS_LANDSCAPE_INTERVAL = 200
        LOSS_LANDSCAPE_GSIZE = 1000
        ANGLE_GSIZE = 100
        LOSS_LANDSCAPE_BATCHES = 10
        HAS_LR_SCHEDULE = True
        HAS_LR_SCHEDULE_DUAL = True
        MAX_DUAL_LR_FACTOR = 8
        MAX_DUAL_LR_EPOCH = 120
        MIN_DUAL_LR_EPOCH = 180
        MIN_DUAL_LR_FACTOR = 0.01

        # test adversary parameters
        ADV_STEP_SIZE = 2/255.
        N_ADV_STEPS = 20

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
            self.splits['val'] = Subset(train_data, range(45000, 50000))

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
        
        @staticmethod
        def adjust_lr_dual(pd_optimizer, epoch):
            lr = pd_optimizer.eta
            if epoch <= self.MAX_DUAL_LR_EPOCH:
                lr = lr*self.MAX_LR_FACTOR*(epoch-self.MAX_DUAL_LR_EPOCH)
            elif epoch <= self.MIN_LR_EPOCH:
                lr = lr*self.MIN_LR_FACTOR*(epoch-self.MAX_DUAL_LR_EPOCH)/(self.MIN_DUAL_LR_EPOCH-self.MAX_DUAL_LR_EPOCH)

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
        HAS_LR_SCHEDULE_DUAL = True

        # test adversary parameters
        ADV_STEP_SIZE = 2/255.
        N_ADV_STEPS = 20

        def __init__(self, root, augmentation=True):
            super(CIFAR10, self).__init__()

            self.ffcv=False
            if augmentation:
                train_transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
            else:
                train_transforms = transforms.Compose([
                    transforms.ToTensor()])
            test_transforms = transforms.ToTensor()

            train_data = CIFAR10_(root, train=True, transform=train_transforms, download=True)
            self.splits['train'] = train_data
            # self.splits['train'] = Subset(train_data, range(5000))

            train_data = CIFAR10_(root, train=True, transform=train_transforms)
            self.splits['val'] = Subset(train_data, range(45000, 50000))

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
        @staticmethod
        def adjust_lr_dual(pd_optimizer, epoch):
            lr = pd_optimizer.eta
            if epoch == 20:
                lr = lr * 1.5
            if epoch == 40:
                lr = lr * 2
            if epoch == 80:
                lr = lr * 2
            if epoch == 150:
                lr = lr * 5
            pd_optimizer.eta = lr
    
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
        HAS_LR_SCHEDULE_DUAL = True

        # test adversary parameters
        ADV_STEP_SIZE = 2/255.
        N_ADV_STEPS = 20

        def __init__(self, root, augmentation=True):
            super(CIFAR100, self).__init__()

            self.ffcv=False
            
            if augmentation:
                train_transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN['CIFAR100'], STD['CIFAR100'])])
            else:
                train_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN['CIFAR100'], STD['CIFAR100'])])
            test_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(MEAN['CIFAR100'], STD['CIFAR100'])])

            train_data = CIFAR100_(root, train=True, transform=train_transforms, download=True)
            self.splits['train'] = train_data
            # self.splits['train'] = Subset(train_data, range(5000))

            train_data = CIFAR100_(root, train=True, transform=train_transforms)
            self.splits['val'] = Subset(train_data, range(45000, 50000))

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
        
        @staticmethod
        def adjust_lr_dual(pd_optimizer, epoch):
            lr = pd_optimizer.eta
            if epoch == 20:
                lr = lr * 1.5
            if epoch == 40:
                lr = lr * 2
            if epoch == 80:
                lr = lr * 2
            if epoch == 130:
                lr = lr / 10
            if epoch == 180:
                lr = lr / 10
            pd_optimizer.eta = lr    

class MNIST(AdvRobDataset):
    INPUT_SHAPE = (1, 28, 28)
    NUM_CLASSES = 10
    N_EPOCHS = 100
    CHECKPOINT_FREQ = 50
    LOG_INTERVAL = 25
    ATTACK_INTERVAL = 25
    LOSS_LANDSCAPE_INTERVAL = 10
    LOSS_LANDSCAPE_BATCHES = 40
    HAS_LR_SCHEDULE = False
    LOSS_LANDSCAPE_GSIZE = 1000#28000
    ANGLE_GSIZE = 100
    LOSS_LANDSCAPE_BATCHES = 10
    HAS_LR_SCHEDULE_DUAL = True

    # test adversary parameters
    ADV_STEP_SIZE = 0.1
    N_ADV_STEPS = 10

    def __init__(self, root):
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

        lr = hparams['learning_rate']
        if epoch >= 55:
            lr = hparams['learning_rate'] * 0.1
        if epoch >= 75:
            lr = hparams['learning_rate'] * 0.01
        if epoch >= 90:
            lr = hparams['learning_rate'] * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    @staticmethod
    def adjust_lr_dual(pd_optimizer, epoch):
        lr = pd_optimizer.eta
        if epoch == 10:
            lr = lr * 2
        if epoch == 20:
            lr = lr * 2
        if epoch == 25:
            lr = lr * 2
        if epoch == 50:
            lr = lr * 5
        if epoch == 90:
            lr = lr * 5
        pd_optimizer.eta = lr
  

class MNIST(AdvRobDataset):
    INPUT_SHAPE = (1, 28, 28)
    NUM_CLASSES = 10
    N_EPOCHS = 100
    CHECKPOINT_FREQ = 50
    LOG_INTERVAL = 100
    ATTACK_INTERVAL = 10
    LOSS_LANDSCAPE_INTERVAL = 10
    LOSS_LANDSCAPE_BATCHES = 40
    HAS_LR_SCHEDULE = False
    LOSS_LANDSCAPE_GSIZE = 1000#28000
    ANGLE_GSIZE = 100
    LOSS_LANDSCAPE_BATCHES = 20
    HAS_LR_SCHEDULE_DUAL = True

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
        lr = hparams['learning_rate']
        if epoch >= 55:
            lr = hparams['learning_rate'] * 0.1
        if epoch >= 75:
            lr = hparams['learning_rate'] * 0.01
        if epoch >= 90:
            lr = hparams['learning_rate'] * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    @staticmethod
    def adjust_lr_dual(pd_optimizer, epoch):
        lr = pd_optimizer.eta
        if epoch == 10:
            lr = lr * 2
        if epoch == 20:
            lr = lr * 2
        if epoch == 25:
            lr = lr * 2
        if epoch == 90:
            lr = lr / 10
        if epoch == 120:
            lr = lr / 10
        pd_optimizer.eta = lr

class dataloader_attack(object):
    def __init__(self, attack):
        self.attack = attack
        