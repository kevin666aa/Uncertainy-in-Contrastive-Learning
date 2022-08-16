from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from data_aug.cutout import Cutout

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, aug='origin', s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        trans = [transforms.RandomResizedCrop(size=size),
                    transforms.RandomHorizontalFlip(),
                    
                    transforms.RandomApply([color_jitter], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(kernel_size=int(0.1 * size)),
                    # transforms.RandomApply([Cutout(1, 20)], p=0.9),
                    transforms.ToTensor()]
        print('Training with data aug setting ' + aug)
        if aug == 'nojitter':
            trans.pop(2)
        elif aug == 'nogray':
            trans.pop(3)
        elif aug == 'noblur':
            trans.pop(4)
        elif aug == 'withcrop':
            trans[0] = transforms.RandomResizedCrop(size=int(size*0.75))

        data_transforms = transforms.Compose(trans)
        return data_transforms

    def get_dataset(self, name, n_views, aug='origin'):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32, aug),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96, aug),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
