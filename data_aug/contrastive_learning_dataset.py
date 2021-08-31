from data_aug.augmentations import GaussianNoise, RandomCrop, ToTensor
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from data_aug.myDataset import mydataset

class ContrastiveLearningDataset:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def get_simclr_pipeline_transform(mean = 0 , std = 1 , n = 100):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        data_transforms = transforms.Compose([GaussianNoise(),
                                              RandomCrop(),
                                              ToTensor()
                                              ])
        return data_transforms

    def get_dataset(self , n_views=2):
        dataset = mydataset(self.df , transform = ContrastiveLearningViewGenerator(
            self.get_simclr_pipeline_transform(),
            n_views))
        return dataset

