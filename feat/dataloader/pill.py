import numpy as np
import os
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

THIS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(os.path.join(THIS_PATH, "..", ".."))
SPLIT_PATH = os.path.join(ROOT_PATH, "data/pill/split")


class PillDataset(Dataset):
    def __init__(self, subset, args):
        """Dataset class representing a pill dataset

        # Arguments:
            subset: Whether the dataset represents the background, evaluation or test set
        """
        if subset not in ("train", "val", "test"):
            raise (ValueError, "Subset must be one of (train, val, test)")

        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(n-1) integers
        self.unique_characters = sorted(self.df["class_name"].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df["class_name"].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()["filepath"]
        self.datasetid_to_class_id = self.df.to_dict()["class_id"]

        # Transformation
        if args.model_type == "ConvNet":
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.model_type == "ResNet":
            image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif args.model_type == "AmdimNet":
            INTERP = 3
            post_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.transform = transforms.Compose([
                transforms.Resize(146, interpolation=INTERP),
                transforms.CenterCrop(128),
                post_transform
            ])
        else:
            raise ValueError("Non-supported network types. Please revise the data pre-processing scripts.")

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df["class_name"].unique())

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the pill dataset
        """
        images = []
        print(f"Indexing {subset}...")

        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, filenames in os.walk(SPLIT_PATH + "/images_{}/".format(subset)):
            subset_len += len([filename for filename in filenames if filename.endswith(".jpg")])

        progress_bar = tqdm(total=subset_len, position=0, leave=True)
        for root, folders, filenames in os.walk(SPLIT_PATH + "/images_{}/".format(subset)):
            if len(filenames) == 0:
                continue

            class_name = root.split("/")[-1]

            for file in filenames:
                progress_bar.update(1)
                images.append({
                    "subset": subset,
                    "class_name": class_name,
                    "filepath": os.path.join(root, file)
                })

        progress_bar.close()
        return images
