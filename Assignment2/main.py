from collections import Counter

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Subset, DataLoader

from sklearn.model_selection import train_test_split


def load_dataset(transform):

    trainval_dataset = datasets.OxfordIIITPet(
        root='./data',
        download=True,
        transform=transform,
        split ="trainval"
    )

    test_dataset = datasets.OxfordIIITPet(
        root='./data',
        download=True,
        transform=transform,
        split ="test"
    )

    return trainval_dataset, test_dataset

def split_dataset(dataset, seed):

    labels = dataset._labels

    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        stratify=labels,      
        random_state=seed,      
    )

    train_dataset = Subset(dataset=dataset, indices=train_idx)
    val_dataset = Subset(dataset=dataset, indices=val_idx)

    return train_dataset, val_dataset


def get_split_labels(split: Subset):
    return [split.dataset._labels[i] for i in split.indices]

def get_breeds_per_split(splits: dict, class_names):
    counts = {name: Counter(labels) for name, labels in splits.items()}

    for label, breed in enumerate(class_names):
        print(breed, {name: counts[name][label] for name in splits})


SEED = 1
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("on device: ", device)


    #Load and split dataset
    transform = transforms.Compose([  # TODOO describe resizing and normalization
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    trainval_dataset, test_dataset = load_dataset(transform=transform)
    train_dataset, val_dataset = split_dataset(dataset=trainval_dataset, seed=SEED)


    # Report number of images per breed in each split
    get_breeds_per_split(
        splits={"train": get_split_labels(train_dataset),
                "val": get_split_labels(val_dataset),
                "test": test_dataset._labels},
        class_names=trainval_dataset.classes,
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)







    # MobileNet_V3_Small_Weights used by default
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(progress = True, weights = weights)





if __name__ == "__main__":
    main()
