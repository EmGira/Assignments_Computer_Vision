from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Subset, DataLoader
from torchvision.models import MobileNetV3

from sklearn.model_selection import train_test_split


from train_and_evaluate import train_and_evaluate

import csv
import json
import os
import random
import time

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def count_trainable_params(model: MobileNetV3):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_results(name, test_results, training_info, n_trainable, class_names):

    os.makedirs("results", exist_ok=True)
    summary = {
        "method": name,
        "trainable_params": n_trainable,
        "training_info": training_info,
        "test_accuracy": float(test_results["accuracy"]),
        "test_macro_f1": float(test_results["macro_f1"]),
        "class_names": class_names,
        "confusion_matrix": test_results["confusion_matrix"].tolist(),
    }
    with open(f"results/{name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    save_classification_report_csv(
        test_results["classification_report"], class_names,
        path=f"results/{name}_classification_report.csv")


def save_classification_report_csv(report, class_names, path):

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for key, scores in report.items():
            if key == "accuracy":
                continue

            label = class_names[int(key)] if key.isdigit() else key
            writer.writerow([label,
                             round(scores["precision"], 4),
                             round(scores["recall"], 4),
                             round(scores["f1-score"], 4),
                             int(scores["support"])])


def load_dataset(eval_transform, train_transform):

    trainval_augmented_dataset = datasets.OxfordIIITPet(
        root='./data',
        download=True,
        transform=train_transform,
        split ="trainval"
    )

    trainval_evaluation_dataset = datasets.OxfordIIITPet(
            root='./data',
            download=True,
            transform=eval_transform,
            split ="trainval"
        )
    

    test_dataset = datasets.OxfordIIITPet(
        root='./data',
        download=True,
        transform=eval_transform,
        split ="test"
    )

    return trainval_augmented_dataset, trainval_evaluation_dataset, test_dataset

def split_dataset(augmented_dataset, evaluation_dataset, seed):

    labels = evaluation_dataset._labels

    train_idx, val_idx = train_test_split(
        list(range(len(evaluation_dataset))),
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )

    train_dataset = Subset(dataset=augmented_dataset, indices=train_idx)
    val_dataset = Subset(dataset=evaluation_dataset, indices=val_idx)

    return train_dataset, val_dataset


def get_split_labels(split: Subset):
    return [split.dataset._labels[i] for i in split.indices]

def get_breeds_per_split(splits: dict, class_names):
    counts = {name: Counter(labels) for name, labels in splits.items()}

    for label, breed in enumerate(class_names):
        print(breed, {name: counts[name][label] for name in splits})


def extract_features(feature_extractor, loader, device):

    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            raw_features = feature_extractor(images.to(device))         
            all_features.append(torch.flatten(raw_features, 1).cpu())   
            all_labels.append(labels)
        
    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()



SEED = 1
NUM_CLASSES = 37

def main():
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("on device: ", device)


    #find transform associated with the pre trained weights
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    eval_transform = weights.transforms()
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    #Load and split dataset
    trainval_augmented_dataset, trainval_evaluation_dataset, test_dataset = load_dataset(eval_transform, train_transform)

    train_dataset, val_dataset = split_dataset(
        augmented_dataset=trainval_augmented_dataset,
        evaluation_dataset=trainval_evaluation_dataset,
        seed=SEED
        )


    # Report number of images per breed in each split
    get_breeds_per_split(
        splits={"train": get_split_labels(train_dataset),
                "val": get_split_labels(val_dataset),
                "test": test_dataset._labels},
        class_names=trainval_evaluation_dataset.classes,
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)



    # load model with pretrained weights
    model = models.mobilenet_v3_small(progress = True, weights = weights)


    #replace final layer to output 37 classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)

    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()


    class_names = trainval_evaluation_dataset.classes


    NUM_EPOCHS = 10


    #1. FULL FINE TUNING
    n_trainable = count_trainable_params(model)
    print("trainable parameters: ", n_trainable)

    _, test_results, best_weights, training_info = train_and_evaluate(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS
    )
    torch.save(best_weights, "checkpoints/full_finetune_best.pt")
    save_results("full_finetune", test_results, training_info, n_trainable, class_names)


    #2. CLASSIFIER HEAD TRAINING

    #load fresh model
    model = models.mobilenet_v3_small(progress = True, weights = weights)
    print("trainable layers: ", model.classifier)
    #disable learning parameters except for the classification layers
    for param in model.features.parameters():
        param.requires_grad = False


    #replace final classification layer to output 37 classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    n_trainable = count_trainable_params(model)
    print("trainable parameters: ", n_trainable)

    _, test_results, best_weights, training_info = train_and_evaluate(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        freeze_features=True,  # keep BatchNorm statistics of model.features fixed
    )
    torch.save(best_weights, "checkpoints/classifier_head_best.pt")
    save_results("classifier_head", test_results, training_info, n_trainable, class_names)




    #3. FEATURE EXTRACTION WITH SKLEARN CLASSIFIER

    model = models.mobilenet_v3_small(progress = True, weights = weights)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()

    # train images WITHOUT augmentation
    train_plain = Subset(trainval_evaluation_dataset, train_dataset.indices)
    train_plain_loader = DataLoader(train_plain, batch_size=32, shuffle=False)


    start = time.time()
    X_train, y_train = extract_features(feature_extractor, train_plain_loader, device)
    X_val,   y_val   = extract_features(feature_extractor, val_loader, device)

    scaler = StandardScaler()
    scaler.fit(X_train) # compute mean and std on TRAIN features only
    # standardization (val uses the train mean/std)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)


    best_val_f1, best_C, best_svm = 0.0, None, None

    for C in [0.0001, 0.001, 0.01, 0.1]:

        svm = LinearSVC(C=C, max_iter=5000)  
        svm.fit(X_train, y_train)   

        val_f1 = f1_score(y_val, svm.predict(X_val), average="macro")  
        print(f"C={C}: val macro F1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, best_C, best_svm = val_f1, C, svm

    train_time = time.time() - start
    print(f"Best C: {best_C} (val F1 {best_val_f1:.4f}), training time: {train_time:.1f} s")

    # test: extract features and scale them, then evaluate once
    X_test, y_test = extract_features(feature_extractor, test_loader, device)
    X_test = scaler.transform(X_test)

    y_pred = best_svm.predict(X_test)
    test_results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }

    print(f"TEST RESULTS: \n\tacc {test_results['accuracy']:.4f} \n\ttest F1 {test_results['macro_f1']:.4f}\n\n")

    training_info = {"best_epoch": None,  
                     "best_C": best_C,    
                     "best_val_f1": best_val_f1,
                     "train_time_sec": train_time}
    
    save_results("feature_extraction_svm", test_results, training_info, 0, class_names)
    



if __name__ == "__main__":
    main()
