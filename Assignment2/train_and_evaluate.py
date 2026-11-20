import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import copy
import time
from torchvision.models import MobileNetV3

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)          
           
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "predictions": all_preds,
        "classification_report": classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    }

def train_and_evaluate(model:MobileNetV3, optimizer, criterion, device, train_loader, val_loader, test_loader, train_eval_loader, num_epochs, freeze_features=False):
  
    best_f1 = 0.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())


    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        if freeze_features:
            model.features.eval()
        running_loss = 0.0
        
        
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
        
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()


            running_loss += loss.item()

            if (step + 1) % 10 == 0:
                    print(f"step [{step + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}")
                    running_loss = 0.0
            


        #evaluation
        val_results = evaluate(model, val_loader, device)
        print(f"\nEPOCH {epoch}: \n accuracy: {val_results["accuracy"]} \n f1_score: {val_results["macro_f1"]}\n\n")
        if val_results["macro_f1"] > best_f1:
            best_f1 = val_results["macro_f1"]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())


    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f} s")



    # test
    print(f"Test on best epoch: {best_epoch} (val F1 {best_f1:.4f}) \n")
    model.load_state_dict(best_weights)

    # training F1 of the best model, on the training images WITHOUT augmentation
    train_f1 = evaluate(model, train_eval_loader, device)["macro_f1"]


    test_results = evaluate(model, test_loader, device)

    print(f"TEST RESULTS: \n\tacc {test_results['accuracy']:.4f} \n\ttest F1 {test_results['macro_f1']:.4f}\n\n")

    
    training_info = {
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "train_f1": train_f1,
        "train_time_sec": train_time,
    }
    return model, test_results, best_weights, training_info
