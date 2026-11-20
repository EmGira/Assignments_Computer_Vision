import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets


def load_results(path):

    results = {}
    for file in sorted(os.listdir(path)):

        if file.endswith(".json"):

            with open(os.path.join(path, file)) as f:
                data = json.load(f)
                
            data["confusion_matrix"] = np.array(data["confusion_matrix"])

            
            csv_path = os.path.join(path, f"{data['method']}_classification_report.csv")
            
            if os.path.exists(csv_path):
                data["report"] = pd.read_csv(csv_path)
            results[data["method"]] = data

        else:
            continue

    return results

def results_table(results):
    rows = []
    for name, r in results.items():
        rows.append({
            "Method": name.replace("_", " "),   
            "Trainable params": r["trainable_params"],
            "Training time (s)": f"{r['training_info']['train_time_sec']:.1f}",
            "Train macro F1": r["training_info"]["train_f1"],
            "Best val macro F1": r["training_info"]["best_val_f1"],
            "Test accuracy": r["test_accuracy"],
            "Test macro F1": r["test_macro_f1"],
        })
    return pd.DataFrame(rows)


def plot_metrics(results):
    
    names = list(results)
    acc = [results[n]["test_accuracy"] for n in names]
    f1 = [results[n]["test_macro_f1"] for n in names]
    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots()
    bars_acc = ax.bar(x - w / 2, acc, w, label="Test accuracy")
    bars_f1 = ax.bar(x + w / 2, f1, w, label="Macro F1")
    ax.bar_label(bars_acc, fmt="%.3f")
    ax.bar_label(bars_f1, fmt="%.3f")

    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test accuracy and macro F1 per method")
    ax.legend(loc="lower right")
    plt.tight_layout()

    fig.savefig(f"results/figures/metrics_comparison.png")


def plot_confusion_matrix(cm, class_names, name):
    fig, ax = plt.subplots(figsize=(14, 14))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=7)

    ax.set_xticks(range(len(class_names)), class_names, rotation=90)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix - {name}")
    plt.tight_layout()

    fig.savefig(f"results/figures/confusion_matrix_{name}.png")


def plot_predictions(r, test_dataset, name, n=6):
    """
    Top row: n random correct predictions.
    Bottom row: one wrong prediction for each of the n most confused (true, predicted) pairs.
    """
    
    preds = np.array(r["test_predictions"])
    labels = np.array(test_dataset._labels)
    names = r["class_names"]

    rng = np.random.default_rng(0)
    correct = list(rng.choice(np.where(preds == labels)[0], n, replace=False))

    cm = r["confusion_matrix"].copy()
    np.fill_diagonal(cm, 0)    # keep only mistakes
    wrong = []

    for k in np.argsort(-cm.ravel())[:n]:
        true, pred = np.unravel_index(k, cm.shape)
        wrong.append(np.where((labels == true) & (preds == pred))[0][0])


    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7.5))
    for ax, i in zip(axes.ravel(), correct + wrong):
        image, _ = test_dataset[i]
        image = image.resize((256, 256))   

        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"True: {names[labels[i]]}\n  Pred: {names[preds[i]]}",
                     color="green" if preds[i] == labels[i] else "red", fontsize=10)
    fig.suptitle(f"Predictions - {name}")
    plt.tight_layout()

    fig.savefig(f"results/figures/predictions_{name}.png")


def error_analysis(results, test_dataset, n=6):
    """
    - top_confusions.csv: the most frequently confused classes for each method
    - error_summary.csv: total test errors and errors between a cat and a dog breed
    """

  
    species = {}
    for label, bin_label in zip(test_dataset._labels, test_dataset._bin_labels):
        species[label] = bin_label

    pairs, summary = [], []
    for name, r in results.items():
        errors = r["confusion_matrix"].copy()
        np.fill_diagonal(errors, 0)  

        for k in np.argsort(-errors.ravel())[:n]:
            pair = tuple(int(i) for i in np.unravel_index(k, errors.shape))
            if pair not in pairs:
                pairs.append(pair)

        cross = sum(errors[t, p] for t in range(len(errors)) for p in range(len(errors))
                    if species[t] != species[p])
        summary.append({"method": name, "errors": int(errors.sum()), "cat_dog_errors": int(cross),
                        "cat_dog_share": round(cross / errors.sum(), 3)})

    names = next(iter(results.values()))["class_names"]
    rows = [{"true": names[t], "predicted": names[p],
             **{name: int(r["confusion_matrix"][t, p]) for name, r in results.items()}}
            for t, p in pairs]


    pd.DataFrame(rows).to_csv("results/top_confusions.csv", index=False)
    pd.DataFrame(summary).to_csv("results/error_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(summary).to_string(index=False))


def main():
    results = load_results("./results")
    os.makedirs("results/figures", exist_ok=True)

    table = results_table(results)
    print(table.to_string(index=False))
    table.to_latex("results/results_table.tex", index=False, float_format="%.3f")

    plot_metrics(results)

    r = results["full_finetune"]
    plot_confusion_matrix(r["confusion_matrix"], r["class_names"], "full_finetune")
    r = results["classifier_head"]
    plot_confusion_matrix(r["confusion_matrix"], r["class_names"], "classifier_head")
    r = results["feature_extraction_svm"]
    plot_confusion_matrix(r["confusion_matrix"], r["class_names"], "feature_extraction_svm")
    
    test_dataset = datasets.OxfordIIITPet(root="./data", split="test", download=True,
                                          target_types=["category", "binary-category"])
    for name, r in results.items():
        plot_predictions(r, test_dataset, name)

    error_analysis(results, test_dataset)

    plt.show()


if __name__ == "__main__":
    main()
