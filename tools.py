import random
import numpy as np
import torch
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多張 GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def compute_f1_score(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels, indices in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)  # [B, num_classes]
            preds = (probs > 0.5).int()     # [B, num_classes]
            y_true.extend(labels.cpu().numpy().tolist())    # labels shape: (batch, num_classes)
            y_pred.extend(preds.tolist())                   # preds shape: (batch, num_classes)

    return f1_score(y_true, y_pred, average='macro')

def multilabel_confusion_matrix_mix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        # 對每一個 ground truth 的正類別都視為一筆單一分類樣本
        true_classes = np.where(np.array(yt) == 1)[0]
        pred_classes = np.where(np.array(yp) == 1)[0]

        for t in true_classes:
            for p in pred_classes:
                cm[t][p] += 1
    return cm

def plot_custom_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.set_title("Pastch TST Confusion Matrix")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, ha="right")
    ax.set_yticklabels(class_names)

    # 顯示數值與百分比
    cm_sum = cm.sum(axis=1, keepdims=True)  # 每一列總數
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            total = cm_sum[i][0]
            if total == 0:
                percentage = 0
            else:
                percentage = count / total * 100
            ax.text(j, i, f"{count}\n({percentage:.1f}%)",
                    ha="center", va="center",
                    color="white" if count > cm.max() * 0.5 else "black",
                    fontsize=10)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.colorbar(im, ax=ax)
    plt.savefig(save_path)
    plt.close()