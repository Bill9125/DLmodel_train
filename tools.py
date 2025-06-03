from collections import Counter
import random
import numpy as np
import torch
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多張 GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 計算 F1-score 的函數
def f1_score(y_true, y_pred):
    # Get unique classes
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    # Initialize
    class_f1_scores = {}
    class_weights = {}
    
    # Count instances of each class in true labels
    total_samples = len(y_true)
    class_counts = Counter(y_true)
    
    # Calculate weights for each class
    for cls in classes:
        class_weights[cls] = class_counts.get(cls, 0) / total_samples
    
    # For each class, calculate F1 score
    for cls in classes:
        # True positives, false positives, false negatives
        tp = np.sum((y_true == cls) & (y_pred == cls) & ~((y_true == 0) & (y_pred == 0)))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 score for this class
        if precision + recall > 0:
            class_f1_scores[cls] = 2 * (precision * recall) / (precision + recall)
        else:
            class_f1_scores[cls] = 0
    
    # Calculate weighted F1 score
    weighted_f1 = sum(class_weights[cls] * class_f1_scores[cls] for cls in classes)
    return weighted_f1

def compute_f1_score(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels, indices in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return f1_score(y_true, y_pred)  

def write_results(seeds, all_f1_scores, best_f1, best_seed, best_model_path, save_dir):
    # 🔍 顯示結果 & 建立結果字串
    summary_lines = []
    summary_lines.append("\n✅ F1 scores from each seed:")
    for se, f1 in zip(seeds, all_f1_scores):
        summary_lines.append(f"Seed {se}: F1 = {f1:.4f}")

    summary_lines.append(f"\n📊 Average F1 Score: {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}")
    summary_lines.append(f"🏆 Best F1: {best_f1:.4f} from Seed {best_seed}")
    summary_lines.append(f"📁 Best model saved at: {best_model_path}")

    # 印出結果到 terminal
    for line in summary_lines:
        print(line)

    # 📄 寫入 txt 檔案
    txt_output_path = os.path.join(save_dir, "results_summary.txt")
    with open(txt_output_path, "w", encoding="utf-8") as f:
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\n✅ 寫入完成：{txt_output_path}")