import torch
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import time
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import random
from models import ResNet32
from models import PatchTSTClassifier
from torchsummary import summary

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

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path, num_epochs=100, patience=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_f1 = 0.0  # 用來儲存最佳 F1-score
    patience_counter = 0

    # **存放訓練過程的數據**
    train_losses = []
    train_f1_scores = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        y_true, y_pred = [], []

        for inputs, labels, indices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(y_true, y_pred)
        val_f1 = compute_f1_score(model, valid_loader)

        scheduler.step()
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # **紀錄數據**
        train_losses.append(avg_loss)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

        # 根據 F1-score 儲存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("✅ Model Saved (Best F1-score)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early Stopping Triggered")
                break

    # **繪製 Loss 和 F1-score**
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label="Train Loss", color='blue', marker='o')
    plt.plot(epochs, train_f1_scores, label="Train F1-score", color='green', marker='s')
    plt.plot(epochs, val_f1_scores, label="Validation F1-score", color='red', marker='d')

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training Loss & F1-score per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")  # 儲存高解析度圖片

# ----------------------
# Validation Function
# ----------------------
def validate_model(model, valid_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels, indices in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(valid_loader)

# ----------------------
# Testing Function
# ----------------------
def test_model_with_path_tracking(model, test_loader, test_dataset, criterion, save_dir, save_path, full_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()
    
    total_loss, total_time = 0.0, 0.0  
    y_true, y_pred = [], []

    false_positives = []
    false_negatives = []
    
    with torch.no_grad():
        for inputs, labels, indices in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            for i in range(len(inputs)):  # 只迭代當前批次中的實際樣本數量
                sample_idx = indices[i].item()  # 直接拿到 full_dataset index！
                detailed_path = full_dataset.get_sample_path(sample_idx)
                
                if predicted[i] == 1 and labels[i] == 0:
                    false_positives.append(f"{str(detailed_path)}")
                elif predicted[i] == 0 and labels[i] == 1:
                    false_negatives.append(f"{str(detailed_path)}")
                    
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    avg_time_per_sample = total_time / len(y_true)
    f1 = f1_score(y_true, y_pred) 

    model_name = os.path.splitext(os.path.basename(save_path))[0]
    txt_dir = os.path.join(save_dir, f"{model_name}_results")
    os.makedirs(txt_dir, exist_ok=True)

    with open(f"{txt_dir}/false_positives.txt", "w") as fp_file:
        fp_file.write("\n".join(false_positives))
    
    with open(f"{txt_dir}/false_negatives.txt", "w") as fn_file:
        fn_file.write("\n".join(false_negatives))
        
    print(f"共有 {len(false_positives)} FP，{len(false_negatives)} FN")
    print(f"已保存到 {txt_dir}/false_positives.txt 和 false_negatives.txt")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(f"{txt_dir}/confusion_matrix.png")
    plt.close()

    return avg_loss, f1, avg_time_per_sample, false_positives, false_negatives

                                     
# ----------------------
# (6) Main Execution
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT_class',type=str)
    parser.add_argument('--SHAP',type=str, default=None)
    parser.add_argument('--F_type',type=str)
    args = parser.parse_args()
    GT_class = args.GT_class
    SHAP_mode = args.SHAP
    F_type = args.F_type
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if SHAP_mode is None:
        if F_type == '2D':
            from dataset import Dataset_25f
            datasets_path = os.path.join(os.getcwd(), '2D_traindata_with_hand')
            full_dataset = Dataset_25f(datasets_path, GT_class)
            save_dir = f'./models_dd2voz_hand/{GT_class}'
            
        elif F_type == '3D':
            from dataset import Dataset_3D
            datasets_path = os.path.join(os.getcwd(), '3D_traindata')
            full_dataset = Dataset_3D(datasets_path, GT_class)
            save_dir = f'./models_3D/{GT_class}'
        input_dim = full_dataset.dim
    
    else:
        from dataset import Dataset_SHAP
        datasets_path = os.path.join(os.getcwd(), 'dataset')
        full_dataset = Dataset_SHAP(datasets_path, GT_class, SHAP_mode)
        input_dim = full_dataset.dim
        save_dir = f'./models_SHAP/{GT_class}/SHAP_{SHAP_mode}'
    
    category_ratio = full_dataset.get_ratio()
    print(f'Category : {category_ratio}')
    train_size = int(0.75 * len(full_dataset))
    valid_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    
    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    seeds = [42, 2023, 7, 88, 100, 999]

    for se in seeds:
        set_seed(se)

        # 分割資料
        train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])
        train_labels = [full_dataset.labels[i] for i in train_dataset.indices]

        # 建立 Weighted Sampler
        class_weights = [1.0 / sum(np.array(train_labels) == i) for i in range(2)]
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
        valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # 訓練與測試
        model = ResNet32(input_dim).to(device)
        P_ratio = category_ratio[GT_class]
        class_counts = torch.tensor([P_ratio, 1 - P_ratio])
        criterion = CrossEntropyLoss(weight=(1.0 / class_counts).to(device))
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        save_path = os.path.join(save_dir, f"ResNet32_model_seed{se}.pth")
        fig_path = os.path.join(save_dir, f"ResNet32_train_results_seed{se}.png")

        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path)

        avg_loss, f1, avg_time_per_sample, false_positives, false_negatives = test_model_with_path_tracking(
            model, test_loader, test_dataset, criterion, save_dir, save_path, full_dataset
        )

        print(f"Seed {se} Test F1: {f1:.4f}")
        all_f1_scores.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = se
            best_model_path = save_path

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