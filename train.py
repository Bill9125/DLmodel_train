import torch
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import argparse
from models import ResNet32, BiLSTMModel
from tools import set_seed, f1_score, compute_f1_score, write_results
from test import test_model_with_path_tracking

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
            from dataset import Dataset_dd2voz
            datasets_path = os.path.join(os.getcwd(), 'data', '2D_traindata_bodylength_vision1')
            full_dataset = Dataset_dd2voz(datasets_path, GT_class)
            save_dir = os.path.join(os.getcwd(), 'models', 'dd2voz_vision1_body', f'{GT_class}')
            
        elif F_type == '3D':
            from dataset import Dataset_3D
            datasets_path = os.path.join(os.getcwd(), 'data', '3D_traindata')
            full_dataset = Dataset_3D(datasets_path, GT_class)
            save_dir = os.path.join(os.getcwd(), 'models', '3D', f'{GT_class}')
        input_dim = full_dataset.dim
        print('input_dim',input_dim)
    
    else:
        from dataset import Dataset_SHAP
        datasets_path = os.path.join(os.getcwd(), 'dataset')
        full_dataset = Dataset_SHAP(datasets_path, GT_class, SHAP_mode)
        input_dim = full_dataset.dim
        save_dir = os.path.join(os.getcwd(), 'models_SHAP', f'{GT_class}', f'SHAP_{SHAP_mode}')

    category_ratio = full_dataset.get_ratio()
    print(f'Category : {category_ratio}')
    train_size = int(0.75 * len(full_dataset))
    valid_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    
    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    all_avg_times = []
    all_acc = []
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

        avg_loss, f1, acc, avg_time_per_sample, false_positives, false_negatives = test_model_with_path_tracking(
            model, test_loader, test_dataset, criterion, save_dir, save_path, full_dataset
        )

        print(f"Seed {se} Test F1: {f1:.4f}")
        all_f1_scores.append(f1)
        all_avg_times.append(avg_time_per_sample)
        all_acc.append(acc)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = se
            best_model_path = save_path

    write_results(seeds, all_f1_scores, best_f1, best_seed, best_model_path, save_dir)