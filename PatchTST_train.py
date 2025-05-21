import torch
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from models import PatchTSTClassifier
from sklearn.metrics import f1_score
from tools import *
import argparse
import json

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

            probs = torch.sigmoid(outputs)  # [B, num_classes]
            preds = (probs > 0.5).int()     # [B, num_classes]
            y_true.extend(labels.cpu().numpy().tolist())    # labels shape: (batch, num_classes)
            y_pred.extend(preds.tolist())                   # preds shape: (batch, num_classes)

        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(y_true, y_pred, average='macro')
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
    
def test_model_with_path_tracking(model, test_loader, criterion, save_dir, save_path, full_dataset, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()
    
    model_name = os.path.splitext(os.path.basename(save_path))[0]
    txt_dir = os.path.join(save_dir, f"{model_name}_results")
    os.makedirs(txt_dir, exist_ok=True)
    
    # **存放測試過程的數據**
    total_loss, total_time = 0.0, 0.0  
    y_true, y_pred = [], []
    cm_details = {str(i): [] for i in range(num_classes * num_classes)}
    
    with torch.no_grad():
        for inputs, labels, indices in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            print(inputs.ndim)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)  # [B, num_classes]
            preds = (probs > 0.5).int()     # [B, num_classes]

            for i in range(len(inputs)):
                sample_idx = indices[i].item()
                detailed_path = full_dataset.get_sample_path(sample_idx)
                
                # labels/preds shape: (B, 4)
                true_vec = labels[i].cpu().numpy()
                pred_vec = preds[i].cpu().numpy()
                
                if true_vec.sum() == 0 and pred_vec.sum() == 0:
                    continue  # 忽略全0樣本（屬於Category_0）
            
                # 找出 true_label 和 pred_label
                true_label = np.argmax(true_vec)
                pred_label = np.argmax(pred_vec)
                pred_confidence = probs[i][pred_label].item()
                cm_index = true_label * num_classes + pred_label
                cm_details[str(cm_index)].append([detailed_path, pred_label, pred_confidence])
                    
            y_true.extend(labels.cpu().numpy().tolist())    # labels shape: (batch, num_classes)
            y_pred.extend(preds.tolist())                   # preds shape: (batch, num_classes)

    avg_loss = total_loss / len(test_loader)
    avg_time_per_sample = total_time / len(y_true)
    f1 = f1_score(y_true, y_pred, average='macro')

    # 繪製混淆矩陣
    classes = [str(i) for i in range(1, num_classes+1)]
    cm = multilabel_confusion_matrix(y_true, y_pred, sample_weight=None, labels=None, samplewise=False)
    n_classes = cm.shape[0]
    fig, axes = plt.subplots(nrows=(n_classes+1)//2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(n_classes):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i], display_labels=['False', 'True'])
        disp.plot(include_values=True, cmap="Blues", ax=axes[i], 
                xticks_rotation="horizontal", values_format="d")
        axes[i].set_title(f'Class: {classes[i]}')

    plt.tight_layout()
    plt.savefig(f"{txt_dir}/confusion_matrix.png")
    plt.close()
    
    cm = multilabel_confusion_matrix_4x4(y_true, y_pred, num_classes)
    plot_custom_confusion_matrix(cm, classes, f"{txt_dir}/confusion_matrix_mix.png")
    with open(f"{txt_dir}/confusion_matrix_detail_paths.json", "w", encoding="utf-8") as f:
        json.dump(cm_details, f, indent=2, ensure_ascii=False)

    return avg_loss, f1, avg_time_per_sample
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str)
    args = parser.parse_args()
    
    from dataset.PatchTST import *
    if args.sport == 'deadlift':
        dataset = os.path.join(os.getcwd(), '3D_traindata')
        full_dataset = Dataset_TST_Deadlift(dataset)
        save_dir = f'./model_TST_Deadlift/8'
        num_classes = 4
    elif args.sport == 'benchpress':
        dataset = os.path.join(os.getcwd(), 'BPdata', 'bench_press_multilabel_dataset_cut3.csv')
        full_dataset = Dataset_TST_Benchpress(dataset)
        save_dir = f'./model_TST_Benchpress/1'
        num_classes = 5
    input_dim = full_dataset.dim
    print('Input dimention',input_dim)
    
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
        gen = torch.Generator().manual_seed(se)  # 為每個seed創建獨立生成器
        train_indices, valid_indices, test_indices = random_split(
            range(len(full_dataset)), [train_size, valid_size, test_size],
            generator=gen
        )
        
        train_dataset = TransformSubset(full_dataset, train_indices, transform=True)
        valid_dataset = TransformSubset(full_dataset, valid_indices, transform=False)
        test_dataset  = TransformSubset(full_dataset, test_indices, transform=False)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 訓練與測試
        model = PatchTSTClassifier(input_dim, num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        save_path = os.path.join(save_dir, f"PatchTST_model_seed{se}.pth")
        fig_path = os.path.join(save_dir, f"PatchTST_train_results_seed{se}.png")

        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path)

        avg_loss, f1, avg_time_per_sample = test_model_with_path_tracking(
            model, test_dataset, criterion, save_dir, save_path, full_dataset, num_classes
        )
        print(f"Seed {se} Test F1: {f1:.4f}, cost {avg_time_per_sample} sec")
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