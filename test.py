import torch
import time
from tools import set_seed, f1_score, write_results
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
from models import ResNet32, BiLSTMModel
from torch.nn import CrossEntropyLoss
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score

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
    acc = accuracy_score(y_true, y_pred) 

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

    return avg_loss, f1, acc, avg_time_per_sample, false_positives, false_negatives

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
            save_dir = os.path.join(os.getcwd(), 'models', f'dd2voz_vision1_body/{GT_class}')
            
        elif F_type == '3D':
            from dataset import Dataset_3D
            datasets_path = os.path.join(os.getcwd(), '3D_traindata')
            full_dataset = Dataset_3D(datasets_path, GT_class)
            save_dir = f'./models_3D/{GT_class}'
        input_dim = full_dataset.dim
        print('input_dim',input_dim)
    
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

        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # 訓練與測試
        model = ResNet32(input_dim).to(device)
        P_ratio = category_ratio[GT_class]
        class_counts = torch.tensor([P_ratio, 1 - P_ratio])
        criterion = CrossEntropyLoss(weight=(1.0 / class_counts).to(device))

        save_path = os.path.join(save_dir, f"ResNet32_model_seed{se}.pth")
        fig_path = os.path.join(save_dir, f"ResNet32_train_results_seed{se}.png")

        avg_loss, f1, acc, avg_time_per_sample, false_positives, false_negatives = test_model_with_path_tracking(
            model, test_loader, test_dataset, criterion, save_dir, save_path, full_dataset
        )

        print(f"Seed {se} Test F1: {f1:.4f}")
        all_f1_scores.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = se
            best_model_path = save_path
    
    write_results(seeds, all_f1_scores, best_f1, best_seed, best_model_path, save_dir)
