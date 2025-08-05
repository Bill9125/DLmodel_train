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
from sklearn.metrics import accuracy_score
from dataset import *

def test_model_with_path_tracking(model, test_loader, criterion, save_path, full_dataset, title = 'Confusion Matrix'):
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

    return y_true, y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Resnet32', choices=['Resnet32', 'BiLSTM'], help='Model type to use for training')
    parser.add_argument('--data',type=str)
    args = parser.parse_args()
    model_type = args.model
    data = args.data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    class_names = {0: 'tilting_to_the_left', 1: 'tilting_to_the_right', 2: 'elbows_flaring', 3: 'scapular_protraction'}
    data_path = os.path.join(os.getcwd(), 'data', data)

    y_ts = []
    y_ps = []
    
    dir = os.path.join(os.getcwd(), 'models', 'benchpress', model_type, 'BP_data_new_skeleton', 'no_wrist_press')
    
    seeds = [42, 2023, 7, 88, 100, 999]

    for se in seeds:
        for GT_class, class_name in class_names.items():
            model_path = os.path.join(dir, class_name, f"{model_type}_model_seed{se}.pth")
            # 讀取 dataset
            full_dataset = Dataset_Benchpress(data_path, GT_class)
            category_ratio = full_dataset.get_ratio()
            P_ratio = category_ratio[1]
            input_dim = full_dataset.dim
            print('input_dim',input_dim)
            print(f'Category : {category_ratio}')

            test_dataset  = ResnetSubset(full_dataset, [i for i in range(len(full_dataset))], transform=False)
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

            # 測試
            if model_type == 'BiLSTM':
                model = BiLSTMModel(input_dim).to(device)
            elif model_type == 'Resnet32':
                model = ResNet32(input_dim).to(device)
            class_counts = torch.tensor([P_ratio, 1 - P_ratio])
            criterion = CrossEntropyLoss(weight=(1.0 / class_counts).to(device))

            y_true, y_pred = test_model_with_path_tracking(
                model, test_loader, criterion, model_path, full_dataset, title=class_names[GT_class]
            )
            y_ts.extend(y_true)
            y_ps.extend(y_pred)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure(figsize=(6, 6))
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix for All Classes')
        plt.savefig(os.path.join(dir, f'confusion_matrix_all_classes_seed{se}.png'))
        plt.close()