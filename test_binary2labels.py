import torch
import time
from sklearn.metrics import f1_score
from tools import multilabel_confusion_matrix_mix, plot_custom_confusion_matrix, set_seed
import os, json
from torch.utils.data import DataLoader, random_split
import argparse
from models import ResNet32, BiLSTMModel
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from dataset.dataset import *

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
    parser.add_argument('--output_dir',type=str)
    args = parser.parse_args()
    model_type = args.model
    data = args.data
    output_dir = args.output_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    dir = os.path.join(os.getcwd(), 'models', 'benchpress', model_type, 'BP_data_new_skeleton', 'no_wrist_press')
    class_names = {0: 'tilting_to_the_right', 1: 'tilting_to_the_left', 2: 'elbows_flaring', 3: 'scapular_protraction'}
        
    results_dir = os.path.join(dir, output_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    data_path = os.path.join(os.getcwd(), 'data', data, 'data.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_f1 = []
    seeds = [42, 2023, 7, 88, 100, 999]

    for se in seeds:
        print(f'Testing seed: {se}')
        set_seed(se)
        y_ts = {i: [] for i in range(len(class_names))}
        y_ps = {i: [] for i in range(len(class_names))}
        for GT_class, class_name in class_names.items():
            model_path = os.path.join(dir, class_name, f"{model_type}_model_seed{se}.pth")
            
            # 讀取 dataset
            full_dataset = Dataset_Benchpress(data_path, GT_class)
            train_size = int(0.75 * len(full_dataset))
            valid_size = int(0.15 * len(full_dataset))
            test_size = len(full_dataset) - train_size - valid_size
            
            # 分割資料
            gen = torch.Generator().manual_seed(se)  # 為每個seed創建獨立生成器
            train_indices, valid_indices, test_indices = random_split(
                range(len(full_dataset)), [train_size, valid_size, test_size],
                generator=gen
            )
            
            train_dataset = ResnetSubset(full_dataset, train_indices, transform=True)
            valid_dataset = ResnetSubset(full_dataset, valid_indices, transform=False)
            test_dataset  = ResnetSubset(full_dataset, test_indices, transform=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            category_ratio = full_dataset.get_ratio()
            P_ratio = category_ratio[1]
            input_dim = full_dataset.dim

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
            y_ts[GT_class].extend([y.astype(int) for y in y_true])
            y_ps[GT_class].extend([y.astype(int) for y in y_pred])
            
        y_ts = np.array(list(y_ts.values())).T
        y_ps = np.array(list(y_ps.values())).T
        f1 = f1_score(y_ts, y_ps, average='macro')
        all_f1.append(f1)

        cm = multilabel_confusion_matrix_mix(y_ts, y_ps, len(class_names))
        print('Finished calculating confusion matrix for seed:', se)
        plot_custom_confusion_matrix(cm, ['right'] + list(class_names.values()), f"{results_dir}/confusion_matrix_mix_{se}.png", f1)
        
    with open(f"{results_dir}/test_summary.json", 'w') as f:
        json.dump({
            'model': model_type,
            'training data': 'BP_data_new_skeleton',
            'testing data': data,
            'best seed': seeds[np.argmax(all_f1)],
            'f1 score': np.max(np.array(all_f1))
        }, f, indent=4)