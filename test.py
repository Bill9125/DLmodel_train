import torch
import time
from tools import set_seed, f1_score, write_results
import matplotlib.pyplot as plt
import os, json
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
from models import ResNet32, BiLSTMModel
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from dataset import *

def test_model_with_path_tracking(model, test_loader, criterion, txt_dir, save_path, title = 'Confusion Matrix'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()
    
    total_loss, total_time = 0.0, 0.0  
    y_true, y_pred = [], []
    
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

                    
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    avg_time_per_sample = total_time / len(y_true)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred) 
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.savefig(f"{txt_dir}/confusion_matrix.png")
    plt.close()

    return avg_loss, f1, acc, avg_time_per_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT_class',type=int)
    parser.add_argument('--model', type=str, default='BiLSTM', choices=['Resnet32', 'BiLSTM'], help='Model type to use for training')
    parser.add_argument('--data',type=str)
    args = parser.parse_args()
    GT_class = args.GT_class
    model_type = args.model
    data_file = args.data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = {0: 'tilting_to_the_left', 1: 'tilting_to_the_right', 2: 'elbows_flaring', 3: 'scapular_protraction'}
    
    data_path = os.path.join(os.getcwd(), 'data', data_file, 'data.json')
    save_dir = os.path.join(os.getcwd(), 'models', 'benchpress', model_type, data_file, 'no_wrist_press', class_names[GT_class])
    os.makedirs(save_dir, exist_ok=True)
    print(f'read {data_path} as dataset ...')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    all_avg_times = []
    all_acc = []
    seeds = [42, 2023, 7, 88, 100, 999]

    for se in seeds:
        set_seed(se)
        
        random_keys = random.sample(list(map(int, data.keys())), 20)
        test_data = {str(k): data[str(k)] for k in random_keys}
        train_data = {str(k): data[str(k)] for k in data if int(k) not in random_keys}
        
        full_train_dataset = Dataset_Benchpress(train_data, GT_class)
        test_dataset = Dataset_Benchpress(test_data, GT_class)
        
        category_ratio = full_train_dataset.get_ratio()
        P_ratio = category_ratio[1]
        input_dim = full_train_dataset.dim
        print('input_dim',input_dim)
        print(f'Category : {category_ratio}')
        
        train_size = int(0.85 * len(full_train_dataset))
        valid_size = int(len(full_train_dataset)) - train_size
        test_size = int(len(test_dataset))
        print(f'total training size : {len(full_train_dataset)}')
        print(f'train_size : {train_size}, valid_size : {valid_size}, test_size : {test_size}')
        
        # 分割資料
        gen = torch.Generator().manual_seed(se)  # 為每個seed創建獨立生成器
        train_indices, valid_indices = random_split(
            range(len(full_train_dataset)), [train_size, valid_size],
            generator=gen
        )
        train_dataset = ResnetSubset(full_train_dataset, train_indices, transform=True)
        valid_dataset = ResnetSubset(full_train_dataset, valid_indices, transform=False)
        
        train_labels = [full_train_dataset.labels[i] for i in train_dataset.indices]

        # 建立 Weighted Sampler
        class_weights = [1.0 / sum(np.array(train_labels) == i) for i in range(2)]
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(train_dataset),
                                replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
        valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # 訓練與測試
        if model_type == 'BiLSTM':
            model = BiLSTMModel(input_dim).to(device)
        elif model_type == 'Resnet32':
            model = ResNet32(input_dim).to(device)
        class_counts = torch.tensor([P_ratio, 1 - P_ratio])
        criterion = CrossEntropyLoss(weight=(1.0 / class_counts).to(device))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        save_path = os.path.join(save_dir, f"{model_type}_model_seed{se}.pth")
        txt_dir = os.path.join(save_dir, f"{model_type}_train_results_seed{se}_results")
        fig_path = os.path.join(txt_dir, f"{model_type}_train_results_seed{se}.png")
        os.makedirs(txt_dir, exist_ok=True)

        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path)

        avg_loss, f1, acc, avg_time_per_sample = test_model_with_path_tracking(
            model, test_loader, criterion, txt_dir, save_path, title=class_names[GT_class]
        )

        print(f"Seed {se} Test F1: {f1:.4f}")
        all_f1_scores.append(f1)
        all_avg_times.append(avg_time_per_sample)
        all_acc.append(acc)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = se
            best_model_path = save_path

    write_results(model, input_dim, category_ratio, seeds, all_f1_scores, all_avg_times, all_acc, best_f1, best_seed, best_model_path, save_dir)