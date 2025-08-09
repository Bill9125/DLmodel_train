from torch.utils.data import Dataset
import os, glob
import torch
import pandas as pd
import numpy as np
import random
    
class Dataset_Benchpress(Dataset):
    def __init__(self, dataset_root, GT_class):
        self.sample_paths = []  
        self.features = []
        self.labels = []
        self.count_info = []  # [negative_count, positive_count]
        
        df = pd.read_csv(dataset_root, skiprows=1)
        counter = 0
        tmp_data = []
        label_counter = {0: 0, 1: 0}

        for _, row in df.iterrows():
            row.iloc[[54, 55]] = row.iloc[[55, 54]]
            # 拿掉壓手腕資料
            # if row.iloc[96] == 1:
            #     continue
            # 定義你要擷取的欄位範圍或索引
            # slices = [
            #     (0, 3), (4, 8), (9, 13), (14, 18), (19, 23), (24, 28),  # data_1 to data_6
            #     29,                                                    # data_7
            #     (35, 38),                                              # data_8
            #     39,                                                    # data_9
            #     (50, 53), (54, 58), (59, 63), (64, 68),                # data_10 to data_13
            #     69,                                                    # data_14
            #     (75, 78),                                              # data_15
            #     79,                                                    # data_16
            #     (90, 93),                                              # data_17
            #     94                                                     # data_18
            # ]

            # 自動組合資料
            data = []
            # for s in slices:
            #     if isinstance(s, tuple):
            #         data.extend(row.iloc[s[0]:s[1]].values.astype(float).tolist())
            #     else:
            #         data.append(row.iloc[s])
            
            data = row.iloc[0:52].values.astype(float).tolist()
            label = row.iloc[53 + GT_class + 1]
            # ground_true = row.iloc[96:101].values.astype(int)
            # label = ground_true[GT_class]
            tmp_data.append(data)
            label_counter[label] += 1

            counter += 1
            if counter == 100:
                block = np.array(tmp_data)
                path = row.iloc[-1]
                self.sample_paths.append(path)
                self.features.append(torch.tensor(block).float())
                self.labels.append(torch.tensor(label).long())

                tmp_data = []
                counter = 0

        self.features = torch.stack(self.features)
        self.labels = torch.stack(self.labels)
        self.dim = self.features.shape[-1]

        self.count_info = [label_counter[0], label_counter[1]]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y, idx
                    
    def get_sample_path(self, idx):
        return self.sample_paths[idx]
    
    def get_ratio(self):
        category_ratio = {}
        total = sum(self.count_info)
        if total > 0:
            category_ratio[0] = self.count_info[0] / total
            category_ratio[1] = self.count_info[1] / total
        return category_ratio

class ResnetSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=False):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y, true_idx = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.time_stretch(x, random.uniform(0.8, 1.2))
            x = self.add_gaussian_noise(x, std=0.01)
        return x, y, true_idx

    def time_stretch(self, x, stretch_factor):
        # 假設 x.shape = (T, F)
        T, F = x.shape
        new_T = int(T * stretch_factor)
        x_stretched = torch.nn.functional.interpolate(
            x.unsqueeze(0).permute(0, 2, 1),  # (1, F, T)
            size=new_T,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1).squeeze(0)  # 回到 (T, F)
        if new_T < T:
            pad = torch.zeros(T - new_T, F, dtype=x.dtype, device=x.device)
            x_stretched = torch.cat([x_stretched, pad], dim=0)
        else:
            x_stretched = x_stretched[:T]
        return x_stretched

    def add_gaussian_noise(self, x, std=0.01):
        noise = torch.randn_like(x) * std
        return x + noise