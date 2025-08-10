import torch, os,json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def reorganize(row, idx):
    data = []
    idx_label = {'correct': idx, 'wrists_bending_backward': idx+1, 'tilting_to_the_left': idx+2, 'tilting_to_the_right': idx+3, 'elbows_flaring': idx+4, 'scapular_protraction': idx+5}
    # 定義你要擷取的欄位範圍或索引
    slices = [
        (0, 3), (4, 8), (9, 13), (14, 18), (19, 23), (24, 28),  # data_1 to data_6
        29,                                                    # data_7
        (35, 38),                                              # data_8
        39,                                                    # data_9
        (50, 53), (54, 58), (59, 63), (64, 68),                # data_10 to data_13
        69,                                                    # data_14
        (75, 78),                                              # data_15
        79,                                                    # data_16
        (90, 93),                                              # data_17
        94                                                     # data_18
    ]
    for s in slices:
        if isinstance(s, tuple):
            data.extend(row.iloc[s[0]:s[1]].values.astype(float).tolist())
        else:
            data.append(row.iloc[s])
    label = row.iloc[idx_label['tilting_to_the_left']:idx_label['scapular_protraction']+1].astype(int).tolist()
    return data, label

def label2str(ground_truth):
    if isinstance(ground_truth, list):
        gt_key = "_".join(map(str, ground_truth))  # e.g. [1,2] → "1_2"
    else:
        gt_key = str(ground_truth)
    return gt_key

def csv2json(dataset_root, split_index):
    df = pd.read_csv(dataset_root, skiprows=1)
    i = 0
    counter = 0
    old_number = -1
    tmp_data = []
    # 巢狀字典結構: subjects_data[number][ground_truth][i] = tensor
    subjects_data = defaultdict(lambda: defaultdict(dict))

    for _, row in df.iterrows():
        # 拿掉壓手腕資料
        if row.iloc[split_index+1] == 1:
            continue
        data, ground_truth = reorganize(row, split_index)
        tmp_data.append(data)

        counter += 1
        if counter == 100:
            path = row.iloc[-1]
            path_str = Path(path.replace("\\", "/")) # 先把反斜線統一成正斜線
            number = path_str.parts[3]
            if old_number != number: # 換下一組
                i = 0
            gt_key = label2str(ground_truth)
            subjects_data[number][gt_key][i] = tmp_data
            i+=1
            old_number = number
            tmp_data = []
            counter = 0
    return subjects_data
            
def save_json(data, dir):
    json.dump(data, open(os.path.join(dir, 'data.json'), 'w'), indent=4)

csv_path = os.path.join(os.getcwd(), 'data', 'BP_data_new_skeleton', 'bench_press_multilabel_cut4.csv')
save_dir = os.path.dirname(csv_path)
subjects_data = csv2json(csv_path, split_index=95)
save_json(subjects_data, save_dir)