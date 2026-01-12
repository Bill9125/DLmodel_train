import torch, os,json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

def reorganize(row, idx, del_col):
    data = []
    idx_label = {'correct': idx, 'wrists_bending_backward': idx+1, 'tilting_to_the_left': idx+2, 'tilting_to_the_right': idx+3, 'elbows_flaring': idx+4, 'scapular_protraction': idx+5}
    if del_col:
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
    else:
        data = row.iloc[0:idx].values.astype(float).tolist()
    label = row.iloc[idx_label['wrists_bending_backward']:idx_label['scapular_protraction']+1].astype(int).tolist()
    no_wrist_label = row.iloc[idx_label['tilting_to_the_left']:idx_label['scapular_protraction']+1].astype(int).tolist()
    
    return data, label, no_wrist_label

def label2str(ground_truth):
    if isinstance(ground_truth, list):
        gt_key = "_".join(map(str, ground_truth))  # e.g. [1,2] → "1_2"
    else:
        gt_key = str(ground_truth)
    return gt_key

def csv2json(dataset_root, split_index, del_col):
    df = pd.read_csv(dataset_root, skiprows=1)
    counter = 0
    tmp_data = []
    # 巢狀字典結構: subjects_data[number][ground_truth][i] = tensor
    subjects_data = defaultdict(lambda: defaultdict(dict))
    subjects_data_no_wrist = defaultdict(lambda: defaultdict(dict))

    for _, row in df.iterrows():
        # 拿掉壓手腕資料
        if row.iloc[split_index+1] == 1:
            continue
        data, label, no_wrist_label = reorganize(row, split_index+1, del_col)
        tmp_data.append(data)

        counter += 1
        if counter == 100:
            path = row.iloc[-2]
            number = row.iloc[-1]
            gt_key = label2str(label)
            no_wrist_gt_key = label2str(no_wrist_label)
            subjects_data[path][gt_key][number] = tmp_data
            subjects_data_no_wrist[path][no_wrist_gt_key][number] = tmp_data
            tmp_data = []
            counter = 0
    return subjects_data, subjects_data_no_wrist

def save_json(data, dir):
    os.makedirs(os.path.join(dir, 'wrist'), exist_ok=True)
    json.dump(data, open(os.path.join(dir, 'wrist', 'data.json'), 'w'), indent=4)

def save_json_no_wrist(data, dir):
    os.makedirs(os.path.join(dir, 'no_wrist'), exist_ok=True)
    json.dump(data, open(os.path.join(dir, 'no_wrist', 'data.json'), 'w'), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=os.path.join(os.getcwd(), 'data', 'benchpress', 'step5_normalized.csv'))
    args = parser.parse_args()
    save_dir = os.path.dirname(args.data_path)
    subjects_data, subjects_data_no_wrist = csv2json(args.data_path, split_index=52, del_col=False)
    save_json(subjects_data, save_dir)
    save_json_no_wrist(subjects_data_no_wrist, save_dir)