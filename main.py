import os
### ----------------------deadlift----------------------
# os.system("python train.py --GT_class 2 --F_type 3D")
# os.system("python train.py --GT_class 2 --SHAP abs")
# os.system("python train.py --GT_class 2 --SHAP avg_abs_min")
# os.system("python train.py --GT_class 2 --SHAP avg_min")

# os.system("python train.py --GT_class 3 --F_type 3D")
# os.system("python train.py --GT_class 3 --SHAP abs")
# os.system("python train.py --GT_class 3 --SHAP avg_abs_min")
# os.system("python train.py --GT_class 3 --SHAP avg_min")

# os.system("python train.py --GT_class 4 --F_type 3D")
# os.system("python train.py --GT_class 4 --SHAP abs")
# os.system("python train.py --GT_class 4 --SHAP avg_abs_min")
# os.system("python train.py --GT_class 4 --SHAP avg_min")

# os.system("python train.py --GT_class 5 --F_type 3D")
# os.system("python train.py --GT_class 5 --SHAP abs")
# os.system("python train.py --GT_class 5 --SHAP avg_abs_min")
# os.system("python train.py --GT_class 5 --SHAP avg_min")

# os.system("python train.py --GT_class 2 --sport deadlift --data 3D_Real_Final --model BiLSTM")
# os.system("python train.py --GT_class 3 --sport deadlift --data 3D_Real_Final --model Resnet32")
# os.system("python train.py --GT_class 4 --sport deadlift --data 3D_Real_Final --model Resnet32")
# os.system("python train.py --GT_class 5 --sport deadlift --data 3D_Real_Final --model Resnet32")

### ----------------------benchpress----------------------
os.system("python train.py --GT_class 0 --sport benchpress --model Resnet32 --data BP_data_new_skeleton")
os.system("python train.py --GT_class 1 --sport benchpress --model Resnet32 --data BP_data_new_skeleton")
os.system("python train.py --GT_class 2 --sport benchpress --model Resnet32 --data BP_data_new_skeleton")
os.system("python train.py --GT_class 3 --sport benchpress --model Resnet32 --data BP_data_new_skeleton")
os.system("python train.py --GT_class 4 --sport benchpress --model Resnet32 --data BP_data_new_skeleton")

# os.system("python test.py --GT_class 0 --sport benchpress --model BiLSTM --data subject_test.csv")
# os.system("python test.py --GT_class 1 --sport benchpress --model BiLSTM --data subject_test.csv")
# os.system("python test.py --GT_class 2 --sport benchpress --model BiLSTM --data subject_test.csv")
# os.system("python test.py --GT_class 3 --sport benchpress --model BiLSTM --data subject_test.csv")
# os.system("python test.py --GT_class 4 --sport benchpress --model BiLSTM --data subject_test.csv")
# os.system("python test.py --GT_class 0 --sport benchpress --data BPdata --model Resnet32")
# os.system("python test.py --GT_class 1 --sport benchpress --data BPdata --model Resnet32")
# os.system("python train.py --GT_class 2 --sport benchpress --data BPdata --model Resnet32")
# os.system("python train.py --GT_class 3  --sport benchpress --data BPdata --model Resnet32")
# os.system("python train.py --GT_class 4  --sport benchpress --data BPPdata --model Resnet32")
