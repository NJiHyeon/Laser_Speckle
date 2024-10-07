import os
import pickle 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split


# X_list
all_names = []
fnames = os.listdir('../frames')
for f in fnames:
    if f.split('_')[0] == '2':
        pass
    else:
        all_names.append(f)
all_X_list = all_names
all_X_list.sort()


# y_list
label1 = pd.read_excel('../data/label_1.xlsx')
label3 = pd.read_excel('../data/label_3.xlsx')
label4 = pd.read_excel('../data/label_4.xls', sheet_name=1)
label5 = pd.read_excel('../data/label_5.xls', sheet_name=1)
label6 = pd.read_excel('../data/label_6.xls', sheet_name=1)
label7 = pd.read_excel('../data/label_7.xls', sheet_name=1)
label8 = pd.read_excel('../data/label_8.xls', sheet_name=1)
label9 = pd.read_excel('../data/label_9.xls', sheet_name=1)
label10 = pd.read_excel('../data/label_10.xls', sheet_name=1)
label12 = pd.read_excel('../data/label_12.xls', sheet_name=1)
label13 = pd.read_excel('../data/label_13.xls', sheet_name=1)
label14 = pd.read_excel('../data/label_14.xls', sheet_name=1)

all_labels = [label1, label3, label4, label5, label8, label9, label10, label12, label13, label14]

for label in all_labels :
    label['target'] = np.sqrt(label['Storage modulus']**2 + label['Loss modulus']**2)

targets = [label10.iloc[[10,15,20,25,30], :]['target'].values, 
           label12.iloc[[10,15,20,25,30], :]['target'].values, 
           label13.iloc[[10,15,20,25,30], :]['target'].values, 
           label14.iloc[[10,15,20,25,30], :]['target'].values, 
           label1.iloc[[2,7,12,17,22], :]['target'].values, 
           label3.iloc[[10,15,20,25,30], :]['target'].values, 
           label4.iloc[[10,15,20,25,30], :]['target'].values, 
           label5.iloc[[10,15,20,25,30], :]['target'].values, 
        #    label6.iloc[[10,15,20,25,30], :]['target'].values, 
        #    label7.iloc[[10,15,20,25,30], :]['target'].values, 
           label8.iloc[[10,15,20,25,30], :]['target'].values, 
           label9.iloc[[10,15,20,25,30], :]['target'].values]


# 각 동영상 별로 길이 저장
list_length = []
for num in ['10', '12', '13', '14']:
    num_vid = 0
    path = os.path.join('../data', num)
    for pos in os.listdir(path):
        full_path = os.path.join(path, pos)
        num_vid += len(os.listdir(full_path))
    list_length.append(num_vid)

for i in range(2): #['1', '2']
    list_length.append(60)

for num in ['4', '5', '8', '9']: # for num in ['4', '5', '6', '7', '8', '9']:
    num_vid = 0
    path = os.path.join('../data', num)
    for pos in os.listdir(path):
        full_path = os.path.join(path, pos)
        num_vid += len(os.listdir(full_path))
    list_length.append(num_vid)

'''
list_length : [60, 60, 53, 60, 60, 60, 60, 60, 60, 60]
'''

all_y_list = []
for i, target in enumerate(targets):
    for i in range(list_length[i]):
        all_y_list.append(target)

all_y_list = np.array(all_y_list)


# train, test split
all_train_list = []
all_val_list = []
all_test_list = []
all_train_label = []
all_val_label = []
all_test_label = []
start_idx = 0
for length in list_length:
    x_list = all_X_list[start_idx : start_idx+length]
    y_list = all_y_list[start_idx : start_idx+length]
    start_idx += length
    train_list, temp_X, train_label, temp_y = train_test_split(x_list, y_list, test_size=0.2, random_state=42, shuffle=True, stratify=y_list)
    val_list, test_list, val_label, test_label = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42, shuffle=True, stratify=temp_y)
    all_train_list.extend(train_list)
    all_val_list.extend(val_list)
    all_test_list.extend(test_list)
    all_train_label.extend(train_label)
    all_val_label.extend(val_label)
    all_test_label.extend(test_label)

'''
len(all_train_list), len(all_val_list), len(all_test_list)
(474, 59, 60)
'''

with open('train_split.pickle', 'wb') as f :
    pickle.dump([all_train_list, all_train_label], f)
f.close()

with open('valid_split.pickle', 'wb') as f:
    pickle.dump([all_val_list, all_val_label], f)
f.close()

with open('test_split.pickle', 'wb') as f:
    pickle.dump([all_test_list, all_test_label], f)
f.close()

'''
len(all_train_list), len(all_val_list), len(all_test_list)
(570, 71, 72)
'''
