import pandas as pd
import numpy as np

# All y list
label1 = pd.read_excel('./data/label_1.xlsx')
label2 = pd.read_excel('./data/label_2.xlsx')
label3 = pd.read_excel('./data/label_3.xlsx')
label4 = pd.read_excel('./data/label_4.xlsx')
label5 = pd.read_excel('./data/label_5.xlsx')
label6 = pd.read_excel('./data/label_6.xlsx')
label7 = pd.read_excel('./data/label_7.xlsx')
label8 = pd.read_excel('./data/label_8.xlsx')
label9 = pd.read_excel('./data/label_9.xlsx')
label10 = pd.read_excel('./data/label_10.xlsx')
label12 = pd.read_excel('./data/label_12.xlsx')
label13 = pd.read_excel('./data/label_13.xlsx')

all_labels = [label1, label2, label3, label4, label5, label6, label7, label8, label9, label10, label12, label13]

for label in all_labels:
    label['target'] = np.sqrt(label['Storage modulus']**2 + label['Loss modulus']**2)

targets = [label['target'].values,
           label2['target'].values, 
           label3['target'].values, 
           label4['target'].values, 
           label5['target'].values, 
           label6['target'].values,
           label6['target'].values,
           label7['target'].values,
           label8['target'].values,
           label9['target'].values,
           label10['target'].values,
           label12['target'].values,
           label13['target'].values,
           ]

t_min = np.min(targets, axis=0)
t_max = np.max(targets, axis=0)
targets_scaled = (targets-t_min)/(t_max-t_min)

print(targets)
print('#'*30)
print(t_min)
print(t_max)
print('#'*30)
print(targets_scaled)