import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.ion()

gt_path = '/home/barbados/lseg-minimal/examples/results/'
lseg_output_path = '/home/barbados/offroad_terrain_dataset_kaggle/archive/TrainingImages/TrainingImages/EnumMasks/png_0_255/'

gt_files = os.listdir(gt_path)
lseg_files = os.listdir(lseg_output_path)

gt_files.sort()
lseg_files.sort()
dice_list = []
for item1,item2 in zip(gt_files,lseg_files):
    gt_file = cv2.imread(gt_path + item1)[:,:,0]
    gt = np.zeros(shape=(gt_file.shape[0], gt_file.shape[1]))
    gt[gt_file==0]=1
    seg_file = cv2.imread(lseg_output_path + item2)[:,:,0]
    seg = np.zeros(shape=(seg_file.shape[0], seg_file.shape[1]))
    seg[seg_file == 0] = 1

    k = 1 # black
    dice = np.sum(seg[gt == k]) * 2.0 / (np.sum(seg) + np.sum(gt))
    dice_list.append([dice,np.sum(gt)])

dice_list = np.array(dice_list)
counter = np.zeros(10)
list_dice_members = []
x = []
for item in range(len(counter)):
    list_dice_members.append([])
    x.append(0.1*(item+1)*np.max(dice_list[:,1])/(gt.shape[0]*gt.shape[1]))
for item in dice_list:
    counter[int((len(counter)-1)*item[1]/np.max(dice_list[:,1])*0.99)] += 1
    list_dice_members[int((len(counter)-1)*item[1]/np.max(dice_list[:,1]))].append(item[0])
print('hey')
plt.scatter(x,counter)
plt.xlabel("Percentage coverage of the entire image")
plt.title('Distribution of actual segment data throughout the entire image')
plt.ylabel('Number of samples')
plt.figure()
mean_values = []
var_values = []
for item in list_dice_members:
    mean_values.append(np.array(item).mean())
    var_values.append(np.sqrt(np.array(item).var()))

plt.errorbar(x, mean_values, var_values, fmt='-o')#linestyle='None', marker='^')
plt.title('Outcomes of LSeg Considering the Dice Metric')
plt.xlabel("Percentage coverage of the entire image")
plt.ylabel("Mean and standard deviation")
plt.grid()
