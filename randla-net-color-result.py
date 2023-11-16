import open3d as o3d
from helper_ply import read_ply
import numpy as np
import os
import pandas as pd


# 0: 绿色 1: 蓝色 2: 青色 3: 黄色 4: 紫色 5: 淡蓝色 6: 淡黄色 7: 淡紫色 8: 红色 9: 淡红色 10: 淡绿色  #11 LightBlue 12: 深灰色
color_map = {   
    0:	[0,255,0],
    1:	[0,0,255],
    2:	[0,255,255],
    3:  [255,255,0],
    4:  [255,0,255],
    5:  [100,100,255],
    6:  [200,200,100],
    7:  [170,120,200],
    8:  [255,0,0],
    9:  [200,100,100],
    10: [10,200,100],
    11: [173,216,230],  # LightBlue
    12: [50,50,50]
}

# 读取点云数据
filefolder="/mnt/disk2/datasets/S3DIS/original_ply/" #原始点云数据


# 预测结果数据
result_folder="/mnt/disk2/yscode/ys_code/RandLA-Net-Pytorch-New/test_output_S3DIS/2023-11-09_18-00-36/val_preds" 

#ply_files = [os.path.join(filefolder, f) for f in os.listdir(filefolder) if f.endswith('.ply')]
pred_files = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.endswith('.ply')]

count=1
for f in pred_files:
    #filename=f.split('/')[-1].split('.')[0] #对所有结果进行可视化
    filename="Area_5_office_8" #需要可视化的文件名称，对单个进行可视化
    print("Processing:%d"%count,filename)
    #filename="Area_5_office_8"
    original_ply=os.path.join(filefolder, filename+".ply")
    pred_label_ply=os.path.join(result_folder, filename+".ply")

    original_data=read_ply(original_ply)
    pred_data=read_ply(pred_label_ply)

    print(original_data.shape )
    print(pred_data.shape)

    xyz=np.vstack((original_data['x'],original_data['y'],original_data['z'])).T
    
    preds=pred_data['pred']
    
    colors=[color_map[int(pred)] for pred in preds]
    
    # 创建一个包含点云数据的DataFrame
    df = pd.DataFrame({
    'x': xyz[:, 0],
    'y': xyz[:, 1],
    'z': xyz[:, 2],
    'r': np.array(colors)[:, 0],
    'g': np.array(colors)[:, 1],
    'b': np.array(colors)[:, 2],
    'pred': preds
    })
    # 将DataFrame保存为CSV文件
    out_filename = filename + ".csv" # 保存为csv文件 N*7, xyz,rgb,pred
    df.to_csv(out_filename, index=False)
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # print(pcd)
    # out_filename = filename + ".pcd"

    # o3d.io.write_point_cloud(out_filename, pcd, write_ascii=True)
    count+=1
    break


