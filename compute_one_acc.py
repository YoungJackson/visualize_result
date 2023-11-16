
import torch

from  helper_ply import read_ply
import os
import numpy as np

def class_accuracy(preds, labels, num_classes):
    
    true_labels = labels.tolist()
    predicted_labels = preds.tolist()
     # 计算每个类别的准确度
    TP = [0] * num_classes
    FP = [0] * num_classes

    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            TP[true_labels[i]] += 1
        else:
            FP[predicted_labels[i]] += 1
    acc = []
    for i in range(num_classes):
        # 检查TP[i] + FP[i]是否为0
        acc.append(TP[i] / (TP[i] + FP[i]) if TP[i] + FP[i] != 0 else 0)

    print("class acc is:",acc)
    macro_acc = np.mean(acc)

    print("Macro acc: %.2f%%" % (macro_acc * 100))
    
    return acc, macro_acc

if __name__=="__main__":
   
    # 计算单个点云每一类的准确率
    num_classes=13
    filefolder="../test_output_S3DIS/2023-11-03_12-32-53/val_preds"#普通结果
    #filefolder="../test_output_S3DIS/2023-11-09_18-00-36/val_preds"#best result 
    filename="Area_5_office_8.ply"
    
    filepath = os.path.join(filefolder, filename)
    data=read_ply(filepath)
    
    preds =data['pred']
    labels=data['label']
    acc,macc=class_accuracy(preds, labels, num_classes)
    