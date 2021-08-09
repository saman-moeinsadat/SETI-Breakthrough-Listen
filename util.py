import os
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
import torch
import librosa



def dataset_build(train_val_ratio=0.1):
    path_project = str(((Path(__file__).parent).resolve().parent).resolve())
    path_data = path_project + '/seti-breakthrough-listen'

    data = pd.read_csv(path_data+'/train_labels.csv')
    len_dst = len(data)
    len_val = int(len_dst * train_val_ratio) 
    target_1 = data.loc[data.target==1].id
    target_0 = data.loc[data.target==0].id
    len_val_1 = int(train_val_ratio * len(target_1))
    len_val_0 = int(train_val_ratio * len(target_0))
    val_1 = np.random.choice(target_1, len_val_1, replace=False)
    val_0 = np.random.choice(target_0, len_val_0, replace=False)
    data_val = np.concatenate((val_1, val_0), axis=0)
    np.random.shuffle(data_val)
    train_count, val_count = 0, 0
    train_num, val_num, = 1, 1

    val = np.zeros(shape=(1000, 12, 273, 256))
    val_label = []
    train = np.zeros(shape=(1000, 12, 273, 256))
    train_label = []

    for dir in os.listdir(path_data+'/train/'):
        for file in os.listdir(path_data+'/train/'+dir):
            file_name = file.split(".")[0]

            if file_name in data_val:
                if val_count < 1000:
                    np_array = np.load(path_data+'/train/'+dir+'/'+file)
                    val[val_count, : 6, :, :] = np_array
                    np_array = np.transpose(np_array, (0, 2, 1))
                    mel = np.array([librosa.feature.melspectrogram(S=x, sr=192000, n_mels=256, fmax=96000) for x in np_array])
                    mel = np.transpose(mel, (0, 2, 1))
                    val[val_count, 6:, :, :] = mel
                    val_label.append(list(data[data.id==file_name]['target'])[0])
                    if val_count == 999 and val_num == 6:
                        val = torch.from_numpy(val).float()
                        val_label = torch.from_numpy(np.array(val_label))
                        val_dst = TensorDataset(val, val_label)
                        torch.save(val_dst, '%s/datasets_mel/val/val_%d.pt' % (path_data, 6))
                        print(val_num)


                    val_count += 1
                     
                else:
                    val_count = 0
                    val = torch.from_numpy(val).float()
                    val_label = torch.from_numpy(np.array(val_label))
                    val_dst = TensorDataset(val, val_label)
                    torch.save(val_dst, '%s/datasets_mel/val/val_%d.pt' % (path_data, val_num))
                    print(val_num)

                    val_num += 1

                    val = np.zeros(shape=(1000, 12, 273, 256))
                    val_label = []

                    np_array = np.load(path_data+'/train/'+dir+'/'+file)
                    val[val_count, : 6, :, :] = np_array
                    np_array = np.transpose(np_array, (0, 2, 1))
                    mel = np.array([librosa.feature.melspectrogram(S=x, sr=192000, n_mels=256, fmax=96000) for x in np_array])
                    mel = np.transpose(mel, (0, 2, 1))
                    val[val_count, 6:, :, :] = mel
                    val_label.append(list(data[data.id==file_name]['target'])[0])
                    val_count += 1
            
            else:
                if train_count < 1000:
                    np_array = np.load(path_data+'/train/'+dir+'/'+file)
                    train[train_count, : 6, :, :] = np_array
                    np_array = np.transpose(np_array, (0, 2, 1))
                    mel = np.array([librosa.feature.melspectrogram(S=x, sr=192000, n_mels=256, fmax=96000) for x in np_array])
                    mel = np.transpose(mel, (0, 2, 1))
                    train[train_count, 6:, :, :] = mel
                    train_label.append(list(data[data.id==file_name]['target'])[0])

                    if train_count == 999 and train_num == 54:
                        train = torch.from_numpy(train).float()
                        train_label = torch.from_numpy(np.array(train_label))
                        train_dst = TensorDataset(train, train_label)
                        torch.save(train_dst, '%s/datasets_mel/train/train_%d.pt' % (path_data, 54))
                        print(train_num)

                    train_count += 1
                else:
                    train_count = 0
                    print(train.shape)
                    train = torch.from_numpy(train).float()
                    train_label = torch.from_numpy(np.array(train_label))
                    train_dst = TensorDataset(train, train_label)
                    torch.save(train_dst, '%s/datasets_mel/train/train_%d.pt' % (path_data, train_num))
                    print(train_num)

                    train_num += 1

                    train = np.zeros(shape=(1000, 12, 273, 256))
                    train_label = []
                    
                    np_array = np.load(path_data+'/train/'+dir+'/'+file)
                    train[train_count, : 6, :, :] = np_array
                    np_array = np.transpose(np_array, (0, 2, 1))
                    mel = np.array([librosa.feature.melspectrogram(S=x, sr=192000, n_mels=256, fmax=96000) for x in np_array])
                    mel = np.transpose(mel, (0, 2, 1))
                    train[train_count, 6:, :, :] = mel
                    train_label.append(list(data[data.id==file_name]['target'])[0])
                    train_count += 1


                
    
    





if __name__ == "__main__":
    dataset_build()