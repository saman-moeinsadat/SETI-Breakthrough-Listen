import os
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
import torch
import librosa


class numpy_array:
    def __init__(self, shape):
        self.shape = shape
        self.array = np.zeros(shape=self.shape)
    
    def transform(self, np_array, count):
        self.array[count, : 6, :, :] = np_array
        np_array = np.transpose(np_array, (0, 2, 1))
        mel = np.array([
            librosa.feature.melspectrogram(S=x, sr=192000, n_mels=256, fmax=96000) \
            for x in np_array
        ])
        mel = np.transpose(mel, (0, 2, 1))
        self.array[count, 6:, :, :] = mel
    
    def save_dataset(self, label, num, path, name):
        self.array = torch.from_numpy(self.array).float()
        label = torch.from_numpy(np.array(label))
        val_dst = TensorDataset(self.array, label)
        torch.save(val_dst, '%s/datasets_mel/%s/%s_%d.pt' % (path, name, name, num))
        print("%s dataset, part %d, is saved." % (name, num))
        print("------------------------------")
    
    def reset(self):
        self.array = np.zeros(shape=self.shape)





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

    val = numpy_array((1000, 12, 273, 256))
    val_label = []
    train = numpy_array((1000, 12, 273, 256))
    train_label = []

    for dir in os.listdir(path_data+'/train/'):
        for file in os.listdir(path_data+'/train/'+dir):
            file_name = file.split(".")[0]

            if file_name in data_val:
                if val_count < 1000:

                    np_array = np.load(path_data+'/train/'+dir+'/'+file)
                    val.transform(np_array=np_array, count=val_count)
                    val_label.append(list(data[data.id==file_name]['target'])[0])

                    if val_count == 999 and val_num == 6:
                        val.save_dataset(
                            label=val_label, num=val_num,
                            path=path_data, name='val'
                        )

                    val_count += 1
                     
                else:
                    val_count = 0
                    val.save_dataset(
                        label=val_label, num=val_num,
                        path=path_data, name='val'
                    )

                    val_num += 1

                    val.reset()
                    val_label = []

                    np_array = np.load(path_data+'/train/'+dir+'/'+file)
                    val.transform(np_array=np_array, count=val_count)
                    val_label.append(list(data[data.id==file_name]['target'])[0])

                    val_count += 1
            
            else:
                if train_count < 1000:

                    np_array = np.load(path_data+'/train/'+dir+'/'+file)
                    train.transform(np_array=np_array, count=train_count)
                    train_label.append(list(data[data.id==file_name]['target'])[0])

                    if train_count == 999 and train_num == 54:
                        train.save_dataset(
                            label=train_label, num=train_num,
                            path=path_data, name='train'
                        )

                    train_count += 1

                else:
                    train_count = 0
                    train.save_dataset(
                        label=train_label, num=train_num,
                        path=path_data, name='train'
                    )

                    train_num += 1

                    train.reset()
                    train_label = []
                    
                    np_array = np.load(path_data+'/train/'+dir+'/'+file)
                    train.transform(np_array=np_array, count=train_count)
                    train_label.append(list(data[data.id==file_name]['target'])[0])

                    train_count += 1


                
    
    





if __name__ == "__main__":
    dataset_build()