import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import copy
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import accuracy_score, roc_auc_score
import torchvision.models as models
from pathlib import Path
import os



def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def model_prepare(pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    model.conv1 = nn.Conv2d(
        6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    model.fc = nn.Linear(
        in_features=512, out_features=2, bias=True
    )
    return model




def train_val(epochs_number=20, lr=0.0001, bs=16):

    path_project = str(((Path(__file__).parent).resolve().parent).resolve())
    path_data = path_project + '/seti-breakthrough-listen'
    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_prepare(pretrained=True)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[round(epochs_number*x) for x in [0.75, 1]],
        gamma=0.1
    )

    scheduler.last_epoch = epochs_number

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0

    cr_weights = torch.FloatTensor([0.1111, 1])

    criterion = nn.BCELoss(weight=cr_weights)
    train_dir = path_data + '/datasets/train/'
    val_dir = path_data + '/datasets/val'

    train_dst = os.listdir(train_dir)
    val_dst = os.listdir(val_dir)


    for epoch in range(epochs_number):

        t1 = time.time()
        print('Epoc {}/{}'.format(epoch, epochs_number - 1))
        print('-' * 10)

        flag_t = False
        flag_v = False

        model.train()
        running_loss_t = 0.0
        running_loss_v = 0.0

        for dst in train_dst:
            dst_path = train_dir + dst
            train_dataloader = DataLoader(torch.load(dst_path), batch_size=bs, shuffle=False)

            for inputs, labels in train_dataloader:

                labels_expanded = torch.zeros(len(inputs), 2, dtype=torch.float64)
                for item in range(len(inputs)):
                    labels_expanded[item, labels[item].item()] = 1.00
                labels_expanded = Variable(labels_expanded)

                inputs = Variable(inputs.float(), requires_grad=True)
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    m = nn.Sigmoid()
                    outputs = m(outputs).double()

                    if not flag_t:
                        outputs_all = outputs
                        detections = preds
                        labels_all = labels
                        flag_t = True
                    else:
                        outputs_all = torch.cat((outputs_all, outputs), 0)
                        detections = torch.cat((detections, preds), 0)
                        labels_all = torch.cat((labels_all, labels), 0)
                    
                    loss = criterion(outputs, labels_expanded)
                    loss.backward()
                    clip_gradient(model, 0.5)
                    optimizer.step()

                running_loss_t += loss.item() * inputs.size()[0]
        
        epoch_loss = running_loss_t / 54000
        epoch_f1 = f1_score(labels_all, detections)
        epoch_recall = recall_score(labels_all, detections)
        epoch_precision = precision_score(labels_all, detections)
        epoch_acc = accuracy_score(labels_all, detections)
        outputs_all = outputs_all[:, 1]
        epoch_roc_auc_score = roc_auc_score(labels_all, outputs_all)


        print("""
                Train  ==>  Loss: {:.4f}   Accuracy: {:.2f} %   Recall: {:.2f} %
                         Precision: {:.2f} %   F1_score: {:.2f} %    ROC_AUC_score: {:.2f} %
              """.format(
                epoch_loss, epoch_acc * 100, epoch_recall * 100,
                epoch_precision * 100, epoch_f1 * 100, epoch_roc_auc_score * 100
            ))

        
        model.eval()
        with torch.no_grad():
            for dst in val_dst:
                dst_path = val_dir + dst
                val_dataloader = DataLoader(torch.load(dst_path), batch_size=bs, shuffle=False)

                for inputs, labels in train_dataloader:

                    labels_expanded = torch.zeros(len(inputs), 2, dtype=torch.float64)
                    for item in range(len(inputs)):
                        labels_expanded[item, labels[item].item()] = 1.00
                    labels_expanded = Variable(labels_expanded)

                    inputs = Variable(inputs.float(), requires_grad=True)
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    m = nn.Sigmoid()
                    outputs = m(outputs).double()

                    if not flag_v:
                        outputs_all = outputs
                        detections = preds
                        labels_all = labels
                        flag_v = True
                    else:
                        outputs_all = torch.cat((outputs_all, outputs), 0)
                        detections = torch.cat((detections, preds), 0)
                        labels_all = torch.cat((labels_all, labels), 0)
                    
                    loss = criterion(outputs, labels_expanded)
                    running_loss_v += loss.item() * inputs.size()[0]

        epoch_loss = running_loss_v / 6000
        epoch_f1 = f1_score(labels_all, detections)
        epoch_recall = recall_score(labels_all, detections)
        epoch_precision = precision_score(labels_all, detections)
        epoch_acc = accuracy_score(labels_all, detections)
        outputs_all = outputs_all[:, 1]
        epoch_roc_auc_score = roc_auc_score(labels_all, outputs_all)


        print("""
                Validation  ==>  Loss: {:.4f}   Accuracy: {:.2f} %   Recall: {:.2f} %
                         Precision: {:.2f} %   F1_score: {:.2f} %    ROC_AUC_score: {:.2f} %
              """.format(
                epoch_loss, epoch_acc * 100, epoch_recall * 100,
                epoch_precision * 100, epoch_f1 * 100, epoch_roc_auc_score * 100
            ))



        if epoch_roc_auc_score > best_auc:
            best_auc = epoch_roc_auc_score
            best_model_wts = copy.deepcopy(model.state_dict())
        
        if epoch_acc*100 >= 80:
            torch.save(
                model.state_dict(), '%s/weights/model_weights_%.2f_%d.pt' %\
                (path_data, epoch_roc_auc_score*100, epoch)
            )
        
        model.train()
        t2 = time.time()
        scheduler.step()
        print('Epoch running time:  {:.0f}m {:.0f}s'.format(
            (t2 - t1) // 60, (t2 - t1) % 60))
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best ROC_AUC: {:.2f} %'.format(best_auc * 100))
    model.load_state_dict(best_model_wts)

    return model

                












if __name__ == "__main__":
    path_project = str(((Path(__file__).parent).resolve().parent).resolve())
    path_data = path_project + '/seti-breakthrough-listen'
    model = train_val()
    torch.save(model.state_dict(), path_data+'/weights/model_best_weights.pt')
    # model = models.resnet18(pretrained=True)
    # print(model)

