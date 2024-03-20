import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# import resnet as models
import numpy as np
import cv2
import os
import random
from torchsummary import summary
import torch.optim as optim

dataload_num_workers = 1
input_shape = (3, 224, 224)


def load_data(path, class_id, shape):
    data = []
    exts = [".jpg", ".jpeg", ".png"]
    files = os.listdir(path)
    for name in files:
        if not os.path.splitext(name.lower())[1] in exts:
            continue
        img = cv2.imread(os.path.join(path, name))
        if type(img) == type(None):
            print("read file {} fail".format(os.path.join(path, name)))
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (shape[2], shape[1]))
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) # hwc to chw layout
        img = (img - 127.5) * 0.0078125
        data.append((torch.from_numpy(img), class_id))
    return data
    
class Dataset:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)

def train_image_classification(path_to_save, data_dir,
        cuda=True, 
        learning_rate=1e-4, 
        batch_size=32, 
        start_epoch=0, 
        epoch=100,
        train_split=80, 
        model_type='resnet18', 
        model_weight=None,
        validate_matrix='val_acc',
        save_method='best',
        step_lr=(150, 200),
        weight_decay=5e-4,
        warm_up_epoch=6
    ):

    os.makedirs(path_to_save, exist_ok=True)
    
    # cuda
    if cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("use cpu")

    # dataset and evaluator
    print("----------------------------------------------------------")
    print('Loading the dataset...')
    
    
    labels = os.listdir(data_dir)
    num_classes = len(labels)

    print('Project label:', labels)
    print('The number of classes:', num_classes)

    trainset = []
    valset = []

    for i, label in enumerate(labels):
        path = os.path.join(data_dir, label)
        data = load_data(path, i, input_shape)
        random.shuffle(data)
        val_len = int(train_split / 100 * len(data))
        trainset.extend(data[val_len:])
        valset.extend(data[:val_len])

    random.shuffle(trainset)
    random.shuffle(valset)

    trainset = Dataset(trainset)
    valset = Dataset(valset)
    print("trainset len:{}, valset len:{}".format(len(trainset), len(valset)))


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False)

    # model
    print("----------------------------------------------------------")
    print('Building the model...')

    if model_type == 'resnet18':
        net = models.resnet18(pretrained=False, num_classes=num_classes)
    elif model_type == 'resnet34':
        net = models.resnet34(pretrained=False, num_classes=num_classes)
    elif model_type == 'resnet50':
        net = models.resnet50(pretrained=False, num_classes=num_classes)
    elif model_type == 'resnet101':
        net = models.resnet101(pretrained=False, num_classes=num_classes)
    elif model_type == 'resnet152':
        net = models.resnet152(pretrained=False, num_classes=num_classes)
    else:
        print('model type error')
        return

    if model_weight:
        net.load_state_dict(torch.load(model_weight))

    net.to(device)
    summary(net, input_size=input_shape)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_lr, gamma=0.1)

    # train
    print("----------------------------------------------------------")
    print('Start training...')

    best_acc = 0.0
    best_epoch = 0
    for epoch in range(start_epoch, epoch):
        if epoch < warm_up_epoch:
            lr = learning_rate * (epoch + 1) / warm_up_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('epoch:', epoch, 'lr:', lr)
        else:
            #scheduler.step()
            print('epoch:', epoch, 'lr:', optimizer.param_groups[0]['lr'])
        
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        if (epoch + 1) % 5 == 0:
            net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in valloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            print('Accuracy of the network on the val images: %d %%' % acc)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                print('save model params to', os.path.join(path_to_save, 'classifier_best.pth'))
                torch.save(net.state_dict(), os.path.join(path_to_save, 'classifier_best.pth'))

        if (epoch + 1) % 10 == 0:
            print('save model params to', os.path.join(path_to_save, 'classifier_{}.pth'.format(epoch)))
            torch.save(net.state_dict(), os.path.join(path_to_save, 'classifier_{}.pth'.format(epoch)))

    print('Finished Training')
    torch.save(net.state_dict(), os.path.join(path_to_save, 'classifier_final.pth'))

train_image_classification("out", "data/dog_cat",
        cuda=False, 
        learning_rate=1e-4, 
        batch_size=32, 
        start_epoch=0, 
        epoch=100,
        train_split=80, 
        model_type='resnet18', 
        model_weight=None,
        validate_matrix='val_acc',
        save_method='best',
        step_lr=(150, 200),
        weight_decay=5e-4,
        warm_up_epoch=6
    )