from time import time
import torch
import torchvision
import numpy as np
from d2l import torch as d2l
from networks.mobilenetv1 import MobileNetV1

from utils.load_dataset import dataloaders


def train(model, save_path, dataloaders=dataloaders, epochs=10, lr=0.001, device=None):
    """
    this fun is used to train model on custom dataset.
    model: the model to be trained
    save_path: path of the best training model to be saved in
    dataloaders: default is from utils.load_dataset, can be other
    epochs: the total epochs model will be trained
    lr: learning rate
    device: if it is None, it will be assigned automatically
    """

    # some hyperparameters
    T_0 = epochs // 3

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Now use device is {device}')
    model.to(device)

    # the weight of different classes
    a = [9.5, 7.0, 8.833333333333334, 10.666666666666666, 4.333333333333333, 1.0, 4.0, 12.5, 7.833333333333333, 7.666666666666667, 5.5, 18.166666666666668, 23.583333333333332, 5.166666666666667, 14.916666666666666, 19.583333333333332, 4.0, 14.333333333333334, 3.0, 10.0, 3.0, 15.833333333333334, 11.833333333333334, 16.666666666666668, 7.5, 18.333333333333332, 25.333333333333332, 18.416666666666668, 18.0, 7.833333333333333, 3.5, 9.833333333333334, 8.833333333333334, 5.666666666666667, 12.333333333333334, 9.666666666666666, 9.5, 11.833333333333334, 11.0, 9.333333333333334, 3.6666666666666665, 22.666666666666668, 4.666666666666667, 25.166666666666668, 11.833333333333334, 9.5, 8.833333333333334, 8.916666666666666, 13.5, 13.5, 7.833333333333333, 14.833333333333334, 9.166666666666666, 2.0, 1.0, 8.5, 9.166666666666666, 9.666666666666666, 7.5, 1.0, 13.833333333333334, 23.5, 6.583333333333333, 22.083333333333332, 6.0, 22.333333333333332, 2.0, 3.3333333333333335, 6.666666666666667, 6.833333333333333, 4.666666666666667, 9.416666666666666, 9.333333333333334, 3.0, 7.666666666666667, 3.0, 11.833333333333334, 8.0, 3.25, 14.583333333333334, 8.333333333333334, 20.666666666666668, 11.416666666666666, 10.333333333333334, 12.583333333333334, 23.583333333333332, 9.5, 2.9166666666666665, 8.833333333333334, 8.166666666666666, 13.583333333333334, 3.0, 14.666666666666666, 14.25, 15.583333333333334, 22.666666666666668, 16.0, 14.666666666666666, 7.333333333333333, 8.5, 2.6666666666666665, 3.0, 6.833333333333333, 20.25, 9.166666666666666, 11.833333333333334, 8.166666666666666, 7.5, 15.583333333333334, 10.0, 8.583333333333334, 6.0, 13.0, 23.333333333333332, 8.666666666666666, 3.0, 9.5, 7.916666666666667, 7.666666666666667, 14.0, 16.083333333333332, 8.833333333333334, 5.416666666666667, 10.583333333333334, 11.583333333333334]
    weight = torch.tensor(a, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer
    opt = torch.optim.SGD(  # params根据freeze修改
        lr=lr, params=model.parameters(), momentum=0.9, weight_decay=5e-4)
    opt_step = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=T_0)

    # the max accuracy in val dataset
    max_acc = 0
    

    for epoch in range(epochs):
        print(f'epoch: {epoch} *********************')
        t_s = time()

        # one complete epoch contains steps of train and val
        for type_id in ['train', 'val']:
            # metric[0] is total loss, metric[1] is total positive samples, metric[2] is number of all samples.
            metric = d2l.Accumulator(3)
            loader = dataloaders[type_id]

            for images, labels in loader:
                # set model to train or eval mode
                if type_id == 'train':
                    model.train()
                else:
                    model.eval()

                images = images.to(device)
                labels = labels.to(device).long()

                # clear the previous grad
                opt.zero_grad()

                # if mode == 'train', track grad
                with torch.set_grad_enabled(type_id == 'train'):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    _, predict = torch.max(outputs, 1)

                if type_id == 'train':
                    # compute grad and update grad
                    loss.backward()
                    opt.step()
                    opt_step.step()

                # compute accuracy and cumulative
                acc = torch.sum(predict == labels)
                num_batch = images.shape[0]
                metric.add(loss * num_batch, acc, num_batch)

            cur_acc = metric[1] / metric[2]
            cur_loss = metric[0] / metric[2]
            
            
            if type_id == 'val':
                # if cur_acc > max_acc, save current model to save_path
                if max_acc < cur_acc:
                    max_acc = cur_acc
                    torch.save(model, save_path)
                    # torch.save(model.state_dict(), save_path)


            print(
                f"{type_id}, loss is: {cur_loss:.4f}, acc is: {cur_acc:.4f} lr is: {opt_step.get_last_lr()[0]:.5f}")
        print(
            f'The time of this epoch including train and val is {time()-t_s:.3f}s')
        print('\n')
    print('######################')
    print('the best acc is :', max_acc)


if __name__ == '__main__':
    # model = MobileNetV1(n_class=125)
    model = torch.load('./models/best_pretrained_6718.pth', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = torch.load('./models/real_full_all_fcandothers.pth')
    # model = torch.load('./new_models/finetune1.pth')

    # model.load_state_dict(torch.load('./new_models/fc1.pth'), strict=False)   # 然后加载模型的state_dict
    

    # freeze the previous layers
    for param in model.parameters():
        param.requires_grad = True

    # replace old fc
    model.classifier = torch.nn.Linear(1024, 125, bias=False)

    train(model, './models/ywyp.pth', epochs=90, lr=0.001)


    # 注意优化器是否冻结
