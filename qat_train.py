from time import time
import torch
from d2l import torch as d2l

# from utils.load_dataset import dataloaders
from utils.load_quan_dataset import dataloaders
from test import test


def train_one_epoch(model, criterion, optimizer, data_loader, device, batchsize):
    model.train()
    model.to(device)

    # metric[0] is total loss, metric[1] is total positive samples, metric[2] is number of all samples.
    metric = d2l.Accumulator(3)

    cnt = 0
    for images, labels in data_loader:
        print('.', end='')
        cnt += 1
        optimizer.zero_grad()

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, predict = torch.max(outputs, 1)
        acc = torch.sum(predict == labels)
        num_batch = images.shape[0]
        metric.add(loss * num_batch, acc, num_batch)

        if cnt >= batchsize:
            break

    print(
        f'Loss is: {metric[0] / metric[2]:.3f}, Acc@1: {metric[1] / metric[2]:.3f}')
    return


def train(model, save_path, opt, epochs=8, device='cuda', dataloaders=dataloaders, batchsize=200):
    """
    model: the model need to be trained in QAT mode
    save_path: path the best qat tarining model be saved to
    opt: the optimizer
    epochs: number of training epoch
    """
    max_acc = 0.0

    # some hyperparameters setting
    loss_fn = torch.nn.CrossEntropyLoss()

    for nepoch in range(epochs):
        train_one_epoch(model, loss_fn, opt, dataloaders['train'], device, batchsize)
        if nepoch > 3:
            # Freeze quantizer parameters
            model.apply(torch.ao.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        test(model, dataloaders=dataloaders)

        # Check the accuracy after each epoch
        model.to('cpu')
        

        quantized_model = torch.ao.quantization.convert(
            model.eval(), inplace=False)
        quantized_model.eval()
        print('start val')

        
        val_acc = test(quantized_model, 'cpu', dataloaders=dataloaders)
        if max_acc < val_acc:
            max_acc = val_acc
            # torch.save(quantized_model, save_path)
            # must save state_dict()
            torch.save(quantized_model.state_dict(), save_path)

    print('the max acc is:', max_acc)


if __name__ == '__main__':
    qat_state = torch.load('./8bits_dict_98733.pth')
    val_acc = test(qat_state, 'cpu', dataloaders=dataloaders)