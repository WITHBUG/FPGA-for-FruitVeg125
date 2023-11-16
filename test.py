import torch
from utils.load_dataset import dataloaders
from time import time
from d2l import torch as d2l

def test(model, device=None, is_eval=True, batches=50, dataloaders=dataloaders):
    """
    model: the model to be test
    device: cpu or cuda, if it is None, it will be decided automatically
    is_eval: Bool, if True, use val dataset, else use train dataset.
    batches: the number of batches the model will run on the train dataset.
    dataloader: default dataloader is from utils.load_dataset, can be replace with utils.load_quan_dataset
    """
    start_time = time()
    if is_eval:
        data_loader = dataloaders['val']
    else:
        data_loader = dataloaders['train']
        
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        
    # the weight of different class 
    loss_fn = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.eval()

    # metric[0] is total loss, metric[1] is total positive samples, metric[2] is number of all samples.
    metric = d2l.Accumulator(3)
    count = 0
    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device).long()

        y = model(X)
        loss = loss_fn(y, Y)
        
        _, predict = torch.max(y, 1)
        acc = torch.sum(predict == Y)
        num_batch = X.shape[0]
        metric.add(loss * num_batch, acc, num_batch)
        
        count += 1
        if is_eval == True and count >= batches:
            break
    
    end_time = time()
    top1 = metric[1] / metric[2]
    print(f'Loss is {metric[0] / metric[2]:.4f}, acc is {top1:.4f}')
    print(f'the time is {end_time - start_time:.3f}s')
    
    return (top1)


if __name__ == '__main__':
    # model = MobileNetV1()
    model = torch.load('./models/nwyp_99430.pth', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = torch.load('./models/best_pretrained.pth')
    

    # freeze the previous layers
    # for param in model.parameters():
    #     param.requires_grad = True

    # replace old fc
    # model.classifier = torch.nn.Linear(1024, 10, bias=True)

    test(model)
