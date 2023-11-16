import numpy as np
import torch
import sys
from torch import nn
from networks import quantized_net


# sys.path.append("..")

def int2hex(num, width=8):
    if num < 0:
        num += 16**(width)

    temp = hex(num)[2:]
    return temp.rjust(width, '0')

from utils.load_quan_dataset import dataloaders as quan_dataloaders

def print_fun(model_path):  
    # get a test image and save it to csv file
    imgs, labels = next(iter(quan_dataloaders['val']))
    img1 = imgs[0]
    # print(img1)
    img1 = torch.round(img1.permute((1, 2, 0))).int()
    hei, wid, _ = img1.shape
    with open('print/img1.txt', 'w') as file:
        for i in range(hei):
            for j in range(wid):
                for _ in range(4):
                    for k in range(3):
                        print(f'{int2hex(img1[i][j][2-k], 2)}', end='', file=file)
                    print('\n', end='', file=file)
            print('', end='', file=file)
    with open('print/img2see.csv', 'w') as file1:
        for i in range(3):
            for j in range(hei):
                for k in range(wid):
                    print(f'{img1[j][k][i]:>4}', end=',', file=file1)
                print('',end='\n',file=file1)
            print('',end='\n',file=file1)
    
    # load simulation model
    model = quantized_net.QuanNet()
    model.load_state_dict(torch.load(model_path))
    # model = torch.load(model_path)
    
    # definite a function as a hook to store the intermediate feature map
    def print_FeatureMap_hook(m, inputs, output):
        m.activate_output = output.clone().detach()

    # register hook    
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.register_forward_hook(print_FeatureMap_hook)
  
    # inference a img
    img = imgs[:1].to('cpu')
    # print(img)
    model.to('cpu')
    model(img)
    
    # use hook to get actitive output
    inter_FeaMap = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            inter_FeaMap.append(np.array(module.activate_output.to('cpu').clone().int().squeeze()))

    # print FeaMap to see 
    for i, tensor in enumerate(inter_FeaMap[:-1]):

        if i+1 == 1:  # featuremap 1
            channels, height, width = tensor.shape
            with open(f'print/feature/FeaMap{i+1}.txt', 'w') as file:
                for h in range(height):
                    for w in range(width):
                        for c in range(channels//2):
                            print(f'{int2hex(tensor[c*2+1, h, w], 2)}', end='', file=file)
                            print(f'{int2hex(tensor[c*2, h, w], 2)}', end='', file=file)
                            print('\n', end='', file=file)

        # # every 8 channels to print
        # if i+1 == 1 or i+1 == 2 or i+1 == 5 or i+1 == 6 or i+1 == 7:  # FeaMap1 2 5 6 7
        #     channels, height, width = tensor.shape
        #     with open(f'print/feature/FeaMap{i+1}.csv', 'w') as file:
        #         for c in range(channels//8):
        #             for h in range(height):
        #                 for w in range(width):
        #                     for j in range(8):
        #                         print(f'{tensor[c*8+j, h, w]:>4}', end='', file=file)
                            
        #                     print('', end=',', file=file)
        #                 print('\n', end='', file=file)
        #             print('\n', file=file)


        # # every 16 channels to print
        # if i+1 == 3:  # FeaMap3
        #     channels, height, width = tensor.shape
        #     with open(f'print/feature/FeaMap{i+1}.csv', 'w') as file:
        #         for c in range(channels//16):
        #             for h in range(height):
        #                 for w in range(width):
        #                     for j in range(16):
        #                         print(f'{tensor[c*16+j, h, w]:>4}', end='', file=file)
                            
        #                     print('', end=',', file=file)
        #                 print('\n', end='', file=file)
        #             print('\n', file=file)


        # # every 4 channels to print
        # if i+1 == 4 or i+1 == 9 or i+1 == 10 or i+1 == 11:  # FeaMap4 9 10 11
        #     channels, height, width = tensor.shape
        #     with open(f'print/feature/FeaMap{i+1}.csv', 'w') as file:
        #         for c in range(channels//4):
        #             for h in range(height):
        #                 for w in range(width):
        #                     for j in range(4):
        #                         print(f'{tensor[c*4+j, h, w]:>4}', end='', file=file)
                            
        #                     print('', end=',', file=file)
        #                 print('\n', end='', file=file)
        #             print('\n', file=file)


        # # every 2 channels to print
        # if i+1 == 8 or ( (i+1 >= 13) and (i+1 <= 23) ):  # FeaMap8 13-23
        #     channels, height, width = tensor.shape
        #     with open(f'print/feature/FeaMap{i+1}.csv', 'w') as file:
        #         for c in range(channels//2):
        #             for h in range(height):
        #                 for w in range(width):
        #                     for j in range(2):
        #                         print(f'{tensor[c*2+j, h, w]:>4}', end='', file=file)
                            
        #                     print('', end=',', file=file)
        #                 print('\n', end='', file=file)
        #             print('\n', file=file)


        # # every 1 channels to print
        # if i+1 == 12:  # FeaMap12
        #     channels, height, width = tensor.shape
        #     with open(f'print/feature/FeaMap{i+1}.csv', 'w') as file:
        #         for c in range(channels):
        #             for h in range(height):
        #                 for w in range(width):
        #                     print(f'{tensor[c, h, w]:>4}', end='', file=file)
                            
        #                     print('', end=',', file=file)
        #                 print('\n', end='', file=file)
        #             print('\n', file=file)

if __name__ == '__main__':
    print_fun('./models/simulation_model.pth')