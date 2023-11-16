import torch
import torch.nn as nn
import numpy as np
from test import test
from utils.load_quan_dataset import dataloaders as quan_dataloaders

from copy import copy, deepcopy
from networks import quantized_net




def int2hex(num, width=8):
    if num < 0:
        num += 16**(width)

    temp = hex(num)[2:]
    return temp.rjust(width, '0')

def print_ParamsAndFeaMap(path):
    # load quantized parameters
    state = torch.load(path)
    # print(state)

    # get weight and scale from state
    weight_list = []
    weight_scale = []
    act_scale = []
    for key, value in state.items():
        if 'weight' in key:
            weight_list.append(value.int_repr().float())
            weight_scale.append(value.q_scale())
        elif 'scale' in key:
            # print(key)
            act_scale.append(float(value))
    # fc's weight and scale
    weight_list.append(state['classifier._packed_params._packed_params'][0].int_repr().float())
    weight_scale.append(state['classifier._packed_params._packed_params'][0].q_scale())
    weight_scale = torch.tensor((weight_scale))

    # print(act_scale)
    act_scale = torch.tensor(act_scale)
    act_scale[0] = 1.0

    # compute bias scale
    bias_scale = act_scale[:-1] * weight_scale

    # compute S = s1 * s2 / s3
    S = (act_scale[:-1] * weight_scale) / act_scale[1:]

    # compute constant and divisor
    for count in range(20):
        constant = torch.round(S * 2**count).int()
        temp2 = constant / 2**count
        temp3 = torch.abs((temp2 - S) / S)
        if torch.max(temp3) <= 0.01:
            print(count)
            print(constant)
            break

    # get bias from state
    bias_list = []
    index = 0
    for key, value in state.items():
        if 'bias' in key:
            new_bias = deepcopy(value)
            new_bias.data = torch.round(new_bias.detach() / bias_scale[index])
            bias_list.append(new_bias)
            index += 1
    new_bias = deepcopy(state['classifier._packed_params._packed_params'][1])
    new_bias.data = torch.round(new_bias.detach() / bias_scale[13])
    bias_list.append(new_bias)

    # create a quantized model
    model = quantized_net.QuanNet()

    # set quantized parameters to model
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data = weight_list[count].data
            module.bias.data = bias_list[count].data
            module.constant = constant[count].clone().detach()
            module.divisor = torch.tensor(2**14)
            count += 1

    # classifier's bias can be zeros
    # model.classifier.bias.data = torch.zeros([1000])

    # test simulated quantization model
    # test(model, dataloaders=quan_dataloaders, batches=1000)


    count = 0

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            count += 1
            out_c, in_c, hei, wid = module.weight.shape
            # 
            weight = torch.round(module.weight.detach().to('cpu')).int()
            bias = torch.round(module.bias.detach().to('cpu')).int()
            constant = torch.round(module.constant).int().squeeze()
            divisor = round(np.log2(int(torch.round(module.divisor.data))))

            if count == 1:  # CONV1
                print("CONV1打印")
                # print weight
                with open(f'print/parameters/conv1_weight.txt', 'w') as file:
                    print(f"channel:3  width:512  depth:16", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 2

                    # T
                    wh_cnt = 0
                    i = 0
                    wh = [[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,1,2],[0,2,0],[0,2,1],[0,2,2],
                          [1,0,0],[1,0,1],[1,0,2],[1,1,0],[1,1,1],[1,1,2],[1,2,0],[1,2,1],[1,2,2],
                          [2,0,0],[2,0,1],[2,0,2],[2,1,0],[2,1,1],[2,1,2],[2,2,0],[2,2,1],[2,2,2]]
                    for hang in range(16):
                        if hang <= 13:
                            for o in range(out_c):
                                if wh_cnt < 26:
                                    print(f'{int2hex(weight[31-o, wh[wh_cnt+1][0], wh[wh_cnt+1][1], wh[wh_cnt+1][2]], 2)}', end='', file=file)
                                    print(f'{int2hex(weight[31-o, wh[wh_cnt][0], wh[wh_cnt][1], wh[wh_cnt][2]], 2)}', end='', file=file)
                                if wh_cnt == 26:
                                    print('00',end='',file=file)
                                    print(f'{int2hex(weight[31-o, wh[wh_cnt][0], wh[wh_cnt][1], wh[wh_cnt][2]], 2)}', end='', file=file)

                        else:
                            for o in range(out_c):
                                print('0000',end='',file=file)
                        print(',', end='\n', file=file)
                        wh_cnt = wh_cnt + 2

                # print bias
                with open(f'print/parameters/conv{count}_bias.txt', 'w') as file:
                    print(f"out=32, width=512, depth=16", file=file)
                    print(f"This layer's constant is {constant}", file=file)


                    # noT
                    for o in range(out_c//2):
                        # print('*---------------------------------------*', file=file)
                        print(f'{int2hex(bias[o*2+1], 4)}', end='', file=file)
                        print(f'{int2hex(bias[o*2], 4)}', end='', file=file)

                        print(',', end='\n', file=file)

            if count == 2:  # DWCL2
                print("DWCL2打印")
                # print weight
                with open(f'print/parameters/conv{count//2 + 1}_dw_weight.txt', 'w') as file:
                    print(f"channel:32  width:144  depth:16", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 2

                    # T
                    for o in range(out_c//2):
                        for i in range(in_c):
                            for h in range(hei):
                                for w in range(wid):
                                    print(f'{int2hex(weight[2*o+1, i, h, w], 2)}', end='', file=file)
                                    print(f'{int2hex(weight[2*o, i, h, w], 2)}', end='\n', file=file)

            if count == 4:  # DWCL3
                print("DWCL3打印")
                # print weight
                with open(f'print/parameters/conv{count//2 + 1}_dw_weight.txt', 'w') as file:
                    print(f"channel:64  width:72  depth:64", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 1

                    # T
                    for o in range(out_c):
                        for i in range(in_c):
                            for h in range(hei):
                                for w in range(wid):
                                    print(f'{int2hex(weight[o, i, h, w], 2)}', end='', file=file)
                        print(',', end='\n', file=file)

            if count == 6:  # DWCL4
                print("DWCL4打印")
                # print weight
                with open(f'print/parameters/conv{count//2 + 1}_dw_weight.txt', 'w') as file:
                    print(f"channel:128  width:144  depth:64", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 2

                    # T
                    for o in range(out_c//2):
                        for i in range(in_c):
                            for h in range(hei):
                                for w in range(wid):
                                    print(f'{int2hex(weight[2*o+1, i, h, w], 2)}', end='', file=file)
                                    print(f'{int2hex(weight[2*o, i, h, w], 2)}', end='\n', file=file)

            if count == 8:  # DWCL5
                print("DWCL5打印")
                # print weight
                with open(f'print/parameters/conv{count//2 + 1}_dw_weight.txt', 'w') as file:
                    print(f"channel:128  width:40  depth:256", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 1/2

                    # T
                    for o in range(out_c):
                        for i in range(in_c):
                            for h in range(hei):
                                for w in range(wid):
                                    print(f'{int2hex(weight[o, i, h, w], 2)}', end='', file=file)
                                    if h == 1 and w == 1: 
                                        print(',', end='\n', file=file)
                        print('00', end='', file=file)
                        print(',', end='\n', file=file)

            if count == 10:  # DWCL6
                print("DWCL6打印")
                # print weight
                with open(f'print/parameters/conv{count//2 + 1}_dw_weight.txt', 'w') as file:
                    print(f"channel:256  width:72  depth:256", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 1

                    # T
                    for o in range(out_c):
                        for i in range(in_c):
                            for h in range(hei):
                                for w in range(wid):
                                    print(f'{int2hex(weight[o, i, h, w], 2)}', end='', file=file)
                        print(',', end='\n', file=file)


            if count == 12:  # DWCL7
                print("DWCL7打印")
                # print weight
                with open(f'print/parameters/conv{count//2 + 1}_dw_weight.txt', 'w') as file:
                    print(f"channel:256  width:24  depth:1024", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 1/3

                    # T
                    for o in range(out_c):
                        for i in range(in_c):
                            for h in range(hei):
                                for w in range(wid):
                                    print(f'{int2hex(weight[o, i, h, w], 2)}', end='', file=file)
                                    if w == 2: 
                                        print(',', end='\n', file=file)
                            print('000000', end='', file=file)
                        print(',', end='\n', file=file)

            if count == 14 or count == 16 or count == 18 or count == 20 or count == 22:  # DWCL8-x
                print("DWCL8-x打印")
                # print weight
                with open(f'print/parameters/conv8_{(count-12)//2}_dw_weight.txt', 'w') as file:
                    print(f"channel:512  width:40  depth:1024", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 1/2

                    # T
                    for o in range(out_c):
                        for i in range(in_c):
                            for h in range(hei):
                                for w in range(wid):
                                    print(f'{int2hex(weight[o, i, h, w], 2)}', end='', file=file)
                                    if h == 1 and w == 1: 
                                        print(',', end='\n', file=file)
                        print('00', end='', file=file)
                        print(',', end='\n', file=file)

            if count == 3:  # PWCL2
                print("PWCL2打印")
                # print weight
                with open(f'print/parameters/conv{count//2 + 1}_pw_weight.txt', 'w') as file:
                    print(f"out=64, in=32, width=256, depth=64, PS_IN=8, PS_OUT=2", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 2
                    # T
                    for g in range(group):
                        out_group = [[0, out_c//2], [out_c//2, out_c]]
                        for i in range(in_c):
                            for o in range(out_group[g][0], out_group[g][1]):
                                for h in range(hei):
                                    for w in range(wid):
                                        print(f'{int2hex(weight[o, i, h, w], 2)}', end='', file=file)                            
                            print(',', end='\n', file=file)


            if count == 5:  # PWCL3
                print("PWCL3打印")
                # print weight
                with open(f'print/parameters/conv{count//2 + 1}_pw_weight.txt', 'w') as file:
                    print(f"out=128, in=64, width=256, depth=256, PS_IN=16, PS_OUT=4", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 4
                    # T
                    for g in range(group):
                        out_group = [[0, out_c//4], [out_c//4, out_c//2], [out_c//2, out_c//4*3], [out_c//4*3, out_c]]
                        for i in range(in_c):
                            for o in range(out_group[g][0], out_group[g][1]):
                                for h in range(hei):
                                    for w in range(wid):
                                        print(f'{int2hex(weight[o, i, h, w], 2)}', end='', file=file)                            
                            print(',', end='\n', file=file)

            if count == 7:  # PWCL4
                print("PWCL4打印")
                # print weight
                with open(f'print/parameters/conv{count//2 + 1}_pw_weight.txt', 'w') as file:
                    print(f"out=128, in=128, width=512, depth=256, PS_IN=32, PS_OUT=2", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 2
                    # T
                    for g in range(group):
                        out_group = [[0, out_c//2], [out_c//2, out_c]]
                        for i in range(in_c):
                            for o in range(out_group[g][0], out_group[g][1]):
                                for h in range(hei):
                                    for w in range(wid):
                                        print(f'{int2hex(weight[o, i, h, w], 2)}', end='', file=file)                            
                            print(',', end='\n', file=file)

            if count == 9:  # PWCL5
                print("PWCL5打印")
                # print weight
                with open(f'print/parameters/conv{count//2 + 1}_pw_weight.txt', 'w') as file:
                    print(f"out=256, in=128, width=256, depth=1024, PS_IN=32, PS_OUT=8", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 8
                    # T
                    for g in range(group):
                        # print(out_c)
                        out_group = [[0, out_c//8], [out_c//8, out_c//4], [out_c//4, out_c//8*3], [out_c//8*3, out_c//2],
                                     [out_c//2, out_c//8*5], [out_c//8*5, out_c//4*3], [out_c//4*3, out_c//8*7], [out_c//8*7, out_c]]
                        for i in range(in_c):
                            for o in range(out_group[g][0], out_group[g][1]):
                                for h in range(hei):
                                    for w in range(wid):
                                        print(f'{int2hex(weight[o, i, h, w], 2)}', end='', file=file)                            
                            print(',', end='\n', file=file)

            if count == 11:  # PWCL6
                print("PWCL6打印")
                # print weight
                with open(f'print/parameters/conv{count//2 + 1}_pw_weight.txt', 'w') as file:
                    print(f"out=256, in=256, width=512, depth=1024, PS_IN=64, PS_OUT=4", file=file)
                    print(f"This layer's constant is {constant}", file=file)

                    group = 4
                    # T
                    for g in range(group):
                        out_group = [[0, out_c//4], [out_c//4, out_c//2], [out_c//2, out_c//4*3], [out_c//4*3, out_c]]
                        for i in range(in_c):
                            for o in range(out_group[g][0], out_group[g][1]):
                                for h in range(hei):
                                    for w in range(wid):
                                        print(f'{int2hex(weight[o, i, h, w], 2)}', end='', file=file)                            
                            print(',', end='\n', file=file)

                # # print bias
                # with open(f'lzprint/parameters/conv{count}_bias.txt', 'w') as file:
                #     print(f"group=4, out=8, datawidth=4,  8*4=32", file=file)
                #     print(f"This layer's constant is {constant}", file=file)
                #     # T
                #     for o in range(out_c):
                #         # print('*---------------------------------------*', file=file)
                #         print(f'{int2hex(bias[16* (o//8) + 7 - o], 4)}', end='', file=file)
                        
                #         if (o+1)%8 == 0:
                #             print(',', end='\n', file=file)
            
            
        
    # save simulation model
    # torch.save(model.state_dict(), './models/simulation_model.pth')


    
    
if __name__ == '__main__':
    print_ParamsAndFeaMap('./models/best_qat_state_98828.pth')



