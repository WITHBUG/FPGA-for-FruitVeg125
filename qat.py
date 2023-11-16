import torch
import torch.nn as nn

# from networks.fusion_net import FusionNet
from networks.network_qat import MobileNetV1
from test import test

from torch.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize as FMAOFQ
from torch.quantization.observer import MovingAverageMinMaxObserver as MAMM
from utils.load_quan_dataset import dataloaders as quan_dataloaders
import train
import qat_train


def modify_first_conv_and_finetune(model):
    # modify the first conv's parameters
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').reshape(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').reshape(1, -1, 1, 1)
    first_conv = model.conv1[0]
    first_conv.bias =  nn.Parameter(-(first_conv.weight.detach() * mean / std).sum(dim=(1, 2, 3)))
    first_conv.weight.data = first_conv.weight.detach() / (std * 255.0)
    
    # finetune
    train.train(model, save_path='./models/best_FuseFirstConv.pth', dataloaders=quan_dataloaders,
                epochs=3, lr=0.0001, device='cuda')
    # return torch.load('best_FuseFirstConv.pth')
    return model


def main():
    # load model
    # Model = torch.load('best_micro_pruned.pth')
    # model = FusionNet()
    # model.load_state_dict(Model.state_dict())
    # print('no change the first conv')
    # test(model)

    # fuse first conv
    # Model = torch.load('./models/ywyp_99303.pth')
    # model = MobileNetV1(n_class=125)
    # model.load_state_dict(Model.state_dict(), strict=False)
    
    # model.to('cuda')
    # model = modify_first_conv_and_finetune(model)
    # test(model, dataloaders=quan_dataloaders)
    
    # qat config
    model = torch.load('./models/FuseFirstConv_99208.pth')

    # 
    # model.qconfig = torch.quantization.qconfig.QConfig(
    # activation=FMAOFQ.with_args(observer=MAMM, quant_min=0, quant_max=255, dtype=torch.quint8),
    # weight=FMAOFQ.with_args(observer=MAMM, averaging_constant=0.1, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    # )


    # model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # model.qconfig = torch.quantization.default_qconfig
    # model.qconfig = torch.quantization.qconfig.default_qat_qconfig_v2
    # model.qconfig = myqconfig
    # model.qconfig = torch.quantization.qconfig.default_qat_qconfig_v2
    

    from torch.ao.quantization.observer import MovingAverageMinMaxObserver, HistogramObserver, FixedQParamsObserver
    
    from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
    default_fused_act_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                       quant_min=0,
                                                                       quant_max=255,
                                                                       dtype=torch.quint8,)
    default_fused_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                      quant_min=-128,
                                                                      quant_max=127,
                                                                      dtype=torch.qint8,
                                                                      qscheme=torch.per_tensor_symmetric)
    from torch.ao.quantization.qconfig import QConfig
    model.qconfig = QConfig(activation=default_fused_act_fake_quant, weight=default_fused_wt_fake_quant)


    # fuse model
    model.eval()
    model.fuse_model()
    opt = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay=5e-4)
    
    # prepare
    model.train()
    model_prepared = torch.quantization.prepare_qat(model)
    
    # quantization aware training
    model_prepared.to('cuda')
    
    print('begin to qat')
    qat_train.train(model_prepared, './models/best_qat_state.pth', opt, 10, device='cpu',
                    dataloaders=quan_dataloaders)
    
    # # test quantized model
    # quantized_model = torch.quantization.convert(model_prepared.eval(), inplace=False)
    # quantized_model.load_state_dict(torch.load('8bits.pth'))
    # quantized_model.eval()
    
    # print('quantized model acc is: ')
    # test(quantized_model, 'cpu', dataloaders=quan_dataloaders)
    
if __name__ == "__main__":
    main()

    

    
    