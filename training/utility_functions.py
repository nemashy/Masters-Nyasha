
import torch
import torch.nn as nn
from pytorch_quantization import tensor_quant
from torch2trt.contrib.qat.layers.quant_conv import QuantConvBN2d,QuantConv2d,IQuantConv2d, IQuantConvBN2d
from torch2trt.contrib.qat.layers.quant_activation import QuantReLU, IQuantReLU
## QAT qrapper for ReLU layer: toggles between training and inference

class qrelu(torch.nn.Module):
    def __init__(self,inplace=False,qat=False,infer=False):
        super().__init__()
        if qat:
            if infer:
                self.relu = IQuantReLU(inplace)
            else:
                self.relu = QuantReLU(inplace)
        else:
            self.relu = nn.ReLU(inplace)

    def forward(self,input):
        return self.relu(input)


'''
Wrapper for conv2d + bn + relu layer. 
Toggles between QAT mode(on and off)
Toggles between QAT training and inference
In QAT mode:
    conv(quantized_weight) + BN + ReLU + quantized op. 
'''

class qconv2d(torch.nn.Module):
    """
    common layer for qat and non qat mode
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=3,
            stride: int=1,
            padding: int=0,
            groups: int=1,
            dilation: int=1,
            bias = None,
            padding_mode: str='zeros',
            eps: float=1e-5,
            momentum: float=0.1,
            freeze_bn = False,
            act: bool= True,
            norm: bool=True,
            qat: bool=False,
            infer: bool=False):
        super().__init__()
        if qat:
            if infer:
                if norm:
                    layer_list = [IQuantConvBN2d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        dilation=dilation,
                        bias=bias,
                        padding_mode=padding_mode)]

                else:
                    layer_list = [IQuantConv2d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        dilation=dilation,
                        bias=bias,
                        padding_mode=padding_mode)]
            else:
                if norm:
                    layer_list=[QuantConvBN2d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        dilation=dilation,
                        bias=bias,
                        padding_mode=padding_mode,
                        quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)]

                else:
                    layer_list = [QuantConv2d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        dilation=dilation,
                        bias=bias,
                        padding_mode=padding_mode,
                        quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)]
           
            if act:
                if infer:
                    layer_list.append(IQuantReLU())
                else:
                    layer_list.append(QuantReLU())
            
            self.qconv = nn.Sequential(*layer_list)
    
        else:
            layer_list=[
                    nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        bias=bias,
                        groups=groups)]
            if norm:
                layer_list.append(nn.BatchNorm2d(out_channels))
           
            if act:
                layer_list.append(nn.ReLU())
            
            self.qconv = nn.Sequential(*layer_list)

    def forward(self,inputs):
        return self.qconv(inputs)



