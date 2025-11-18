import torch.nn as nn 
from cyclegan.PatchGANDiscriminator import PatchGANDiscriminator


class MultiScaleDiscriminator_Sentinel(nn.Module) : 
    def __init__(self, ):
        super().__init__()
        self.discriminator_scale1 = PatchGANDiscriminator()
        self.discriminator_scale2 = PatchGANDiscriminator()

        self.averagepool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)



    def forward(self,x):
        out1 = self.discriminator_scale1(x)
        out2 = self.averagepool(x)
        out2 = self.discriminator_scale2(out2)

        return out1 , out2
    
