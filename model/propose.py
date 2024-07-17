import torch
import torch.nn as nn
import torch.nn.functional as F
    
class CA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out) 
        
class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Feature_Pool, self).__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Linear(dim, dim // ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim // ratio, dim)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2).view(b,c)
        return y

class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention, self).__init__()
        self.gap_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim//ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim//ratio, dim)
    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2)
        return max_out

class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1,bias=True)
    def forward(self, x):
        x1 = self.conv1(x)
        return x1
    
class EAEF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_pool_rgb = Feature_Pool(dim)
        self.mlp_pool_t = Feature_Pool(dim)
        self.dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=7,padding=3,groups=dim)
        self.cse = Channel_Attention(dim*2)
        self.sse_r = Spatial_Attention(dim)
        self.sse_t = Spatial_Attention(dim)
    def forward(self, RGB, T):
        ############################################################################
        b, c, h, w = RGB.size()
        rgb_y = self.mlp_pool_rgb(RGB)
        t_y = self.mlp_pool_t(T)
        rgb_y = rgb_y / rgb_y.norm(dim=1, keepdim=True)
        t_y = t_y / t_y.norm(dim=1, keepdim=True)
        rgb_y = rgb_y.view(b, c, 1)
        t_y = t_y.view(b, 1, c)
        logits_per = c * rgb_y @ t_y
        cross_gate = torch.diagonal(torch.sigmoid(logits_per)).reshape(b, c, 1, 1)
        add_gate = torch.ones(cross_gate.shape).cuda() - cross_gate
        ##########################################################################
        New_RGB_A = RGB * cross_gate
        New_T_A = T * cross_gate
        x_cat = torch.cat((New_RGB_A,New_T_A),dim=1)
        ##########################################################################
        fuse_gate = torch.sigmoid(self.cse(self.dwconv(x_cat)))
        rgb_gate, t_gate = fuse_gate[:, 0:c, :], fuse_gate[:, c:c * 2, :]
        ##########################################################################
        New_RGB = RGB * add_gate + New_RGB_A * rgb_gate
        New_T = T * add_gate + New_T_A * t_gate
        ##########################################################################
        New_fuse_RGB = self.sse_r(New_RGB)
        New_fuse_T = self.sse_t(New_T)
        attention_vector = torch.cat([New_fuse_RGB, New_fuse_T], dim=1)
        attention_vector = torch.softmax(attention_vector,dim=1)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        New_RGB = New_RGB * attention_vector_l + New_T * attention_vector_r
        New_T = New_T * attention_vector_r
        ##########################################################################
        return New_RGB, New_T


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
    
class propose(nn.Module):
    def __init__(self, num_classes):
        super(propose, self).__init__()
        self.rgb_conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.rgb_conv2 = BasicConv2d(64, 256, kernel_size=3, stride=2, padding=1)
        self.rgb_conv3 = BasicConv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.x_conv1 = BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.x_conv2 = BasicConv2d(64, 256, kernel_size=3, stride=2, padding=1)
        self.x_conv3 = BasicConv2d(256, 512, kernel_size=3, stride=2, padding=1)

        # 디코더
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 64, kernel_size=2, stride=2)  # 256 + 256 -> 512
        self.up3 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2)  # 64 + 64 -> 128
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)  # 32 + 32 -> 64

        # 추가적인 컨볼루션 레이어
        self.additional_conv_1 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.additional_conv_2 = BasicConv2d(256, 256, kernel_size=3, padding=1)
        self.additional_conv_3 = BasicConv2d(512, 512, kernel_size=3, padding=1)

        self.additional_conv_1x = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.additional_conv_2x = BasicConv2d(256, 256, kernel_size=3, padding=1)
        self.additional_conv_3x = BasicConv2d(512, 512, kernel_size=3, padding=1)

      
        self.idam = EAEF(512)
        self.ca1_rgb = CA(64)
        self.ca1_x = CA(64)
        self.ca2_rgb = CA(256)
        self.ca2_x = CA(256)

        self.sa = SA()

    def forward(self, input):
        # 인코더
        rgb = input[:, :3]
        x = input[:, 3:]

        rgb1 = self.rgb_conv1(rgb)
        rgb1 = self.additional_conv_1(rgb1)
        rgb1 = self.ca1_rgb(rgb1)
        rgb1 = self.sa(rgb1)
        
        x1 = self.x_conv1(x)
        x1 = self.additional_conv_1x(x1)
        x1 = self.ca1_x(x1)
        x1 = self.sa(x1)

        rgb2 = self.rgb_conv2(rgb1)
        rgb2 = self.additional_conv_2(rgb2)
        rgb2 = self.ca2_rgb(rgb2)
        rgb2 = self.sa(rgb2)
        x2 = self.x_conv2(x1)
        x2 = self.additional_conv_2x(x2)
        x2 = self.ca2_x(x2)
        x2 = self.sa(x2)


        rgb3 = self.rgb_conv3(rgb2)
        rgb3 = self.additional_conv_3(rgb3)


        x3 = self.x_conv3(x2)
        x3 = self.additional_conv_3x(x3)

        rgb3, x3 = self.idam(rgb3, x3)

        # 디코더
        x = self.up1(rgb3)
        x = torch.cat((x, rgb2), dim=1)
        x = self.up2(x)
        x = torch.cat((x, rgb1), dim=1)
        x = self.up3(x)
        x = self.final_conv(x)
        
        return x

   
def unit_test():
    rgb = torch.randn(1, 3, 480, 640).cuda(0)
    thermal = torch.randn(1, 1, 480, 640).cuda(0)
    net = propose(2).cuda(0)
    input = torch.cat((rgb, thermal), dim=1)
    net(input)
    print(net)
    # from torchsummary import summary
    # summary(net, (4, 480,640))
    # Calculate the number of parameters
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total number of parameters: {num_params}')
if __name__ == "__main__":
    unit_test()
    