import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import numbers
from einops import rearrange
# import einops

## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type = 'WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        # self.relu = nn.LeakyReLU(0.2, True) if relu else None
        # self.relu = nn.ReLU(inplace=True) if relu else None
        self.relu = nn.GELU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


##########################################################################

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            BasicConv(channel, channel // 16, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True),

            nn.GELU(),
            # nn.LeakyReLU(0.2, True),
            BasicConv(channel // reduction, channel, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class CA_attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_attention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool1 = nn.AdaptiveAvgPool2d(4)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            Dynamic_conv2d(channel, channel //16, 1, stride=1, padding=(1 - 1) // 2),
            # BasicConv(channel, channel // 16, 1, stride=1, padding=(1 - 1) // 2, relu=False),
            nn.GELU(),
            # nn.LeakyReLU(0.2, True),
            #BasicConv(channel // 16, channel, 1, stride=1, padding=(1 - 1) // 2, relu=False),
            Dynamic_conv2d(channel // 16, channel, 1, stride=1, padding=(1 - 1) // 2),

        )
        self.sig = nn.Sigmoid()


    def forward(self, x):
        y = self.avg_pool1(x)
        y = self.avg_pool2(self.conv_du(y))
        return x * self.sig(y)
##########################################################################


class ChannelPool(nn.Module):
    def forward(self, x):
        # return (torch.mean(x, 1).unsqueeze(1))

        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)



class pixelConv(nn.Module):
    # Generate pixel kernel  (3*k*k)xHxW
    def __init__(self, in_feats, out_feats, rate=1, ksize=3):
        super(pixelConv, self).__init__()
        self.padding = (ksize - 1) // 2
        self.ksize = ksize
        self.out_feats = out_feats
        self.zero_padding = nn.ZeroPad2d(self.padding)
        mid_feats = in_feats * rate ** 2
        self.kernel_conv = nn.Sequential(*[
            # nn.Conv2d(in_feats, mid_feats, kernel_size=3, padding=1),
            # nn.Conv2d(mid_feats, mid_feats, kernel_size=3, padding=1),
            nn.Conv2d(in_feats, self.out_feats *ksize ** 2, kernel_size=3, padding=1)
        ])

    def forward(self, x_feature, x):
        kernel_set = self.kernel_conv(x_feature)

        dtype = kernel_set.data.type()
        ks = self.ksize
        N = self.ksize ** 2  # patch size
        # padding the input image with zero values
        if self.padding:
            x = self.zero_padding(x)

        p = self._get_index(kernel_set, dtype)
        p = p.contiguous().permute(0, 2, 3, 1).long()
        x_pixel_set = self._get_x_q(x, p, N)
        b, c, h, w = kernel_set.size()
        kernel_set_reshape = kernel_set.reshape(-1, self.ksize ** 2, self.out_feats, h, w).permute(0, 2, 3, 4, 1)
        x_ = x_pixel_set

        out = x_ * kernel_set_reshape
        out = out.sum(dim=-1, keepdim=True).squeeze(dim=-1)
        out = out
        return out

    def _get_index(self, kernel_set, dtype):
        '''
        get absolute index of each pixel in image
        '''
        N, b, h, w = self.ksize ** 2, kernel_set.size(0), kernel_set.size(2), kernel_set.size(3)
        # get absolute index of center index
        p_0_x, p_0_y = np.meshgrid(range(self.padding, h + self.padding), range(self.padding, w + self.padding),
                                   indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        # get relative index around center pixel
        p_n_x, p_n_y = np.meshgrid(range(-(self.ksize - 1) // 2, (self.ksize - 1) // 2 + 1),
                                   range(-(self.ksize - 1) // 2, (self.ksize - 1) // 2 + 1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2 * N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)
        p = p_0 + p_n
        p = p.repeat(b, 1, 1, 1)
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()  # dimension of q: (b,h,w,2N)
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*padded_w)
        x = x.contiguous().view(b, c, -1)
        # (b, h, w, N)
        # index_x*w + index_y
        index = q[..., :N] * padded_w + q[..., N:]

        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        # if init_weight:
        #     self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        # self.spatial1 = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        # self.spatial2 = pixelConv(2, 1)
        self.spatial3 = Dynamic_conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial3(x_compress)
        scale = torch.sigmoid(x_out)
        # scale = self.spatial2(x_out, x_out)

        # x_out = self.spatial2(x_compress, x_compress)
        # scale = self.spatial1(x_out)
        #scale = torch.sigmoid(scale)  # broadcasting
        return x * scale


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
##########################################################################


## Dual Attention Block (DAB)
class DAB(nn.Module):
    def __init__(
            self, n_feat , kernel_size=3, reduction=16,
            bias=True, bn=False, act=nn.LeakyReLU(0.2, True)):

        super(DAB, self).__init__()
        modules_body = []
        for i in range(3):
            modules_body.append(
                BasicConv(n_feat, n_feat, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=True),
            )
        modules_body.append(BasicConv(n_feat, n_feat, 3, stride=1, padding=(3 - 1) // 2, relu=False))

        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)  ## Channel Attention
        self.body = nn.Sequential(*modules_body)
        self.conv1x1 = BasicConv(n_feat * 2, n_feat, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res

class attention(nn.Module):
    def __init__(
            self, n_feat , kernel_size=3, reduction=16,
            bias=True, bn=False, act=nn.LeakyReLU(0.2, True)):

        super(attention, self).__init__()
        self.norm = LayerNorm(n_feat)
        modules_body = []
        # for i in range(1):
        modules_body.append(
                BasicConv(n_feat, n_feat, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=True),
            )
        # modules_body.append(BasicConv(n_feat, n_feat, 3, stride=1, padding=(3 - 1) // 2, relu=False))

        # modules_att = [BasicConv(64, 4, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=True),
        #                BasicConv(4, 64, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)]
        # self.att_1 = nn.Sequential(*modules_att)
        # self.att_2 = nn.Sequential(*modules_att)
        self.SA = spatial_attn_layer()  ## Spatial Attention
        # self.CA = CALayer(n_feat, reduction)  ## Channel Attention
        self.CA = CA_attention(n_feat)
        # self.CA = simam_module()

        # self.soft = nn.Softmax(dim=2)
        self.body = nn.Sequential(*modules_body)
        # self.sub = nn.Sequential(BasicConv(n_feat, 8, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=True),
        #                          BasicConv(8, 8, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False))
        self.conv1x1 = BasicConv(n_feat * 2, n_feat, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True)

    def forward(self, x):
        res = self.body(self.norm(x))
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        # res += x
        return res + x
    # def forward(self, x):
    #     out = self.body(x)
    #     b_, c_, h_, w_ = out.shape
    #     p = self.att_1(out).reshape(b_, c_, h_ * w_)
    #     q = self.att_2(out).reshape(b_, c_, h_ * w_)
    #     # s = torch.matmul(p.transpose(1,2), q)
    #     s = self.SA(out)
    #     t = torch.matmul(p, q.transpose(1,2))
    #     out = out.reshape(b_, c_, h_ * w_)
    #     # s = torch.matmul(self.soft(s), out.transpose(1,2)).transpose(1,2).reshape(b_, c_, h_ , w_)
    #     t = torch.matmul(self.soft(t), out).reshape(b_, c_, h_ , w_)
    #     res = torch.cat([t, s], dim=1)
    #     res = self.conv1x1(res)
    #     # res += x
    #     return res + x

class Corrector(nn.Module):
    def __init__(self, nf = 64, nf_2=64, input_para=1,num_blocks= 5):
        super(Corrector, self).__init__()
        self.head_noisy = BasicConv(nf_2, nf_2, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        # # self.head_HR = nn.Conv2d(in_nc, nf // 2, 9, scale, 4)
        # body = [CRB_Layer(nf_1, nf_2) for _ in range(num_blocks)]
        # self.body = nn.Sequential(*body)
        # self.out  = BasicConv(nf_1, input_para, 3, stride=1, padding=(3 - 1) // 2, relu=False)

        # self.draw_conv1 = BasicConv(input_para, nf, 3, stride=1, padding=(3 - 1) // 2, relu=False)

        self.ConvNet = nn.Sequential(*[
            BasicConv(nf*2, nf, 1, stride=1, padding=(1 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            # BasicConv(nf, input_para, 3, stride=2, padding=(3 - 1) // 2, relu=True)
        ])
        # self.att = CALayer(nf)
        self.att = simam_module()
        # self.conv1 = BasicConv(nf, input_para, 3, stride=1, padding=(3 - 1) // 2, relu=False)


    def forward(self, feature_maps, noisy_map):
        # noisy_map = nn.functional.pixel_shuffle(noisy_map, 2)
        para_maps = self.head_noisy(noisy_map)
        # f = [feature_maps,para_maps]
        # f,_= self.body(f)
        cat_input =self.ConvNet(torch.cat((feature_maps, para_maps), dim=1))
        x = self.att(cat_input)
        # return self.conv1(self.att(cat_input)) #+ noisy_map
        return x

class Predictor(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(Predictor, self).__init__()

        self.ConvNet = nn.Sequential(*[
            BasicConv(in_nc, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True)
            # BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            # BasicConv(nf, 1, 3, stride=1, padding=(3 - 1) // 2, relu=True),
        ])
        # self.att = CALayer(nf)
        self.att = simam_module()
        # self.conv1 = BasicConv(nf, 1, 3, stride=1, padding=(3 - 1) // 2, relu=False)

    def forward(self, input):
        x = self.att(self.ConvNet(input))
        # return self.conv1(self.att(self.ConvNet(input)))
        return x

class Restorer(nn.Module):
    def __init__(self):
        super(Restorer, self).__init__()
        num_crb = 10
        para = 1

        n_feats = 64
        kernel_size = 3
        reduction = 16
        inp_chans = 1  # 4 RGGB channels, and 4 Variance maps
        act = nn.GELU()


        # self.draw_conv1 = BasicConv(para, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False)


        modules_head = [
            BasicConv(n_feats, n_feats, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False, bias=True)]

        self.head1 = BasicConv(n_feats, n_feats, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False, bias=True)
        self.head2 = BasicConv(n_feats, n_feats, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False, bias=True)

        # modules_body = [
        #     attention(n_feats) \
        #     for _ in range(num_crb)]

        # modules_body.append(
        #     BasicConv(n_feats, n_feats, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False, bias=True))
        # modules_body.append(act)

        modules_tail = [
            BasicConv(n_feats, n_feats, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False, bias=True)]
        self.conv1 = BasicConv(inp_chans, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        self.convU2 = BasicConv(3*n_feats, n_feats//2, 1, stride=1, padding=(1 - 1) // 2, relu=True)
        self.convU3 = BasicConv(n_feats//2, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=True)
        self.conv4 = BasicConv(2*n_feats, n_feats, 1, stride=1, padding=(1 - 1) // 2, relu=True)

        self.ConvNet = nn.Sequential(*[
            BasicConv(n_feats, n_feats, 1, stride=1, padding=(1 - 1) // 2, relu=True, bias=True),
            BasicConv(n_feats, n_feats, 1, stride=1, padding=(1 - 1) // 2, relu=True, bias=True),
            BasicConv(n_feats, n_feats, 1, stride=1, padding=(1 - 1) // 2, relu=True, bias=True),
            BasicConv(n_feats, n_feats // 16, 3, stride=1, padding=(3 - 1) // 2, relu=False, bias=True),
            # nn.Conv2d(nf1 + nf2, nf1, 3, 1, 1),
            # nn.LeakyReLU(0.2, True),
            nn.GELU(),
            BasicConv(n_feats // 16, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False, bias=True),
            # nn.Conv2d(nf1, nf1, 3, 1, 1),
            # CALayer(nf1),
            nn.Sigmoid()
            # BasicConv(n_feats, n_feats, 3, stride=2, padding=(3 - 1) // 2, relu=True,bias=True),
            # BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False, bias=True),
            # nn.AdaptiveAvgPool2d(1)
            # BasicConv(nf, input_para, 3, stride=1, padding=(3 - 1) // 2, relu=True)
        ])
        self.head = nn.Sequential(*modules_head)
        # self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = 32
        middle_blk_num = 2
        enc_blk_nums = [2, 2, 2, 2]
        dec_blk_nums = [2, 2, 2, 2]

        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[attention(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[attention(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[attention(chan) for _ in range(num)]
                )
            )


    def forward(self, last_maps,feature_maps, noisy_map):
        para_maps = self.head2(noisy_map)
        feature_maps1 = self.head1(feature_maps)
        conv_feature_maps = self.head(last_maps)
        x = self.convU2(torch.cat((conv_feature_maps, feature_maps1, para_maps), dim=1))
        # cat_input = self.conv4(torch.cat((conv_feature_maps, feature_maps1), dim=1))
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.convU3(x)
        para_maps1 = self.ConvNet(noisy_map)
        f = x*para_maps1 + conv_feature_maps

        # 11111111111111111111111111111111111111111111111111111111111111
        # paraMaps = self.head(noisy_map)
        #return self.tail(f+conv_feature_maps)
        return self.tail(f)
        # x = torch.cat([noisy_img, variance], dim=1)
        # x = self.head(x)
        # x = self.body(x)
        # x = self.tail(x)
        # x = noisy_img + x
        # return x
class DN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, para=1):
        super(DN, self).__init__()
        self.head = nn.Sequential(
            BasicConv(in_nc, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=False)

        )
        self.C = Corrector()
        self.P = Predictor(in_nc=3, nf=nf)
        self.P_tail = BasicConv(64, 1, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        self.F = Restorer()
        # self.delta = nn.Parameter(torch.randn(1,1))


        self.tail = nn.Sequential(
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, out_nc, 3, stride=1, padding=(3 - 1) // 2, relu=False)

        )
        self.initialize_weights()

    def forward(self, noisyImage):
        # noisyImage.transpose(2, 3).flip(dims=[2])
        X0 = self.head(noisyImage)
        m0 = self.P(noisyImage)
        n0 = self.P_tail(m0)

        X1 = self.F(X0, X0, m0) + X0
        # X1 = self.F(X0, X0,0) + X0
        # M1 = self.F(M0, n0) + M0
        outs = []
        outs.append(m0)
        # outs.append(n0)
        for i in range(4):
            m0 = m0 + self.C(X1, m0)
            n0 = n0 + self.P_tail(m0)
            X1 = self.F(X1, X0, m0) + X0
            # X1 = self.F(X1, X0,0) + X0
            outs.append(m0)
            # outs.append(n0)
        return outs,self.tail(X1)
        #return self.tail(X1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                #torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.xavier_uniform_(m.weight.data)
                #torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Net = DN()
# # input = torch.rand(1,3,128,128).cuda()
# # input = torch.rand(1,3,80,80).cuda()
# # print(Net(input))
# # print_network(Net)
# from ptflops import get_model_complexity_info
# inp_shape = (3, 512, 512)
# macs, params = get_model_complexity_info(Net, inp_shape, verbose=False, print_per_layer_stat=False)
# print(macs,params)


