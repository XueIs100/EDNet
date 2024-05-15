import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input[step]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)

class Decoder(nn.Module):
    def  __init__(self, num_step, num_channel):
        super(Decoder, self).__init__()
        self._all_layers = []
        self.num_step = num_step
        self.num_channel = num_channel
        for i in range(self.num_step):
            name = 'conv{}'.format(i)
            conv = nn.Conv2d(self.num_channel, 3, 1, stride=1, padding=0)
            setattr(self, name, conv)
            self._all_layers.append(conv)

    def forward(self, input):
    	output = []
    	for i in range(self.num_step):
    		name = 'conv{}'.format(i)
    		y = getattr(self, name)(input[i])
    		output.append(y)
    	return output

# class Encoder(nn.Module):
#     def __init__(self, hidden_channels, sample_size, sample_duration):
#         super(Encoder, self).__init__()
#         self.convlstm = ConvLSTM(input_channels=3, hidden_channels=hidden_channels, kernel_size=3, step=sample_duration,
#                         effective_step=[sample_duration-1])
# ################## W/o output decoder
#         self.conv2 = nn.Conv2d(32, 3, 1, stride=1, padding=0)
# ################## With output decoder
# #        self.decoder = Decoder(sample_duration, 32)
#     def forward(self, x):
#         b,t,c,h,w = x.size()
#         x = x.permute(1,0,2,3,4)
#         output_convlstm, _ = self.convlstm(x)
# #        x = self.decoder(output_convlstm)
#         x = self.conv2(output_convlstm[0])
#         return x

class Encoder(nn.Module):
    def __init__(self, hidden_channels, sample_size, sample_duration):
        super(Encoder, self).__init__()
        self.convlstm = ConvLSTM(input_channels=3, hidden_channels=hidden_channels, kernel_size=3, step=sample_duration,
                        effective_step=[sample_duration-1])
        self.adjust_dim = nn.Linear(1024, 2048)
        self.fc = nn.Linear(2048, 5)  # 这里也需要将输出特征大小与分类数匹配
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.permute(1, 0, 2, 3, 4)
        output_convlstm, _ = self.convlstm(x)
        print('ConvLSTM output size:', output_convlstm[0].size())
        out_outputs = torch.reshape(output_convlstm[0], (output_convlstm[0].shape[0], -1, 1))
        print('经过reshape后的车外输出：', out_outputs.size())
        # 使用全连接层调整车外特征的维度
        out_outputs_adjusted = self.adjust_dim(out_outputs.squeeze(-1))  # 移除最后一个维度后再调整维度
        print('adjust_dim：', out_outputs_adjusted.size())
        fc_output = self.fc(out_outputs_adjusted)
        print('fc_output:', fc_output.size())
        softmax_output = self.softmax(fc_output)
        return softmax_output

def encoder(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = Encoder(**kwargs)
    return model

# def test():
# #if __name__ == '__main__':
#     # gradient check
#
#     convlstm = ConvLSTM(input_channels=48, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=5,
#                         effective_step=[2,4]).cuda()
#     loss_fn = torch.nn.MSELoss()
#
#     input = Variable(torch.randn(1, 48, 64, 64)).cuda()
#     target = Variable(torch.randn(1, 32, 64, 64)).double().cuda()
#
#     output = convlstm(input)
#     output = output[0][0].double()
#
#     res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
#     print(res)
#
#
# def test_convlstm():
#     """Constructs a convlstm model.
#     """
#     model = encoder(hidden_channels=[128, 64, 64, 32], sample_size=[112,112], sample_duration=4).cuda()
#     input = Variable(torch.randn(20, 3, 4, 112, 112)).cuda()
#
#     output = model(input)
#     print(output.size())

# def encoder(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = Encoder(**kwargs)
#     return model

# if __name__ == '__main__':
#    test_convlstm()


'''
    input = Variable(torch.randn(20, 4, 3, 112, 112)).cuda()
    五个数字分别代表输入张量的维度大小，分别是：
        20：代表输入张量中的样本数。在这个例子中，输入张量包含了20个不同的样本。
        3：代表输入张量中每个样本的通道数。在这个例子中，每个样本都是RGB图像，因此通道数为3。
        4：代表输入张量中时间步的数量。在这个例子中，模型期望每个样本都是由4帧图像组成的视频序列。
        112：代表输入张量中每个图像的高度。在这个例子中，输入张量中的图像都具有112个像素的高度。
        112：代表输入张量中每个图像的宽度。在这个例子中，输入张量中的图像都具有112个像素的宽度。
    因此，这个输入张量可以被认为是一个由20个RGB图像序列组成的批次，每个序列都包含了4帧图像，每帧图像都是一个112x112的矩阵。

    返回一个符合均值为0，方差为1的正态分布（标准正态分布）中填充随机数的张量：
    torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
        size(int…) --定义输出张量形状的整数序列。可以是数量可变的参数，也可以是列表或元组之类的集合。
            如果size是一个整数，则生成的张量是一维的，并且包含了指定数量的随机数。
            如果size是一个元组，则会生成具有指定形状的张量。
        out(Tensor, optional) --输出张量
        dtype(torch.dtype, optional) --返回张量所需的数据类型。默认:如果没有，使用全局默认值
        layout(torch.layout, optional) --返回张量的期望布局。默认值:torch.strided
        device(torch.device, optional) --返回张量的所需 device。默认:如果没有，则使用当前设备作为默认张量类型.(CPU或CUDA)
        requires_grad(bool, optional) –autograd是否应该记录对返回张量的操作(说明当前量是否需要在计算中保留对应的梯度信息)默认False
'''
'''
首先，定义了一个编码器模型，其隐藏通道数分别为[128, 64, 64, 32]。这表示编码器包含了4个卷积层，每个卷积层的输出通道数分别为128、64、64和32。
接下来，使用.cuda()将模型移动到GPU上进行加速。
然后，创建了一个输入变量input，它的维度是[20, 3, 4, 112, 112]。解释一下各维度的含义：
    20：表示一次输入的样本数。
    3：表示输入图像的通道数，例如RGB图像有3个通道。
    4：表示输入的帧数，即视频的长度。
    112, 112：表示输入图像的空间尺寸，这里是112x112像素。
接下来，将输入数据也移动到GPU上。
然后，将输入数据通过模型进行处理，得到输出结果output。最后，打印输出结果及其大小。

这段代码的作用是将一批图像数据输入到编码器模型中，经过模型的前向传播计算，得到对应的特征表示。
这个特征表示可以用来进行后续的任务，如图像分类、目标检测等。
'''