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
        # self.fc = nn.Linear(hidden_channels[-1], 5)
        # Conv-Block 0:
        self.conv0 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm2d(64)

        # Conv-Block 1:
        self.conv1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)

        # Conv-Block 2:
        self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(256)

        # Conv-Block 3:
        self.conv3 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(1024, 5)

    def forward(self, x):
        b,t,c,h,w = x.size()
        x = x.permute(1,0,2,3,4)
        output_convlstm, _ = self.convlstm(x)
        print('ConvLSTM output size:', output_convlstm[0].size())

        # Conv-Block 0:
        conv_0 = self.conv0(output_convlstm[0])
        relu_0 = self.relu(conv_0)
        maxpool_0 = self.maxpool0(relu_0)
        bn_0 = self.bn0(maxpool_0)
        rsize_bn_0 = bn_0[:, :, :37, :59]
        # print('conv_0 output size:', conv_0.size())
        # print('relu_0 output size:', relu_0.size())
        # print('maxpool_0 output size:', maxpool_0.size())
        # print('bn_0 output size:', bn_0.size())
        # print('rsize_bn_0 output size:', rsize_bn_0.size())

        # Conv-Block 1:
        conv_1 = self.conv1(rsize_bn_0)
        relu_1 = self.relu(conv_1)
        maxpool_1 = self.maxpool1(relu_1)
        bn_1 = self.bn1(maxpool_1)
        rsize_bn_1 = bn_1[:, :, :12, :20]
        # print('conv_1 output size:', conv_1.size())
        # print('relu_1 output size:', relu_1.size())
        # print('maxpool_1 output size:', maxpool_1.size())
        # print('bn_1 output size:', bn_1.size())
        # print('rsize_bn_1 output size:', rsize_bn_1.size())

        # Conv-Block 2:
        conv_2 = self.conv2(rsize_bn_1)
        relu_2 = self.relu(conv_2)
        maxpool_2 = self.maxpool2(relu_2)
        bn_2 = self.bn2(maxpool_2)
        rsize_bn_2 = bn_2[:, :, :4, :7]
        # print('conv_2 output size:', conv_2.size())
        # print('relu_2 output size:', relu_2.size())
        # print('maxpool_2 output size:', maxpool_2.size())
        # print('bn_2 output size:', bn_2.size())
        # print('rsize_bn_2 output size:', rsize_bn_2.size())

        # Conv-Block 3:
        conv_3 = self.conv3(rsize_bn_2)
        relu_3 = self.relu(conv_3)
        maxpool_3 = self.maxpool3(relu_3)
        bn_3 = self.bn3(maxpool_3)
        rsize_bn_3 = bn_3[:, :, :1, :2]
        # print('conv_3 output size:', conv_3.size())
        # print('relu_3 output size:', relu_3.size())
        # print('maxpool_3 output size:', maxpool_3.size())
        # print('bn_3 output size:', bn_3.size())
        # print('rsize_bn_3 output size:', rsize_bn_3.size())
        reshape_outputs = torch.reshape(rsize_bn_3, (rsize_bn_3.shape[0], -1, 1))
        concatenated = reshape_outputs.view(-1, 1024)
        fc1_output = self.fc1(concatenated)
        # output_fc = self.fc(output_convlstm[0].view(b, -1))
        # print('output_fc output size:', output_fc.size())
        return fc1_output


def encoder(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = Encoder(**kwargs)
    return model
