import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_model, out_model):
        super(Net, self).__init__()
        self.in_model = in_model
        self.out_model = out_model
        self.fc0 = nn.Linear(3072, 2024)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(2024)
        self.fc1 = nn.Linear(2024, 5)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, inputs):
        in_inputs, out_inputs = inputs
        in_outputs = self.in_model(in_inputs)
        in_outputs = in_outputs.unsqueeze(-1) # 增加一个维度，由torch.Size([8, 2048])变为torch.Size([8, 2048, 1])
        #print('车内 outputs size:', in_outputs.size())

        out_outputs = self.out_model(out_inputs)
        out_outputs = torch.reshape(out_outputs, (out_outputs.shape[0], -1, 1)) # 维度[8, 512, 1, 2]、[8, 1024, 1]
        #print('车外 outputs size:', out_outputs.size())

        in_outputs = in_outputs.to("cuda:0")
        concatenated = torch.cat((in_outputs, out_outputs), dim=1)
        #print('拼接后的 size:', concatenated.size())
        concatenated = concatenated.view(-1, 3072)
        #print('拼接后的 size:', concatenated.size())

        concatenated = concatenated.to(self.fc0.weight.device)
        fc0_output = self.fc0(concatenated)
        #print('FC 0 outputs size:', fc0_output.size())

        relu_output = self.relu(fc0_output)
        #print('relu outputs size:', relu_output.size())

        bn_output = self.bn(relu_output)
        #print('BN outputs size:', bn_output.size())

        fc1_output = self.fc1(bn_output)
        #print('FC 1 outputs size:', fc1_output.size())

        softmax_output = self.softmax(fc1_output)
        #print('Softmax outputs size:', softmax_output.size())

        # # print('车内 outputs size:', in_outputs.size())
        # # print('车外 outputs size:', out_outputs.size())
        # # print('车外 outputs调整后的size:', out_outputs.size())
        # # print('拼接 outputs size:', concatenated.size())
        # # print('FC 0 outputs size:', fc0_output.size())
        # # print('FC 1 outputs size:', fc1_output.size())
        # # print('Softmax outputs size:', softmax_output.size())
        return softmax_output
