import torch #导入PyTorch库
import torch.nn as nn #导入PyTorch的神经网络模块
import math     #导入数学库

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):#初始化函数，接受模型维度 和 头数作为输入
        super().__init__()  #调用父类的初始化函数

        self.num_heads = num_heads #定义头数
        self.d_k = d_model // num_heads #计算每个头的维度

        self.Wq = nn.Linear(d_model, d_model) #定义Query的线性投影矩阵 shape为(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model) #定义Key的线性投影矩阵 shape为(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model) #定义Value的线性投影矩阵 shape为(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model) #定义输出的线性投影矩阵，将多头的输出映射回原始维度 shape为(d_model, d_model)

    def forward(self, x):

        B, T, C = x.shape #获取输入的批次大小、序列长度和特征维度

        Q = self.Wq(x) #将输入x通过wq得到Query矩阵 shape为(B, T, d_model)
        K = self.Wk(x) #将输入通过wk得到Key矩阵 shape为(B, T, d_model)
        V = self.Wv(x) #将输入通过wv得到Value矩阵 shape为(B, T, d_model)

        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1,2) #将Q拆分为多头，并调整维度 shape为(B, num_heads, T, d_k)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1,2) #将K拆分为多头，并调整维度 shape为(B, num_heads, T, d_k)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1,2) #将V拆分为多头，并调整维度 shape为(B, num_heads, T, d_k)

        score = torch.matmul(Q, K.transpose(-2,-1)) 
        #将K进行转置，转置后shape为(B, num_heads, d_k, T)，与Q相乘得到注意力分数 shape为(B, num_heads, T, T)
        #matmul函数用于矩阵乘法，Q和K的最后两个维度进行矩阵乘法，得到每个头的注意力分数
        score = score / math.sqrt(self.d_k) #进行缩放，防止数值过大导致softmax梯度消失

        attn = torch.softmax(score, dim=-1) #对注意力分数进行softmax归一化，得到注意力权重 shape为(B, num_heads, T, T)

        out = torch.matmul(attn, V) #将注意力权重与V相乘，得到每个头的输出 shape为(B, num_heads, T, d_k)

        out = out.transpose(1,2).contiguous() #将输出的维度调整回(B, T, num_heads, d_k)，并使用contiguous()确保内存连续
        out = out.view(B, T, C) #将多头的输出重新组合成原始维度 shape为(B, T, d_model)

        out = self.fc(out) #将多头的输出通过线性层映射回原始维度 shape为(B, T, d_model)

        return out #返回多头注意力的输出