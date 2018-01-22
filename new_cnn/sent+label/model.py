#encoding:utf-8
import torch
import  torch.nn as nn
import torch.nn.functional as F
from torch.autograd import  Variable
torch.manual_seed(1)
#双向lstm
class BiLSTM(nn.Module):
    def __init__(self,args):
        super(BiLSTM,self).__init__()
        self.args = args
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        #self.pos_embedding = nn.Embedding(args.pos_size, args.pos_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(args.embed_dim, args.hidden_size,
                            bidirectional=True,
                            batch_first=True,
                            dropout=args.dropout)
        self.linearOut = nn.Linear(args.hidden_size*2, args.class_num)
        self.softmax = nn.Softmax()
    #def forward(self,x,p):
    def forward(self,x):
        hidden = (Variable(torch.zeros(2, x.size(0), self.args.hidden_size)),
                  Variable(torch.zeros(2, x.size(0), self.args.hidden_size)))
		
        x= self.embed(x)
        #print x_embed.size()
        #print hidden[0].size(),hidden[1].size()
        #p_embed = self.pos_embedding(p)
        #x = torch.cat((x_embed, p_embed), 2)
        #x=self.dropout(x)
        #print type(x),hidden[0],hidden[1]
        x, lstm_h = self.lstm(x, hidden)
        x = F.tanh(x)
        # for idx in range(x.size(0)):
        #     h= torch.mm(x[idx], self.myw)
        #     if idx == 0:
        #         output = torch.unsqueeze(h, 0)
        #     else:
        #         output = torch.cat([output, torch.unsqueeze(h, 0)], 0)
        x=torch.transpose(x,1,2)
        x = F.max_pool1d(x, x.size(2))
        x = self.linearOut(x.view(x.size(0),1,-1))
        x = self.softmax(x.view(x.size(0),-1))

        return x
