import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

##word embeddings layer is a (d by V) matrix where V is vocab size and d is embedding size.
##    The V is the one hot vector and the d is the dimensional embedding for a
##    particular token. We just avoid this d by V multiplication by taking
##    embedding matrix of length d for the token

class BOW_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)

        self.fc_hidden = nn.Linear(no_of_hidden_units,no_of_hidden_units)
        self.bn_hidden = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        
        self.loss = nn.BCEWithLogitsLoss()
    '''Intuitively each of these embeddings is a 1-hot vector of token
    ids with tokens corresponding to words'''
    def forward(self, x, t):
    
        bow_embedding = []
        for i in range(len(x)):
            lookup_tensor = Variable(torch.LongTensor(x[i])).cuda()
            #lookup_tensor = sequence of token ids fed into embedding layer
            #returns torch tensor of length sequence_length x embedding length
            embed = self.embedding(lookup_tensor) 
            #Then you take mean over dimension corresponding to sequence length
            #and this mean operation turns it into bag of words, so sentence averages
            embed = embed.mean(dim=0)
            bow_embedding.append(embed)
        bow_embedding = torch.stack(bow_embedding)
    
        h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        # We can remove dropout and also add a second or third layer to overfit
##        h =  F.relu(self.bn_hidden(self.fc_hidden(h)))
        h = self.fc_output(h)
    
        return self.loss(h[:,0],t), h[:,0]
