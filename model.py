
"""
This module contains the class for the model, as well as PyTorch 
block CNNBlock which is used in for the model.

An instance of ChenModel is created within apipeline.py, and sent
to the Trainer class which fits and evaluates it.

Suggested things to try:
    • Batch norm
    • GeLU activation
"""

from itertools import accumulate
import torch
from torch import nn
from torch.nn import functional as F

class CNNBlock(nn.Module):
    """
    This is a block of multiple convolutional layers.
    We repeat this 5 times within the model.
    """

    def __init__(self, activation, activ_params, segment_size=1024, residual=False):
        # This is necessary for any nn.Module class.
        super().__init__()

        self.activation = activation
        self.activ_params = activ_params
        self.residual = residual

        # Here we define the convolutional layers.
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)

        # This is a conv layer with a stride of 2, which acts like a max pooling layer, 
        # in that it reduces the size of the input by about half.
        # This is given in the code, but in the paper they indicate a genuine max pooling.
        # I have defined the conv layer, the max pooling layer, and an identity layer.
        
        # Here we calculate the padding size.
        convreduce_kernel_size = 24
        assert segment_size % 2 == 0, 'Please choose an even segment_size.'
        assert convreduce_kernel_size % 2 == 0, 'Please choose an even convreduce_kernel_size.'

        # Choose one of the below.
        #padding = int((convreduce_kernel_size + segment_size)/2 - 1) # To keep the output size the same.
        padding = int(convreduce_kernel_size/2 - 1) # To reduce the output size by half.

        # Choose one of the below.
        #self.convreduce = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=convreduce_kernel_size, stride=2, padding=padding)
        #self.convreduce = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.convreduce = nn.MaxPool1d(kernel_size=2, stride=2)
        #self.convreduce = nn.Identity()

    def forward(self, x):
        # We apply a leaky relu between the conv layers.
        
        raw = x
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.3)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.3)
        x = self.convreduce(x)
        x = F.leaky_relu(x, 0.3)
        
        # Apply a residual connection if specified in config.
        if self.residual:
            residual = self.convreduce(raw)
            x = x + residual

        return x

class ChenModel(nn.Module):
    """
    This is the main model class.
    
    Citation:

    Chen, Tsai-Min, et al.
    "Detection and classification of cardiac arrhythmias by a challenge-best
    deep learning neural network model."
    Iscience 23.3 (2020): 100886.
    https://www.sciencedirect.com/science/article/pii/S2589004220300705 
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dictionary
            The model config. This is mainconfig['model'], where mainconfig is the config file
            as loaded within apipeline.py.
        """
        super().__init__()

        # Save the config internally.
        self.config = config

        # Define the dropout rate and activation function.
        self.p_drop = config.get('dropout', 0.2)
        activ_name, self.activ_params = config.get('activation', 'relu')
        self.activation = getattr(F, activ_name)

        # Create the conv blocks.
        for i in range(self.config.get('conv_blocks', 5)):
            block = CNNBlock(
                    activation=self.activation,
                    activ_params=self.activ_params,
                    residual=self.config.get('residual', False),
                    )
            setattr(self, 'conv{}'.format(i), block)

        # Create the GRU. The parameters batch_first indicates that the
        # batch size is the first tensor dimension.
        gru_hidden_size = config.get('gru_hidden_size', 25)
        self.gru = torch.nn.GRU(input_size=12, hidden_size=gru_hidden_size, batch_first=True, bidirectional=True)

        # Create the attention layer. I use an MLP for the attention network.
        attn_hidden_size = config.get('attn_hidden_size', 10)
        self.linatten_1 = nn.Linear(in_features=gru_hidden_size*2, out_features=attn_hidden_size)
        self.linatten_2 = nn.Linear(in_features=attn_hidden_size, out_features=1)

        # This is the output layer.
        self.outlayer_1 = nn.Linear(in_features=gru_hidden_size*2, out_features=self.config.get('out_hidden_size', 25))
        self.outlayer_2 = nn.Linear(in_features=self.config.get('out_hidden_size', 25), out_features=self.config.get('num_classes', 9))
        # self.outlayer_2 = nn.Linear(in_features=self.config.get('out_hidden_size', 25), out_features=1)

        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()

        self.initial_print = 1

    def logistic_norm(self, x, dim=1):
        """
            This is a form of mapping logits to probabilities.
            It is similar to the softmax function, but uses the 
            sigmoid function instead of the exponential.

            Parameters
            ----------
            x : torch.tensor
                Input tensor, assumed to be of dimensionality: (Batch, Sequence length, Features=1).

            Returns
            -------
            torch.tensor
                Tensor of dimensionality: (Batch, Sequence length, Features=1).
        """
        assert dim==1, "If dim is not 1, make very sure that this is what you want to do."
        sig = torch.sigmoid(x)
        norm = sig.sum(dim, keepdim=True)
        out = sig / norm
        return out

    def forward(self, x):

        # Apply convolutions
        for i in range(self.config.get('conv_blocks', 5)):
            block = getattr(self, 'conv{}'.format(i))
            x = block(x)
            if self.initial_print:
                print(x.shape, x[:10,0,0].cpu().detach().numpy().std())
            x = F.dropout(x, p=self.p_drop)
        self.initial_print = 0

        # Apply GRU
        x = x.swapaxes(1, 2) # Swap features and sequence dimensions
        x, _ = self.gru(x)

        x = F.leaky_relu(x, 0.3)
        x = F.dropout(x, p=self.p_drop)

        # Apply Attention layer
        raw_attentions = self.linatten_2(F.leaky_relu(self.linatten_1(x), 0.3))
        
        # Normalise the attentions
        norm_form = self.config.get('attention_norm_form', 'logistic')
        if norm_form in ['logistic', 'sigmoid']:
            attentions = self.logistic_norm(raw_attentions, dim=1)
        elif norm_form == 'softmax':
            attentions = F.softmax(raw_attentions, dim=1)

        # We apply the same weight to each dimension of the GRU output
        attentions = attentions.repeat(1, 1, x.shape[2])
        # Apply weightings
        x = x * attentions
        # Sum the weighted GRU outputs
        x = x.sum(dim=1)

        # Pre-output processing
        x = F.leaky_relu(x, 0.3)
        x = F.dropout(x, p=self.p_drop)

        # Apply fully-connected output layers
        x = self.outlayer_1(x)
        x = F.leaky_relu(x, 0.3)
        x = F.dropout(x, p=self.p_drop)
        y = self.outlayer_2(x)

        return y