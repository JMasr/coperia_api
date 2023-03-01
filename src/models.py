import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence


# Torch utils
def activations(act):
    """
    Interface to fetch activations
    """
    activ = {'Tanh': nn.Tanh(), 'ReLU': nn.ReLU(), 'Sigmoid': nn.Sigmoid()}
    act = activ[act]

    if act is not None:
        return act
    else:
        raise ValueError('Unknown activation, add it in activations dictionary in models.py')


class bce_loss(nn.Module):
    """
    Class interface to compute BCE loss
    Default uses mean reduction equal weight for both positive and negative samples
    """

    def __init__(self, reduction='mean', pos_weight=torch.tensor([1])):
        super(bce_loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)

    def forward(self, net_out, ref):
        return self.criterion(net_out, ref)


class FFClassificationHead(nn.Module):
    def __init__(self, args):
        super(FFClassificationHead, self).__init__()

        self.inDim = args['input_dimension']
        self.units = [self.inDim] + [item for item in args['units'] if item > 0]
        self.num_layers = len(self.units) - 1

        self.activation_type = args['activation']
        self.dropout_p = args['dropout']

        for i in range(self.num_layers):
            setattr(self, 'linearlayer_' + str(i), nn.Linear(self.units[i], self.units[i + 1]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(self.dropout_p))
        self.linearOut = nn.Linear(self.units[-1], 1)
        self.activation = activations(self.activation_type)

    def forward(self, inputs):

        x = torch.vstack(inputs)

        for i in range(self.num_layers):
            x = getattr(self, 'linearlayer_' + str(i))(x)
            x = self.activation(x)
            x = getattr(self, 'dropout_' + str(i))(x)
        x = self.linearOut(x)
        return [x[i,] for i in range(x.shape[0])]


# LSTM ENCODER classifier
class LSTMEncoder(nn.Module):
    """ Stacked (B)LSTM Encoder
    Arguments:
    args: Dictionary with below entries
    input_dimenstion: (integer), Dimension of the feature vector input
    units: (integer), Number of LSTM units. Default: 128
    num_layers: (integer), Number of layers in the stacked LSTM. Default: 2
    bidirectional: (bool), if True biLSTM will be used. Default: True
    apply_mean_norm: (bool), subtract the example level mean. Default: False
    apply_var_norm: (bool), normalize by standard deviation. Default: False
    pooltype: (['average' or 'last']). Default: 'average'
    ----> 'average': average of the LSTM output along time dimension is the embedding
    ----> 'last': LSTM hidden state at the last time-step of the last layer is the embedding
    dropout: (float), Dropout probability. Default: 0
    """

    def __init__(self, args):
        super(LSTMEncoder, self).__init__()
        self.inDim = args['input_dimension']
        self.units = args.get('units', 128)
        self.num_layers = args.get('num_layers', 2)
        self.bidirectional = args.get('bidirectional', False)

        self.apply_mean_norm = args.get('apply_mean_norm', False)
        self.apply_var_norm = args.get('apply_var_norm', False)
        self.dropout_p = args.get('dropout', 0)
        assert self.dropout_p < 1

        self.pooltype = args.get('pooltype', False)
        assert self.pooltype in ['average', 'last']

        self.LSTM = nn.LSTM(self.inDim,
                            self.units,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True,
                            dropout=self.dropout_p)

    def forward(self, inputs):
        """
        inputs: a list of torch tensors
        The tensors can be of varying length.
        """
        inlens = [x.shape[0] for x in inputs]
        if self.apply_mean_norm:
            inputs = [F - torch.mean(F, dim=0) for F in inputs]
        if self.apply_var_norm:
            inputs = [F / torch.std(F, dim=0) for F in inputs]

        x = pad_sequence(inputs, batch_first=True)
        x = pack_padded_sequence(x, inlens, batch_first=True, enforce_sorted=False)
        x, hc = self.LSTM(x)

        if self.pooltype == 'average':
            x, _ = pad_packed_sequence(x, batch_first=True)
            x = torch.sum(x, dim=1)
            x = torch.div(x, torch.tensor(inlens).unsqueeze(1).repeat(1, x.shape[1]).to(x.device))
        elif self.pooltype == 'last':
            if self.bidirectional:
                x = hc[0][-2:, :, :].transpose(0, 1).reshape(hc[0].shape[1], 2 * hc[0].shape[2])
            else:
                x = hc[0][-1, :, :]
        else:
            raise ValueError('Unknown pooling method')

        return [x[i, :].view(1, x.shape[1]) for i in range(x.shape[0])]


# LSTM classifier
class LSTMClassifier(nn.Module):
    """
    LSTM Classifier architecture
    """

    def __init__(self, args):
        super(LSTMClassifier, self).__init__()

        self.input_dimension = args['input_dimension']
        self.lstm_encoder_units = args['lstm_encoder_units']
        self.lstm_num_layers = args['lstm_num_layers']
        self.lstm_bidirectional = args['lstm_bidirectional']
        self.lstm_dropout_p = args['lstm_dropout']
        self.lstm_pooling = args['lstm_pooling']
        self.apply_mean_norm = args['apply_mean_norm']
        self.apply_var_norm = args['apply_var_norm']

        encoder_args = {'input_dimension': self.input_dimension, 'units': self.lstm_encoder_units,
                        'num_layers': self.lstm_num_layers, 'bidirectional': self.lstm_bidirectional,
                        'apply_mean_norm': self.apply_mean_norm, 'apply_var_norm': self.apply_var_norm,
                        'dropout': self.lstm_dropout_p, 'pooltype': self.lstm_pooling}

        self.encoder = LSTMEncoder(encoder_args)

        temp = args['classifier_units']
        if type(temp) == list:
            self.classifier_units = temp
        else:
            self.classifier_units = [temp]
        self.classifier_activation = args['classifier_activation']
        self.classifier_dropout_p = args['classifier_dropout']
        cls_idim = 2 * self.lstm_encoder_units if self.lstm_bidirectional else self.lstm_encoder_units
        classifier_args = {'input_dimension': cls_idim, 'units': self.classifier_units,
                           'dropout': self.classifier_dropout_p, 'activation': self.classifier_activation}

        self.classifier = FFClassificationHead(classifier_args)
        self.criterion = bce_loss()

    def init_encoder(self, params):
        """
        Initialize the feature encoder using a pre-trained model
        """
        self.encoder.load_state_dict(params)

    def init_classifier(self, params):
        """
        Initialize the classification-head using a pre-trained classifier model
        """
        self.classifier.load_state_dict(params)

    def predict(self, inputs):
        """
        Prediction of the classifier score
        """
        return self.classifier(self.encoder(inputs))

    def predict_proba(self, inputs):
        """
        Prediction of the posterior probability
        """
        return [torch.sigmoid(item) for item in self.predict(inputs)]

    def forward(self, inputs, targets):
        """
        Forward pass through the network and loss computation
        """
        return self.criterion(torch.stack(self.predict(inputs)), torch.stack(targets))