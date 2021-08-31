from torch import nn

class SimpleModel(nn.Module):
    def __init__(self, input_shape , embedding_size):
      super().__init__()
      ## base model ###################################################################
      self.encoder_hidden_layer = nn.Linear(in_features=input_shape, out_features=embedding_size)
      self.encoder_output_layer = nn.Linear(in_features=embedding_size, out_features=embedding_size)

      ## mlp projection head ##########################################################
      self.backbone = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                                       nn.ReLU(), self.encoder_output_layer , self.encoder_hidden_layer)

    def forward(self , features):
      return self.backbone(features)
