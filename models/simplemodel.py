from torch import nn

class SimpleModel(nn.Module):
    def __init__(self, input_shape , embedding_size):
      super().__init__()
      ## base model ###################################################################
      self.encoder_hidden_layer = nn.Linear(in_features=input_shape, out_features=embedding_size)
      self.encoder_output_layer = nn.Linear(in_features=embedding_size, out_features=embedding_size)
      self.relu = nn.ReLU()

      ## mlp projection head ##########################################################
      self.backbone = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                                       nn.ReLU())

    def forward(self , features):
      l1 = self.encoder_hidden_layer(features)
      l2 = self.encoder_output_layer(l1)
      l3 = self.relu(l2)
      return self.backbone(l3)
      
    def get_features(self , features):
      l1 = self.encoder_hidden_layer(features)
      l2 = self.encoder_output_layer(l1)
      l3 = self.relu(l2)
      return l3

