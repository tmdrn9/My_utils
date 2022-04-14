from torch import nn

class MTLnet(nn.Module):

    def __init__(self,basemodel,shared_layer_size,tower_h1,tower_h2,output_size1,output_size2):
        super(MTLnet, self).__init__()
        self.sharedlayer = basemodel
        self.tower1 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h2, output_size1)
        )
        self.tower2 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h2, output_size2)
        )        

    def forward(self, xxx):
        shared = self.sharedlayer(xxx)
        out1 = self.tower1(shared)
        out2 = self.tower2(shared)
        return out1, out2