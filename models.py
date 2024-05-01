import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision

class ConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 5) # out = in - (kernel-1)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 24, 3)

        self.pool = nn.MaxPool2d(2,2) # out = in / shape
        self.pool_avg = nn.AvgPool2d(2,1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.50)

        self.fc1 = nn.Linear(17496, 4096)
        self.fc2 = nn.Linear(4096, num_classes) # 3040
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool_avg(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        #x = self.dropout2(x)
        x = F.sigmoid(self.fc2(x))
        return x
    
class VITNet_v2(nn.Module):
    def __init__(self, _patch_size, _num_classes, _img_size, _hidden_dim,
                 _num_heads = 8, _hidden_dim_mult = 4, _num_encoder_layer = 2):
        super(VITNet_v2, self).__init__()
        self.patch_size = _patch_size
        self.hidden_dim = _hidden_dim
        self.num_patches = int(_img_size / _patch_size) ** 2

        self.linear_in_1 = nn.Linear(self.patch_size ** 2, self.hidden_dim)
        self.linear_in_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.batchnorm_1 = nn.BatchNorm1d(self.num_patches)

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        self.pos_embed = nn.Parameter(torch.tensor(self.get_positional_embeddings(self.num_patches + 1, self.hidden_dim)))
        self.pos_embed.requires_grad = False

        self.encoder_layer = torch.nn.TransformerEncoderLayer(self.hidden_dim, _num_heads, 
                                                              self.hidden_dim * _hidden_dim_mult, 
                                                              dropout=0.25, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, _num_encoder_layer)

        self.dropout = nn.Dropout(p=0.50)
        self.batchnorm_2 = nn.BatchNorm1d(2048)

        self.linear_2 = nn.Linear(self.hidden_dim, 2048)
        self.linear_3 = nn.Linear(2048, 2048)
        self.output_layer = nn.Linear(2048, _num_classes)

    def get_positional_embeddings(self, sequence_length, d):
        res = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                res[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return res

    def patchify(self, x, batch_size):
        #print(type(x))
        # TODO type checking
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        x = x.reshape(batch_size, 1, -1, self.patch_size, self.patch_size).flatten(3)
        x = torch.squeeze(x, 1)
        return x
    
    def forward(self, x):
        n, c, h, w = x.shape
        patches = self.patchify(x, n)

        # Double pass through hidden_dim nodes, batch norm, then dropout for training aid.
        tokens = self.batchnorm_1(self.linear_in_1(patches))
        tokens = F.relu(self.linear_in_2(tokens))
        tokens = self.dropout(tokens)

        # Prepend the class token to the front of each batch.
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embed

        # Run through the encoder blocks.
        out = self.encoder(out)

        # Get the class token (first elm of T in :(B, T, Dim)).
        out = out[:, 0]
        out = F.relu(self.batchnorm_2(self.linear_2(out)))
        out = self.dropout(out)
        out = F.tanh(self.linear_3(out))
        out = F.sigmoid(self.output_layer(out))
        return out
def construct_vit_v2(num_classes, BATCHES_PER_ITR):
    model = VITNet_v2(_patch_size = 16, _num_classes = num_classes, _img_size = 128, 
                      _hidden_dim = 512, _num_heads = 16, _hidden_dim_mult = 4, _num_encoder_layer = 8)
    # V2 Scheduler code
    vit_optimizer = optim.SGD(model.parameters(), lr=0.050, momentum=0.9) # lr from 0.060 to 0.050 after epoch 89
                                                                                # reset to 0.050 after epoch 132
    vit_scheduler = optim.lr_scheduler.ExponentialLR(vit_optimizer, 0.96) # Swapped to this after epoch 89
        #vit_scheduler = optim.lr_scheduler.CosineAnnealingLR(vit_optimizer, 12, 0.020)
    return model, vit_optimizer, vit_scheduler
    
class VITNet_v3(nn.Module):
    def __init__(self, _patch_size, _num_classes, _img_size, _hidden_dim,
                 _num_heads = 8, _hidden_dim_mult = 4, _num_encoder_layer = 2):
        super(VITNet_v3, self).__init__()
        self.patch_size = _patch_size
        self.hidden_dim = _hidden_dim
        self.num_patches = int(_img_size / _patch_size) ** 2

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        self.pos_embed = nn.Parameter(torch.tensor(self.get_positional_embeddings(self.num_patches + 1, self.hidden_dim)))
        self.pos_embed.requires_grad = False

        self.tokenizer = nn.Sequential(
            nn.Linear(self.patch_size ** 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.num_patches),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.encoder_layer = torch.nn.TransformerEncoderLayer(self.hidden_dim, _num_heads, 
                                                              self.hidden_dim * _hidden_dim_mult, 
                                                              dropout=0.25, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, _num_encoder_layer)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(2048, _num_classes),
            nn.Sigmoid()
        )

    def get_positional_embeddings(self, sequence_length, d):
        res = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                res[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return res

    def patchify(self, x, batch_size):
        # TODO type checking
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        x = x.reshape(batch_size, 1, -1, self.patch_size, self.patch_size).flatten(3)
        x = torch.squeeze(x, 1)
        return x
    
    def forward(self, x):
        n, c, h, w = x.shape
        patches = self.patchify(x, n)

        tokens = self.tokenizer(patches)

        # Prepend the class token to the front of each batch.
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embed

        # Run through the encoder blocks.
        out = self.encoder(out)

        # Get the class token (first elm of T in :(B, T, Dim)).
        out = out[:, 0]
        out = self.classifier(out)
        return out
def construct_vit_v3(num_classes, BATCHES_PER_ITR):
    model = VITNet_v3(_patch_size = 16, _num_classes = num_classes, _img_size = 128, 
                      _hidden_dim = 512, _num_heads = 16, _hidden_dim_mult = 4, _num_encoder_layer = 12)
    # V4 Scheduler code
    vit_optimizer = optim.SGD(model.parameters(), lr=0.050, momentum=0.9)
    sched_01 = optim.lr_scheduler.LinearLR(start_factor=0.2, end_factor=1.00, total_iters=7 * BATCHES_PER_ITR)
    sched_02 = optim.lr_scheduler.LinearLR(start_factor=1.0, end_factor=0.04, total_iters=20 * BATCHES_PER_ITR)
    vit_scheduler = optim.lr_scheduler.SequentialLR(vit_optimizer, schedulers=[sched_01, sched_02],
                                                    milestones=[70 * BATCHES_PER_ITR])
    return model, vit_optimizer, vit_scheduler
    
class VITNet_v4(nn.Module):
    def __init__(self, _patch_size, _num_classes, _img_size, _hidden_dim,
                 _num_heads = 8, _hidden_dim_mult = 4, _num_encoder_layer = 2):
        super(VITNet_v4, self).__init__()
        self.patch_size = _patch_size
        self.hidden_dim = _hidden_dim
        self.num_patches = int(_img_size / _patch_size) ** 2

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        self.pos_embed = nn.Parameter(torch.tensor(self.get_positional_embeddings(self.num_patches + 1, self.hidden_dim)))
        self.pos_embed.requires_grad = False

        self.tokenizer = nn.Sequential(
            nn.Linear(self.patch_size ** 2, self.hidden_dim * _hidden_dim_mult),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * _hidden_dim_mult, self.hidden_dim),
            nn.BatchNorm1d(self.num_patches),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.encoder_layer = torch.nn.TransformerEncoderLayer(self.hidden_dim, _num_heads, 
                                                              self.hidden_dim * _hidden_dim_mult, 
                                                              dropout=0.25, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, _num_encoder_layer)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 3072),
            nn.Tanh(),
            nn.Linear(3072, _num_classes),
            nn.Sigmoid()
        )

    def get_positional_embeddings(self, sequence_length, d):
        res = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                res[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return res

    def patchify(self, x, batch_size):
        # TODO type checking
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        x = x.reshape(batch_size, 1, -1, self.patch_size, self.patch_size).flatten(3)
        x = torch.squeeze(x, 1)
        return x
    
    def forward(self, x):
        n, c, h, w = x.shape
        patches = self.patchify(x, n)

        tokens = self.tokenizer(patches)

        # Prepend the class token to the front of each batch.
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embed

        # Run through the encoder blocks.
        out = self.encoder(out)

        # Get the class token (first elm of T in :(B, T, Dim)).
        out = out[:, 0]
        out = self.classifier(out)
        return out
def construct_vit_v4_1(num_classes, BATCHES_PER_ITR):
    model = VITNet_v4(_patch_size = 16, _num_classes = num_classes, _img_size = 128, 
                      _hidden_dim = 640, _num_heads = 16, _hidden_dim_mult = 4, _num_encoder_layer = 12)
    # V4 Scheduler code
    vit_optimizer = optim.SGD(model.parameters(), lr=0.050, momentum=0.9)
    sched_01 = optim.lr_scheduler.LinearLR(vit_optimizer, start_factor=0.20, end_factor=1.00,
                                           total_iters=7 * BATCHES_PER_ITR)
    sched_02 = optim.lr_scheduler.LinearLR(vit_optimizer, start_factor=1.00, end_factor=0.10,
                                           total_iters=10 * BATCHES_PER_ITR)
    sched_03 = optim.lr_scheduler.LinearLR(vit_optimizer, start_factor=0.10, end_factor=0.05,
                                           total_iters=20 * BATCHES_PER_ITR)
    vit_scheduler = optim.lr_scheduler.SequentialLR(vit_optimizer, schedulers=[sched_01, sched_02, sched_03],
                                                    milestones=[80 * BATCHES_PER_ITR, 100 * BATCHES_PER_ITR])
    return model, vit_optimizer, vit_scheduler
def construct_vit_v4_2(num_classes, BATCHES_PER_ITR):
    hidden_dim = 768
    model = VITNet_v4(_patch_size = 16, _num_classes = num_classes, _img_size = 128, 
                      _hidden_dim = hidden_dim, _num_heads = 16, _hidden_dim_mult = 4, _num_encoder_layer = 12)
    model.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 3072),
            nn.Tanh(),
            nn.Linear(3072, num_classes),
            nn.Sigmoid()
        )
    # V4 Scheduler code
    vit_optimizer = optim.SGD(model.parameters(), lr=0.050, momentum=0.9)
    sched_01 = optim.lr_scheduler.LinearLR(vit_optimizer, start_factor=0.20, end_factor=1.00,
                                           total_iters=7 * BATCHES_PER_ITR)
    sched_02 = optim.lr_scheduler.LinearLR(vit_optimizer, start_factor=1.00, end_factor=0.10,
                                           total_iters=10 * BATCHES_PER_ITR)
    sched_03 = optim.lr_scheduler.LinearLR(vit_optimizer, start_factor=0.10, end_factor=0.05,
                                           total_iters=20 * BATCHES_PER_ITR)
    vit_scheduler = optim.lr_scheduler.SequentialLR(vit_optimizer, schedulers=[sched_01, sched_02, sched_03],
                                                    milestones=[80 * BATCHES_PER_ITR, 100 * BATCHES_PER_ITR])
    return model, vit_optimizer, vit_scheduler

def construct_resnet_18(num_classes, BATCHES_PER_ITR):
    model = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    optimizer = optim.SGD(model.parameters(), lr=0.050, momentum=0.9)
    sched_01 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.20, end_factor=1.00,
                                           total_iters=7 * BATCHES_PER_ITR)
    return model, optimizer, sched_01