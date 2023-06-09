import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F

class ContrastiveLearningViewGenerator(object):
    """
    Implementing a view generation module for contrastive learning.
    """
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class ResNet(nn.Module):
    """
    Implementing a simple ResNet model for encoding images.
    """
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False)

        # change the last layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 128)

    def forward(self, x):
        return self.resnet(x)

class ContrastiveLoss(nn.Module):
    """
    Implementing a contrastive loss (InfoNCE loss)
    """
    def __init__(self, temperature):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, features):
        batch_size = features.shape[0]
        mask = torch.eye(batch_size*2).to(features.device)

        features = F.normalize(features, dim=1)

        logits_aa = torch.matmul(features[:batch_size], features[:batch_size].T) / self.temperature
        logits_aa = logits_aa - mask * 1e9
        logits_bb = torch.matmul(features[batch_size:], features[batch_size:].T) / self.temperature
        logits_bb = logits_bb - mask * 1e9
        logits_ab = torch.matmul(features[:batch_size], features[batch_size:].T) / self.temperature
        logits_ba = torch.matmul(features[batch_size:], features[:batch_size].T) / self.temperature

        loss_a = self.criterion(logits_ab, torch.arange(batch_size).to(features.device))
        loss_b = self.criterion(logits_ba, torch.arange(batch_size).to(features.device))

        loss = loss_a + loss_b

        return loss

# Implementing the contrastive learning training process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming we have dataloaders 'train_dataloader' and 'val_dataloader'
model = ResNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = ContrastiveLoss(temperature=0.5)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, _) in enumerate(train_dataloader):
        optimizer.zero_grad()

        images = torch.cat(images, dim=0)
        images = images.to(device)

        features = model(images)
        loss = criterion(features)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch: {epoch+1}, Loss: {total_loss/i}")
