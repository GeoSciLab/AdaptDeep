import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    This class defines the basic structure of a residual block.
    Each residual block consists of two convolution layers, each followed by a ReLU activation function.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class CropReconstruction(nn.Module):
    """
    This class defines the overall network architecture for cropped field reconstruction.
    The architecture consists of an input layer, a series of residual blocks, and an output layer.
    """
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CropReconstruction, self).__init__()
        self.input_layer = self._make_input_layer(in_channels, out_channels)
        self.res_blocks = self._make_layer(ResidualBlock, out_channels, num_blocks)
        self.output_layer = self._make_output_layer(out_channels, in_channels)

    def _make_input_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return layer

    def _make_output_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return layer

    def _make_layer(self, block, in_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(in_channels, in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        res_out = self.res_blocks(x)
        out = self.output_layer(res_out)
        return out

# Training the model
def train(model, dataloader, criterion, optimizer):
    model.train()
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model

# Evaluating the model
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Implementing the model training and evaluation process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CropReconstruction(in_channels=3, out_channels=64, num_blocks=4).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Assuming we have dataloaders 'train_dataloader' and 'val_dataloader'
num_epochs = 10
for epoch in range(num_epochs):
    model = train(model, train_dataloader, criterion, optimizer)
    train_loss = evaluate(model, train_dataloader, criterion)
    val_loss = evaluate(model, val_dataloader, criterion)
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
