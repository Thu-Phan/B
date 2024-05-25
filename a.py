import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from unet import UNet
from hrnet import HRNet
from d import MammographyDataset, ToTensor
from torchvision import transforms

def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
    return model

if __name__ == '__main__':
    csv_file = '/media/hthieu/Data_New_6TB/breastcancer/cbis-ddsm/processed_reorganized/Calc-Training_mask_processed_set.csv'
    root_dir = '/media/hthieu/Data_New_6TB/breastcancer/cbis-ddsm/processed_reorganized/calc/training'
    
    transform = transforms.Compose([
        ToTensor()
    ])
    
    dataset = MammographyDataset(csv_file, root_dir, transform)
    
    # Enable quick test mode
    quick_test = False
    
    if quick_test:
        print("Quick test mode enabled")
        dataset = Subset(dataset, range(20))  # Use only a small subset of data for testing
        num_epochs = 2  # Use fewer epochs for quick testing
    else:
        num_epochs = 25  # Original number of epochs for full training
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize models
    print("Initializing models...")
    unet = UNet(in_channels=1, out_channels=1).to(device)
    hrnet = HRNet(in_channels=1, out_channels=1).to(device)
    print("Models initialized.")

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    unet_optimizer = optim.Adam(unet.parameters(), lr=0.001)
    hrnet_optimizer = optim.Adam(hrnet.parameters(), lr=0.001)

    

    # Train HRNet
    print("Training HRNet")
    hrnet = train_model(hrnet, dataloader, criterion, hrnet_optimizer, num_epochs=num_epochs)
    torch.save(hrnet.state_dict(), 'hrnet.pth')  # Save HRNet model

    # Train UNet
    print("Training UNet")
    unet = train_model(unet, dataloader, criterion, unet_optimizer, num_epochs=num_epochs)
    torch.save(unet.state_dict(), 'unet.pth')
    print("Training complete.")

