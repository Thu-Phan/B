import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from unet import UNet
from hrnet import HRNet
from d import MammographyDataset, ToTensor
from torchvision import transforms

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    
    total_loss = running_loss / len(dataloader.dataset)
    return total_loss

if __name__ == '__main__':
    csv_file = '/media/hthieu/Data_New_6TB/breastcancer/cbis-ddsm/processed_reorganized/Calc-Training_mask_processed_set.csv'
    root_dir = '/media/hthieu/Data_New_6TB/breastcancer/cbis-ddsm/processed_reorganized/calc/training'
    
    transform = transforms.Compose([
        ToTensor()
    ])
    
    dataset = MammographyDataset(csv_file, root_dir, transform)
    
    # Enable quick test mode
    quick_test = True
    
    if quick_test:
        print("Quick test mode enabled")
        dataset = Subset(dataset, range(20))  # Use only a small subset of data for testing
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize models
    print("Loading models...")
    hrnet = HRNet(in_channels=1, out_channels=1).to(device)
    unet = UNet(in_channels=1, out_channels=1).to(device)
    
    # Load the trained weights
    hrnet.load_state_dict(torch.load('hrnet.pth'))
    unet.load_state_dict(torch.load('unet.pth'))
    print("Models loaded.")

    # Define loss function
    criterion = nn.BCEWithLogitsLoss()


    # Evaluate HRNet
    print("Evaluating HRNet...")
    hrnet_loss = evaluate_model(hrnet, dataloader, criterion, device)
    print(f'HRNet Loss: {hrnet_loss:.4f}')
    # Evaluate UNet
    print("Evaluating UNet...")
    unet_loss = evaluate_model(unet, dataloader, criterion, device)
    print(f'UNet Loss: {unet_loss:.4f}')

    # Summary comparison
    print("Summary Comparison:")
    print(f'HRNet Loss: {hrnet_loss:.4f}')
    print(f'UNet Loss: {unet_loss:.4f}')

