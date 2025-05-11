# evaluate.py
import torch
from model import CNNModel
from utils import get_dataloaders, load_model
import config

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel().to(device)
    model = load_model(model, config.MODEL_PATH)
    model.eval()

    _, test_loader = get_dataloaders(config.BATCH_SIZE)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
