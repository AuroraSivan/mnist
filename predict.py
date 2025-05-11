# predict.py
import torch
from torchvision import datasets, transforms
from model import CNNModel
from utils import load_model
import config
import matplotlib.pyplot as plt

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel().to(device)
    model = load_model(model, config.MODEL_PATH)
    model.eval()

    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    image, label = test_data[0]

    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        pred = torch.argmax(output, dim=1).item()

    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Predicted: {pred} | Actual: {label}")
    plt.show()
