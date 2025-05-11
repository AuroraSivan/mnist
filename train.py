# train.py
import torch
import torch.nn as nn
from torch import optim
from utils import get_dataloaders, save_model
from model import CNNModel
import config

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel().to(device)
    train_loader, _ = get_dataloaders(config.BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    save_model(model, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")
