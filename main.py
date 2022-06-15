from AlexNet import AlexNet
from Inception import Inception
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn.functional as F
from torchsummary import summary

if __name__ == "__main__":

    device = "cuda:0"
    n_epoch = 10
    n_classes = 100

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize((224, 224))])
    train_dataset = torchvision.datasets.CIFAR100("..//data", train=True, transform=transform, download=True)
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = Inception(n_classes).to(device)

    print(summary(model, (3, 224, 224)))

    optim = SGD(model.parameters(), lr=0.01)

    for epoch in range(n_epoch):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = F.cross_entropy(outputs, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"Epoch{epoch+1}/{n_epoch}: loss {loss.item()}")