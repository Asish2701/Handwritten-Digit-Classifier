"""Train a small convolutional neural network on MNIST."""

from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "model.pth"
BATCH_SIZE = 128
EPOCHS = 9
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DigitCNN(nn.Module):
    """A compact CNN that maps a 28x28 grayscale image to 10 class logits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits; CrossEntropyLoss applies the stable softmax internally."""
        return self.classifier(self.features(images))


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> int:
    """Return the number of correct predictions in a batch."""
    return (logits.argmax(dim=1) == labels).sum().item()


def evaluate(
    model: nn.Module, data_loader: DataLoader, loss_function: nn.Module
) -> tuple[float, float]:
    """Calculate average loss and accuracy without tracking gradients."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            total_loss += loss_function(logits, labels).item() * labels.size(0)
            correct += accuracy_from_logits(logits, labels)
            total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    """Download MNIST, train the CNN, report metrics, and save its weights."""
    torch.manual_seed(42)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    full_train_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(42),
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DigitCNN().to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {DEVICE} for {EPOCHS} epochs")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += accuracy_from_logits(logits, labels)
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_function
        )
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.2%} | "
            f"Validation loss: {validation_loss:.4f}, "
            f"accuracy: {validation_accuracy:.2%}"
        )

    test_loss, test_accuracy = evaluate(model, test_loader, loss_function)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Test loss: {test_loss:.4f}, accuracy: {test_accuracy:.2%}")
    print(f"Saved model weights to {MODEL_PATH}")


if __name__ == "__main__":
    main()
