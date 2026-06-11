import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.class_labels import save_class_labels
from src.dataset.char_dataset import get_char_datasets
from src.models.vision_encoder import DeiTClassifier
from src.utils.device import configure_gpu
from src.utils.plotting import plot_stage1_metrics, save_history


def train_stage1():
    # Configuration — tuned for RTX 5090 (32 GB VRAM)
    data_dir = "Datasets/Dataset454"
    batch_size = 64
    grad_accum_steps = 2  # effective batch size 128 with lower VRAM per step
    num_epochs = 30
    learning_rate = 1e-4
    output_dir = "outputs/stage1"

    device, use_amp = configure_gpu()

    print("Loading datasets...")
    train_loader, valid_loader, test_loader, classes = get_char_datasets(
        data_dir=data_dir,
        batch_size=batch_size,
    )
    num_classes = len(classes)
    print(f"Found {num_classes} classes.")
    labels_path = save_class_labels(classes, os.path.join(output_dir, "class_labels.json"))
    print(f"Saved class label map: {labels_path}")

    print("Initializing model...")
    model = DeiTClassifier(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    best_val_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(tqdm(train_loader, desc="Training"), start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast(device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels) / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels) / grad_accum_steps
                loss.backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * grad_accum_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="Validation"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast(device.type):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(valid_loader)
        val_acc = 100 * val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving best model...")
            torch.save(model.state_dict(), os.path.join(output_dir, "best_classifier.pth"))
            torch.save(model.encoder.state_dict(), os.path.join(output_dir, "pretrained_encoder.pth"))

        save_history(history, output_dir, "training_history.json")
        combined, loss_plot, acc_plot = plot_stage1_metrics(history, output_dir)
        print(f"Saved graphs: {combined}, {loss_plot}, {acc_plot}")


if __name__ == "__main__":
    train_stage1()
