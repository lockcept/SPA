import os
import torch
import torch.nn as nn


class RewardModelBase(nn.Module):
    def __init__(self, config, path):
        super(RewardModelBase, self).__init__()
        self.config = config
        self.path = path

        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement the forward method")

    def evaluate(self, data_loader, loss_fn):
        self.eval()
        epoch_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = [x.to(next(self.parameters()).device) for x in batch]
                outputs = self(*batch)

                loss = loss_fn(*outputs)
                epoch_loss += loss.item()
                num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        return avg_epoch_loss

    def _learn(
        self,
        optimizer,
        train_data_loader,
        val_data_loader,
        loss_fn,
        num_epochs=10,
    ):
        best_loss = float("inf")
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0

            for batch in train_data_loader:
                batch = [x.to(next(self.parameters()).device) for x in batch]
                optimizer.zero_grad()

                # Forward and loss computation
                outputs = self(*batch)
                loss = loss_fn(*outputs)

                # Backward pass and optimizer step
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_data_loader)
            val_loss = self.evaluate(val_data_loader, loss_fn)

            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.state_dict(), self.path)
                print(f"New best model saved with Val loss: {val_loss:.4f}")

    def train_model(self, optimizer, train_loader, val_loader, num_epochs=100):
        loss_fn = nn.MSELoss()
        print(f"[Training started] Saving best model to {self.path}")
        self._learn(optimizer, train_loader, val_loader, loss_fn, num_epochs)
        print("Training completed")
