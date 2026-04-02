import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from datasets import load_dataset
from model_vae import VAE
from model_damage import LatentVariableClassifier


def train_damage(
    vae_weights_path="checkpoints/vae_epoch_150.pth",
    epochs=50,
    batch_size=32,
    lr=1e-4,
    latent_dim=256,
    num_classes=12,
    beta=0.1
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # =========================
    # LOAD VAE (FROZEN)
    # =========================
    vae = VAE(latent_dim=latent_dim).to(device)

    if os.path.exists(vae_weights_path):
        vae.load_state_dict(torch.load(vae_weights_path, map_location=device))
        print("Loaded VAE weights")
    else:
        print("WARNING: VAE weights not found")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # =========================
    # CLASSIFIER
    # =========================
    classifier = LatentVariableClassifier(
        z_dim=latent_dim,
        h_dim=64,
        num_classes=num_classes
    ).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # =========================
    # DATA
    # =========================
    train_dataset = load_dataset("MITLL/LADI-v2-dataset", split="train")
    val_dataset = load_dataset("MITLL/LADI-v2-dataset", split="validation")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    best_f1 = 0.0

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(1, epochs + 1):
        classifier.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (Train)")

        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            # Extract latent features (μ)
            with torch.no_grad():
                mu, _ = vae.encoder(data)

            logits, mu_h, logvar_h = classifier(mu)

            loss, bce_loss, kl_loss = classifier.compute_loss(
                logits, labels.float(), mu_h, logvar_h, beta
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # =========================
        # VALIDATION
        # =========================
        classifier.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)

                mu, _ = vae.encoder(data)

                logits, mu_h, logvar_h = classifier(mu)

                loss, _, _ = classifier.compute_loss(
                    logits, labels.float(), mu_h, logvar_h, beta
                )
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        print(
            f"Epoch {epoch}: "
            f"Train Loss {avg_train_loss:.4f} | "
            f"Val Loss {avg_val_loss:.4f} | "
            f"Val Acc {val_accuracy:.4f} | "
            f"Val F1 {val_f1:.4f}"
        )

        # =========================
        # SAVE BEST MODEL
        # =========================
        os.makedirs("checkpoints", exist_ok=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(classifier.state_dict(), "checkpoints/damage_lvm_best.pth")
            print(f"[*] Saved new best model (F1={best_f1:.4f})")

    print("Training complete.")

if __name__ == "__main__":
    train_damage()
