import argparse, os, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

from .utils.seed import set_seed
from .utils.common import load_config, ensure_dir
from .utils.checkpoint import save_checkpoint
from .data.labels import build_or_load_label_map
from .data.dataset import MultiLabelImageDataset
from .models.factory import build_model
from .metrics import multilabel_metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml")
    ap.add_argument("--model.name", dest="model_name", type=str, default=None)
    ap.add_argument("--train.epochs", dest="epochs", type=int, default=None)
    return ap.parse_args()

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip=None):
    model.train()
    epoch_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        epoch_loss += loss.item() * x.size(0)
    return epoch_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    epoch_loss = 0.0
    y_true_all, y_prob_all = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        epoch_loss += loss.item() * x.size(0)
        y_true_all.append(y.cpu().numpy())
        y_prob_all.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    y_pred = (y_prob >= threshold).astype(np.int32)
    metrics = multilabel_metrics(y_true, y_pred)
    return epoch_loss / len(loader.dataset), metrics

def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.model_name: cfg["model"]["name"] = args.model_name
    if args.epochs: cfg["train"]["epochs"] = args.epochs

    set_seed(cfg["data"]["seed"])
    device = cfg["train"]["device"] if torch.cuda.is_available() and cfg["train"]["device"]=="cuda" else "cpu"

    # labels
    label_map = build_or_load_label_map(cfg["data"]["labels_csv"], cfg["data"]["label_map"])
    num_classes = len(label_map)

    # Prepare datasets
    import pandas as pd
    df = pd.read_csv(cfg["data"]["labels_csv"])
    Y = np.zeros((len(df), num_classes), dtype=int)
    for i, s in enumerate(df["labels"].astype(str)):
        for l in s.split("|"):
            l = l.strip()
            if l and l in label_map:
                Y[i, label_map[l]] = 1

    # Split
    from .data.split import multilabel_train_val_indices
    train_idx, val_idx = multilabel_train_val_indices(cfg["data"]["labels_csv"], Y, cfg["data"]["val_size"], cfg["data"]["seed"])

    train_ds = MultiLabelImageDataset(cfg["data"]["root"], cfg["data"]["labels_csv"], label_map, indices=train_idx, img_size=cfg["data"]["img_size"], train=True)
    val_ds   = MultiLabelImageDataset(cfg["data"]["root"], cfg["data"]["labels_csv"], label_map, indices=val_idx, img_size=cfg["data"]["img_size"], train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,  num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    # Model
    model = build_model(cfg["model"]["name"], num_classes, pretrained=cfg["model"]["pretrained"]).to(device)

    # Criterion
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    if cfg["train"]["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["train"]["lr"], momentum=0.9, weight_decay=cfg["train"]["weight_decay"])

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["train"]["mixed_precision"] and device=="cuda"))

    out_dir = cfg["train"]["out_dir"]
    ensure_dir(out_dir)

    best_micro_f1 = -1.0
    patience = cfg["train"]["early_stop_patience"]
    no_improve = 0

    freeze_epochs = int(cfg["train"]["freeze_backbone_epochs"])
    if freeze_epochs > 0:
        for name, p in model.named_parameters():
            if "fc" in name or "classifier" in name or "heads.head" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if epoch == freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg["train"]["grad_clip"])
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, cfg["inference"]["threshold"])

        micro_f1 = val_metrics["micro_f1"]
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} micro_f1={micro_f1:.4f} macro_f1={val_metrics['macro_f1']:.4f} subset_acc={val_metrics['subset_accuracy']:.4f}")

        is_best = micro_f1 > best_micro_f1
        if is_best:
            best_micro_f1 = micro_f1
            no_improve = 0
        else:
            no_improve += 1

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "config": cfg,
        }
        save_checkpoint(state, out_dir, is_best=is_best, filename="last.pth")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

if __name__ == "__main__":
    main()
