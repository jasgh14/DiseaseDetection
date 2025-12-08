import argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils.common import load_config
from .utils.checkpoint import load_weights
from .data.labels import build_or_load_label_map
from .data.dataset import MultiLabelImageDataset
from .models.factory import build_model
from .metrics import multilabel_metrics
import torch.nn as nn

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    return ap.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = cfg["train"]["device"] if torch.cuda.is_available() and cfg["train"]["device"]=="cuda" else "cpu"

    label_map = build_or_load_label_map(cfg["data"]["labels_csv"], cfg["data"]["label_map"])
    num_classes = len(label_map)

    ds = MultiLabelImageDataset(cfg["data"]["root"], cfg["data"]["labels_csv"], label_map, indices=None, img_size=cfg["data"]["img_size"], train=False)
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    model = build_model(cfg["model"]["name"], num_classes, pretrained=False).to(device)
    ckpt = load_weights(model, args.checkpoint, device=device)
    criterion = nn.BCEWithLogitsLoss()

    y_true_all, y_prob_all = [], []
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        y_true_all.append(y.cpu().numpy())
        y_prob_all.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    y_pred = (y_prob >= cfg["inference"]["threshold"]).astype(int)
    metrics = multilabel_metrics(y_true, y_pred)
    avg_loss = loss_sum / len(ds)

    print(json.dumps({"loss": avg_loss, **metrics}, indent=2))

if __name__ == "__main__":
    main()
