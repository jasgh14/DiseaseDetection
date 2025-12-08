import os, torch

def save_checkpoint(state: dict, out_dir: str, is_best: bool = False, filename: str = "last.pth"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(out_dir, "best_model.pth")
        torch.save(state, best_path)

def load_weights(model, checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict, strict=False)
    return ckpt
