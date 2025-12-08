import os, json, time
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .models.factory import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _eval_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class InferenceEngine:
    def __init__(
        self,
        weights: str,
        label_map: Union[str, Dict[str, int]],
        model_name: str = "efficientnet_b0",
        device: str = "cpu",
        img_size: int = 224,
        threshold: float = 0.5,      # used in multilabel mode
        temperature: float = 1.0,    # used in multiclass mode
        multiclass: bool = True,
        unknown_maxp: float = 0.55,
        unknown_entropy: float = 1.5,
        save_uncertain_dir: str = "feedback/needs_label",
        onnx_path: Optional[str] = None,   
    ):
        self.img_size = int(img_size)
        self.tfms = _eval_transforms(self.img_size)

        self.multiclass = bool(multiclass)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.unknown_maxp = float(unknown_maxp)
        self.unknown_entropy = float(unknown_entropy)
        self.save_uncertain_dir = save_uncertain_dir

        # label map
        if isinstance(label_map, str):
            with open(label_map, "r", encoding="utf-8") as f:
                self.label_to_idx: Dict[str, int] = json.load(f)
        else:
            self.label_to_idx = label_map
        self.idx_to_label: List[str] = [None] * len(self.label_to_idx)
        for k, v in self.label_to_idx.items():
            self.idx_to_label[v] = k
        self.num_classes = len(self.idx_to_label)

        # device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        # model
        self.model = build_model(model_name, self.num_classes, pretrained=False).to(self.device)
        ckpt = torch.load(weights, map_location=self.device)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # try to read temperature from calibration.json
        try:
            calib_path = os.path.join(os.path.dirname(weights), "calibration.json")
            if os.path.exists(calib_path) and self.multiclass:
                with open(calib_path, "r", encoding="utf-8") as f:
                    j = json.load(f)
                if "temperature" in j:
                    self.temperature = float(j["temperature"])
        except Exception:
            pass


        self.use_onnx = False
        if onnx_path:
            try:
                import onnxruntime as ort
                self.ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                self.use_onnx = True
            except Exception:
                self.use_onnx = False

    # APIS
    @torch.no_grad()
    def predict_path(self, image_path: str):
        pil = Image.open(image_path).convert("RGB")
        return self._predict_pil_core(pil)

    @torch.no_grad()
    def predict_pil(self, pil_img: Image.Image):
        return self._predict_pil_core(pil_img.convert("RGB"))

    @torch.no_grad()
    def predict_ndarray(self, arr_bgr):
        import cv2
        rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self._predict_pil_core(pil)

    # Grad-CAM
    def grad_cam(self, pil_img: Image.Image, target_class: Optional[int] = None) -> Optional[np.ndarray]:

        try:
            x = self.tfms(pil_img.convert("RGB")).unsqueeze(0).to(self.device)

            # find last conv2d
            last_conv = None
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    last_conv = m
            if last_conv is None:
                return None 

            fmap, grads = [], []

            def fwd_hook(_, __, out):
                fmap.append(out.detach())

            def bwd_full_hook(module, grad_input, grad_output):
                grads.append(grad_output[0].detach())

            # hooks
            h1 = last_conv.register_forward_hook(fwd_hook)
            if hasattr(last_conv, "register_full_backward_hook"):
                h2 = last_conv.register_full_backward_hook(bwd_full_hook)
            else:
                # fallback
                h2 = last_conv.register_backward_hook(lambda m, gi, go: grads.append(go[0].detach()))

            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())
            score = logits[0, target_class]
            score.backward()

            fm = fmap[-1][0]                        
            gd = grads[-1][0] if grads[-1].ndim == 4 else grads[-1]
            weights = gd.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * fm).sum(dim=0))

            cam_np = cam.detach().cpu().numpy()
            cam_np = cam_np - cam_np.min()
            denom = cam_np.max() if cam_np.max() > 0 else 1.0
            cam_np = cam_np / denom

            import cv2
            w, h = pil_img.size
            cam_np = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)
            heat = (cam_np * 255.0).astype(np.uint8).copy()

            h1.remove(); h2.remove()
            return heat
        except Exception:
            return None


    @staticmethod
    def severity_from_cam(heat_uint8: np.ndarray, hi: int = 200) -> Tuple[str, float]:
        mask = (heat_uint8 >= int(hi)).astype(np.uint8)
        cov = float(mask.mean())
        if cov < 0.05: lvl = "Low"
        elif cov < 0.15: lvl = "Medium"
        else: lvl = "High"
        return lvl, cov

    # helpers
    def _predict_pil_core(self, pil_img: Image.Image):
        x = self.tfms(pil_img).unsqueeze(0)
        if self.use_onnx:
            x_np = x.numpy()
            logits_np = self.ort_sess.run(["logits"], {"images": x_np})[0]
            logits = torch.from_numpy(logits_np)
        else:
            x = x.to(self.device)
            logits = self.model(x).detach().cpu()

        results, selected, meta = self._postprocess(logits)
        if meta.get("unknown", False):
            self._maybe_save_uncertain(pil_img, results, meta)
        return results, selected, meta

    def _postprocess(self, logits: torch.Tensor):

        if self.multiclass:
            p = F.softmax(logits / self.temperature, dim=1).numpy()[0]  
            top_idx = int(np.argmax(p))
            maxp = float(p[top_idx])
            ent = float(-(p * np.log(p + 1e-12)).sum())
            unknown = (maxp < self.unknown_maxp) or (ent > self.unknown_entropy)
            results = {self.idx_to_label[i]: float(p[i]) for i in range(self.num_classes)}
            selected = [self.idx_to_label[top_idx]]
            meta = {"max_p": maxp, "entropy": ent, "unknown": unknown, "mode": "multiclass"}
            return results, selected, meta
        else:
            p = torch.sigmoid(logits).numpy()[0]
            results = {self.idx_to_label[i]: float(p[i]) for i in range(self.num_classes)}
            selected = [self.idx_to_label[i] for i in range(self.num_classes) if p[i] >= self.threshold]
            maxp = float(p.max())
            meta = {"max_p": maxp, "unknown": False, "mode": "multilabel"}
            return results, selected, meta

    def _maybe_save_uncertain(self, pil_img: Image.Image, results: Dict[str, float], meta: Dict):
        try:
            os.makedirs(self.save_uncertain_dir, exist_ok=True)
            ts = str(int(time.time()))
            img_path = os.path.join(self.save_uncertain_dir, f"{ts}.jpg")
            json_path = os.path.join(self.save_uncertain_dir, f"{ts}.json")
            pil_img.save(img_path)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"results": results, "meta": meta}, f, indent=2)
        except Exception:
            pass
