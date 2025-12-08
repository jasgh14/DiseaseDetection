import numpy as np

class YOLOv8ONNX:

    def __init__(self, onnx_path: str, conf_thr: float = 0.25, iou_thr: float = 0.45,
                 input_size: int = 480, providers=None, debug=False):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path, providers=providers or ["CPUExecutionProvider"])
        self.inp_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name
        self.conf_thr = float(conf_thr)
        self.iou_thr = float(iou_thr)
        self.input_size = int(input_size)
        self.debug = bool(debug)

    def _letterbox(self, img_bgr):
        import cv2
        h, w = img_bgr.shape[:2]
        s = self.input_size
        r = min(s / h, s / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        dw = (s - new_w) // 2
        dh = (s - new_h) // 2
        canvas[dh:dh + new_h, dw:dw + new_w] = resized
        meta = (r, dw, dh, w, h)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[None, ...]  
        return chw, meta

    @staticmethod
    def _xywh2xyxy(xywh):
        x, y, w, h = xywh.T
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _nms(boxes, scores, iou_thr=0.45):
        if len(boxes) == 0:
            return []
        boxes = boxes.astype(np.float32)
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        return keep

    def _postprocess(self, raw):
        pred = np.squeeze(raw)

        if pred.ndim == 1:
            pred = pred[None, :]
        elif pred.ndim == 2:
            H, W = pred.shape
            if H < W and H <= 256 and W >= 1000:
                pred = pred.T  
        elif pred.ndim == 3:
            B, A, C = pred.shape 
            if A <= 256 and C >= 1000:
                pred = pred.transpose(2, 1, 0).reshape(C, A)  
            else:
                pred = pred.reshape(A, C)  
        else:
            pred = pred.reshape(-1, pred.shape[-1])

        if pred.shape[1] < 6:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        xywh = pred[:, 0:4]
        obj  = pred[:, 4:5]
        cls  = pred[:, 5:]

        # apply sigmoid to objectness & class scores
        obj = 1.0 / (1.0 + np.exp(-obj))
        cls = 1.0 / (1.0 + np.exp(-cls))

        conf = obj * cls            
        cls_ids = conf.argmax(axis=1)
        scores  = conf.max(axis=1)
        xyxy = self._xywh2xyxy(xywh)
        return xyxy, scores, cls_ids


    def detect_bgr(self, img_bgr):
        inp, (r, dw, dh, w0, h0) = self._letterbox(img_bgr)
        out = self.session.run([self.out_name], {self.inp_name: inp})[0]
        if self.debug:
            print("onnx out shape:", out.shape)

        xyxy, scores, cls_ids = self._postprocess(out)

        keep = scores >= self.conf_thr
        if not np.any(keep):
            return []
        xyxy = xyxy[keep]; scores = scores[keep]; cls_ids = cls_ids[keep]

        xyxy[:, [0,2]] -= dw
        xyxy[:, [1,3]] -= dh
        xyxy /= r
        # Clip
        xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, w0 - 1)
        xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, h0 - 1)

        keep_idx = self._nms(xyxy, scores, self.iou_thr)
        xyxy = xyxy[keep_idx]
        scores = scores[keep_idx]
        cls_ids = cls_ids[keep_idx]

        boxes = []
        for i in range(len(scores)):
            x1, y1, x2, y2 = xyxy[i].round().astype(int).tolist()
            boxes.append((x1, y1, x2, y2, float(scores[i]), int(cls_ids[i])))
        return boxes
