import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import json
import ast
try:
    import onnxruntime as ort
except Exception as e:
    ort = None
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    Image = None
    ImageDraw = None
    ImageFont = None


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    score: float
    cls: int
def _letterbox(img: np.ndarray, new_shape: Tuple[int, int], color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw //= 2
    dh //= 2
    if (w, h) != new_unpad:
        img_resized = np.array(Image.fromarray(img).resize(new_unpad, Image.BILINEAR)) if Image else img
    else:
        img_resized = img
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    canvas[dh:dh + img_resized.shape[0], dw:dw + img_resized.shape[1]] = img_resized
    return canvas, r, (dw, dh)
def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep
class ONNXYoloDetector:
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        # 初始化 ONNX 识别会话，如果缺少 onnxruntime 会抛出异常
        if ort is None:
            raise RuntimeError('需要 onnxruntime 才能运行 ONNX 识别。请先安装: pip install onnxruntime')
        self.model_path = os.path.abspath(model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f'模型文件不存在: {self.model_path}')
        self.providers = providers or ['CPUExecutionProvider']
        # 配置 SessionOptions，限制线程数并开启优化
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 创建会话时提供更友好的错误信息，便于定位常见问题（如损坏或非 ONNX 文件）
        try:
            self.session = ort.InferenceSession(self.model_path, sess_options=so, providers=self.providers)
        except Exception as e:
            # 常见错误：INVALID_PROTOBUF -> 文件损坏/不是 ONNX；INVALID_GRAPH -> 导出/opset 问题
            hint = (
                f"加载 ONNX 模型失败: {e}.\n"
                f"请检查: 1) 路径是否正确 2) 文件是否为有效的 .onnx 而非 .pt/.pth 被改名 3) 文件是否完整未损坏。\n"
                f"可使用脚本 Code/scripts/validate_onnx.py 验证或重新导出模型。"
            )
            raise RuntimeError(hint)
        self.input_name = self.session.get_inputs()[0].name
        in_shape = self.session.get_inputs()[0].shape
        # 尝试从输入 shape 中读取 H,W，否则回退到 640
        if isinstance(in_shape[2], int) and isinstance(in_shape[3], int):
            self.in_h, self.in_w = int(in_shape[2]), int(in_shape[3])
        else:
            self.in_h, self.in_w = 640, 640
        # 尝试从模型的 custom metadata 中解析 class names（如果模型内包含）
        self.class_names: Optional[List[str]] = None
        try:
            meta = self.session.get_modelmeta()
            mm = getattr(meta, 'custom_metadata_map', None)
            if isinstance(mm, dict):
                names_raw = mm.get('names') or mm.get('classes') or mm.get('labels') or None
                if names_raw:
                    obj = None
                    try:
                        obj = json.loads(names_raw)
                    except Exception:
                        try:
                            obj = ast.literal_eval(names_raw)
                        except Exception:
                            obj = None
                    # 支持字典或列表两种存储方式
                    if isinstance(obj, dict):
                        try:
                            items = sorted(((int(k), v) for k, v in obj.items()), key=lambda x: x[0])
                            if items:
                                max_k = items[-1][0]
                                names_list = [None] * (max_k + 1)
                                for k, v in items:
                                    if 0 <= k <= max_k:
                                        names_list[k] = str(v)
                                self.class_names = [n if n is not None else str(i) for i, n in enumerate(names_list)]
                        except Exception:
                            self.class_names = [str(v) for _, v in items]
                    elif isinstance(obj, list):
                        self.class_names = [str(x) for x in obj]
        except Exception:
            pass
    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        # 将输入图像转换为模型输入：resize + pad（letterbox），并返回 tensor、缩放比例和偏移
        img0 = img if img.dtype == np.uint8 else img.astype(np.uint8)
        if Image is None:
            # Pillow 缺失时使用 numpy 基本实现
            h, w = img0.shape[:2]
            scale = min(self.in_h / h, self.in_w / w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            resized = np.zeros((self.in_h, self.in_w, 3), dtype=np.uint8)
            ys, xs = (self.in_h - nh) // 2, (self.in_w - nw) // 2
            resized[ys:ys+nh, xs:xs+nw] = img0[:h, :w]
            r, pad = scale, (xs, ys)
        else:
            resized, r, pad = _letterbox(img0, (self.in_h, self.in_w))
        x = resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return x, r, pad
    def _postprocess(self, pred: np.ndarray, img_shape: Tuple[int, int], r: float, pad: Tuple[int, int], conf_thres: float, iou_thres: float) -> List[Detection]:
        if pred.ndim == 3:
            pred = pred[0]
        def deletterbox_xyxy(boxes_xyxy: np.ndarray) -> np.ndarray:
            pw, ph = pad
            gain = r
            boxes_xyxy = boxes_xyxy.copy()
            boxes_xyxy[:, [0, 2]] -= pw
            boxes_xyxy[:, [1, 3]] -= ph
            boxes_xyxy /= gain
            img_h, img_w = img_shape
            boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, img_w)
            boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, img_h)
            return boxes_xyxy
        dets: List[Detection] = []
        # 处理形状为 (N,6) 或 (6,N) 的简单识别格式：x1,y1,x2,y2,score,cls
        if pred.ndim == 2 and 6 in pred.shape:
            if pred.shape[-1] != 6:
                pred = pred.T
            if pred.shape[-1] == 6:
                boxes_xyxy = pred[:, :4].astype(np.float32)
                scores = pred[:, 4].astype(np.float32)
                cls_ids = pred[:, 5].astype(np.int32)
                mask = scores >= conf_thres
                boxes_xyxy = boxes_xyxy[mask]
                scores = scores[mask]
                cls_ids = cls_ids[mask]
                if boxes_xyxy.size == 0:
                    return []
                boxes_xyxy = deletterbox_xyxy(boxes_xyxy)
                keep = _nms(boxes_xyxy, scores, iou_thres)
                for i in keep:
                    x1, y1, x2, y2 = boxes_xyxy[i].tolist()
                    dets.append(Detection((float(x1),
                                           float(y1),
                                           float(x2),
                                           float(y2)),
                                           float(scores[i]),
                                           int(cls_ids[i])))
                return dets
        # 如果输出为 (6,N) 等转置情况，调整为行优先
        if pred.shape[0] <= pred.shape[1]:
            pred = pred.T
        # 处理常见的 xywh + logits/class-scores 输出
        if pred.shape[1] >= 6:
            boxes = pred[:, :4].astype(np.float32)
            scores_cls = pred[:, 4:].astype(np.float32)
            scores_cls = 1.0 / (1.0 + np.exp(-scores_cls))
            scores = scores_cls.max(axis=1)
            cls_ids = scores_cls.argmax(axis=1).astype(np.int32)
            mask = scores >= conf_thres
            boxes = boxes[mask]
            scores = scores[mask]
            cls_ids = cls_ids[mask]
            if boxes.size == 0:
                return []
            x, y, w, h = boxes.T
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
            boxes_xyxy = deletterbox_xyxy(boxes_xyxy)
            keep = _nms(boxes_xyxy, scores, iou_thres)
            for i in keep:
                x1, y1, x2, y2 = boxes_xyxy[i].tolist()
                dets.append(Detection((float(x1), float(y1), float(x2), float(y2)), float(scores[i]), int(cls_ids[i])))
            return dets
        return []
    def predict(self, img: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> List[Detection]:
        # 对单张图像执行识别，返回 Detection 列表
        x, r, pad = self._preprocess(img)
        out_names = [o.name for o in self.session.get_outputs()]
        ort_outs = self.session.run(out_names, {self.input_name: x})
        pred = None
        # 尝试找到形状合理的输出（最后一维 >=6）
        for o in ort_outs:
            if isinstance(o, np.ndarray) and o.ndim in (2, 3) and o.shape[-1] >= 6:
                pred = o
                break
        # 若未找到则回退到取第一个 numpy 输出
        if pred is None:
            for o in ort_outs:
                if isinstance(o, np.ndarray):
                    pred = o
                    break
        if pred is None:
            raise RuntimeError('ONNX 模型输出不包含可解析的检测结果')
        return self._postprocess(pred, (img.shape[0], img.shape[1]), r, pad, conf, iou)
    def annotate(self, img: np.ndarray, dets: List[Detection], class_names: Optional[List[str]] = None) -> np.ndarray:
        if class_names is None:
            class_names = self.class_names
        arr = img.copy()
        # 如果 Pillow 不可用，使用简单的 numpy 绘制边框（线宽固定）
        if Image is None or ImageDraw is None:
            for d in dets:
                x1, y1, x2, y2 = map(int, d.xyxy)
                arr[y1:y1+2, x1:x2] = (0, 255, 0)
                arr[y2-2:y2, x1:x2] = (0, 255, 0)
                arr[y1:y2, x1:x1+2] = (0, 255, 0)
                arr[y1:y2, x2-2:x2] = (0, 255, 0)
            return arr
        im = Image.fromarray(arr)
        draw = ImageDraw.Draw(im)
        font = None  
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for d in dets:
            x1, y1, x2, y2 = d.xyxy
            xi, yi, xa, ya = map(int, [x1, y1, x2, y2])
            label = class_names[d.cls] if class_names and 0 <= d.cls < len(class_names) else str(d.cls)
            txt = f"{label} {d.score:.2f}"
            draw.rectangle([xi, yi, xa, ya], outline=(0, 255, 0), width=2)
            try:
                bbox = draw.textbbox((xi, yi), txt, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = (int(8 * len(txt)), 12)
            by1 = max(0, yi - th - 2)
            bx2 = xi + tw + 2
            draw.rectangle([xi, by1, bx2, yi], fill=(0, 255, 0))
            draw.text((xi + 1, max(0, yi - th - 1)), txt, fill=(0, 0, 0), font=font)
        return np.array(im)
    def predict_and_save(self, in_path: str, out_path: str, conf: float = 0.25, iou: float = 0.45, class_names: Optional[List[str]] = None) -> list:
        # 从路径读取图片，执行识别并将带可视化的结果保存到 out_path，返回检测的标签列表
        if Image is None:
            raise RuntimeError('需要 Pillow 才能保存可视化结果。请安装: pip install pillow')
        img = np.array(Image.open(in_path).convert('RGB'))
        dets = self.predict(img, conf=conf, iou=iou)
        vis = self.annotate(img, dets, class_names=class_names)
        Image.fromarray(vis).save(out_path)
        labels = []
        for d in dets:
            try:
                if class_names is None:
                    name = self.class_names[d.cls] if self.class_names and 0 <= d.cls < len(self.class_names) else str(d.cls)
                else:
                    name = class_names[d.cls] if class_names and 0 <= d.cls < len(class_names) else str(d.cls)
            except Exception:
                name = str(d.cls)
            labels.append(str(name))
        return labels
