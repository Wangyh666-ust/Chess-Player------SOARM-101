import numpy as np
import torch

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def detections_from_result(result, names):
    boxes_obj = result.boxes
    xyxy = _as_numpy(boxes_obj.xyxy)
    cls = _as_numpy(boxes_obj.cls)
    conf = _as_numpy(boxes_obj.conf)

    xyxy = np.asarray(xyxy, dtype=float)
    if xyxy.size == 0:
        xyxy = xyxy.reshape(0, 4)
    if xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ValueError("boxes must have shape (N,4), got %r" % (xyxy.shape,))

    cls_flat = np.asarray(cls).reshape(-1)
    conf_flat = np.asarray(conf, dtype=float).reshape(-1)

    n = xyxy.shape[0]
    if cls_flat.shape[0] != n:
        raise ValueError(
            "length mismatch: boxes=%d classes=%d" % (n, cls_flat.shape[0])
        )
    if conf_flat.shape[0] != n:
        raise ValueError(
            "length mismatch: boxes=%d conf=%d" % (n, conf_flat.shape[0])
        )
    if not np.all(np.isfinite(xyxy)):
        raise ValueError("non-finite box coordinate")
    if not np.all(np.isfinite(conf_flat)):
        raise ValueError("non-finite confidence")

    out = []
    for i in range(n):
        cid = cls_flat[i]
        try:
            cid_int = int(cid)
        except Exception:
            raise ValueError("non-integer class id: %r" % (cid,))
        if float(cid_int) != float(cid):
            raise ValueError("non-integer class id: %r" % (cid,))
        if cid_int not in names:
            raise ValueError("class id %d not present in names" % cid_int)
        box = [float(v) for v in xyxy[i]]
        out.append(
            {
                "box": box,
                "class_id": cid_int,
                "class_name": str(names[cid_int]),
                "confidence": float(conf_flat[i]),
            }
        )
    return out


class FullScorePredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs, **kwargs):
        if getattr(self, "_feats", None) is not None:
            raise ValueError("FullScorePredictor does not support object features")
        if getattr(self.args, "task", "detect") != "detect":
            raise ValueError(
                "FullScorePredictor only supports detection task, got %r"
                % (getattr(self.args, "task", None),)
            )
        if getattr(self.args, "augment", False):
            raise ValueError("FullScorePredictor does not support augmentation")
        if getattr(self.model, "end2end", False):
            raise ValueError("FullScorePredictor does not support end2end models")

        if isinstance(preds, (tuple, list)):
            if len(preds) == 0:
                raise ValueError("preds sequence is empty")
            preds = preds[0]
        if not isinstance(preds, torch.Tensor):
            raise TypeError("preds must be a torch.Tensor, got %r" % (type(preds),))
        if preds.ndim != 3:
            raise ValueError("preds must have shape (B,4+nc,N), got %r" % (tuple(preds.shape),))

        num_classes = len(self.model.names)
        batch = preds.shape[0]
        channels = preds.shape[1]
        num_candidates = preds.shape[2]
        if channels != 4 + num_classes:
            raise ValueError(
                "preds must have (B,4+nc,N) with nc=%d -> %d channels, got %d"
                % (num_classes, 4 + num_classes, channels)
            )

        score_src = preds[:, 4:4 + num_classes, :].clone().detach().cpu()

        nms_preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            kwargs.pop("iou", self.args.iou),
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0,
            end2end=False,
            rotated=False,
            return_idxs=True,
        )
        if not isinstance(nms_preds, (tuple, list)) or len(nms_preds) != 2:
            raise RuntimeError("NMS did not return (detections, indices) pair")
        nms_boxes, nms_idxs = nms_preds
        if len(nms_boxes) != batch or len(nms_idxs) != batch:
            raise RuntimeError(
                "NMS batch mismatch: preds=%d dets=%d idxs=%d"
                % (batch, len(nms_boxes), len(nms_idxs))
            )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = self.construct_results(nms_boxes, img, orig_imgs, **kwargs)
        if len(results) != batch:
            raise RuntimeError(
                "construct_results returned %d results for batch %d"
                % (len(results), batch)
            )

        for b, result in enumerate(results):
            dets = nms_boxes[b]
            idxs = nms_idxs[b]
            if not isinstance(dets, torch.Tensor):
                dets = torch.as_tensor(dets)
            num_dets = int(dets.shape[0])

            idx_arr = torch.as_tensor(idxs).detach().cpu().reshape(-1).long()
            if idx_arr.numel() != num_dets:
                raise RuntimeError(
                    "index count %d != detection count %d"
                    % (idx_arr.numel(), num_dets)
                )
            if idx_arr.numel() > 0:
                if int(idx_arr.min()) < 0 or int(idx_arr.max()) >= num_candidates:
                    raise RuntimeError("NMS candidate index out of range")

            gathered = score_src[b][:, idx_arr]
            gathered = gathered.transpose(0, 1).contiguous()

            result.class_scores = gathered.tolist() if num_dets > 0 else []
            result.candidate_ids = [int(v) for v in idx_arr.tolist()]

        return results


def predict_detections(model, image, **kwargs):
    existing = getattr(model, "predictor", None)
    if existing is not None and not isinstance(existing, FullScorePredictor):
        raise ValueError(
            "model.predictor is already set to %r; FullScorePredictor requires a "
            "fresh predictor or a FullScorePredictor instance"
            % (type(existing).__name__,)
        )

    results = model.predict(image, predictor=FullScorePredictor, **kwargs)
    if len(results) != 1:
        raise ValueError(
            "predict_detections supports a single image, got %d results"
            % (len(results),)
        )

    result = results[0]
    names = model.names
    num_classes = len(names)

    class_scores = getattr(result, "class_scores", None)
    candidate_ids = getattr(result, "candidate_ids", None)
    if class_scores is None or candidate_ids is None:
        raise ValueError("result is missing class_scores/candidate_ids")

    detections = detections_from_result(result, names)
    n = len(detections)

    if not isinstance(class_scores, (list, np.ndarray, torch.Tensor)):
        raise TypeError("class_scores must be a list/array/tensor")
    scores = np.asarray(_as_numpy(class_scores), dtype=float)
    if scores.size == 0:
        scores = scores.reshape(0, num_classes)
    if scores.ndim != 2 or scores.shape[1] != num_classes:
        raise ValueError(
            "class_scores must have shape (N,%d), got %r"
            % (num_classes, scores.shape)
        )
    if scores.shape[0] != n:
        raise ValueError(
            "length mismatch: detections=%d class_scores=%d" % (n, scores.shape[0])
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite class score")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("class scores must be within [0,1]")

    ids = list(candidate_ids)
    if len(ids) != n:
        raise ValueError(
            "length mismatch: detections=%d candidate_ids=%d" % (n, len(ids))
        )

    for i, det in enumerate(detections):
        row = scores[i]
        argmax = int(np.argmax(row))
        conf = float(det["confidence"])
        cid = int(det["class_id"])
        if abs(float(row[cid]) - conf) > 1e-5:
            raise ValueError(
                "detection %d: class score %r disagrees with confidence %r"
                % (i, float(row[cid]), conf)
            )
        if argmax != cid:
            raise ValueError(
                "detection %d: argmax class %d != detection class %d"
                % (i, argmax, cid)
            )
        det["class_scores"] = [float(v) for v in row.tolist()]
        det["candidate_id"] = int(ids[i])

    return detections
