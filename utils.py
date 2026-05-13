import torch
import torch.nn.functional as F
import config

def make_cls_hook(storage):
    def hook(module, inp, out): storage.append(out)
    return hook

def get_raw_logits(yolo_model, x):
    raw_cls = []
    detect = yolo_model.model[-1]
    # Detect 헤드를 train 모드로 전환 → _inference() 호출 생략, inference_mode 텐서 충돌 방지
    was_training = detect.training
    detect.train()
    hooks = [c.register_forward_hook(make_cls_hook(raw_cls)) for c in detect.cv3]
    _ = yolo_model(x)
    for h in hooks: h.remove()
    detect.train(was_training)
    B = x.shape[0]
    return torch.cat([t.view(B, t.shape[1], -1) for t in raw_cls], dim=2)

def gt_to_mask(boxes_list, size=None):
    """GT 박스 정보를 바탕으로 IMG_SIZE 해상도의 바이너리 마스크 생성"""
    size = size or config.IMG_SIZE
    B = len(boxes_list)
    mask = torch.zeros((B, 1, size, size)).to(config.DEVICE)
    for i, boxes in enumerate(boxes_list):
        for box in boxes:
            xc, yc, bw, bh = box
            x1 = int((xc - bw/2) * size)
            y1 = int((yc - bh/2) * size)
            x2 = int((xc + bw/2) * size)
            y2 = int((yc + bh/2) * size)
            mask[i, 0, max(0, y1):min(size, y2), max(0, x1):min(size, x2)] = 1.0
    return mask

def mask_to_anchor_weights(m_hw):
    weights = []
    for s in config.STRIDES:
        w = F.avg_pool2d(m_hw, s, s).view(m_hw.shape[0], -1)
        weights.append(w)
    return torch.cat(weights, dim=1)
