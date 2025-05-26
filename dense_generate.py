# prostate_2D/app/dense_generate.py
import os, random, argparse, numpy as np
import torch, torch.nn.functional as F
import torchvision.transforms as T
from custom.models.itunet import itunet_2d

def unwrap_state_dict(blob):
    """PTFileModelPersistor wraps weights in {'model': …} – unwrap if needed."""
    if isinstance(blob, dict):
        return blob.get("model", blob.get("state_dict", blob))
    return blob

# simple geometry & colour jitter
affine = T.RandomApply([T.RandomAffine(20)], p=0.5)
colour = T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5)
hflip  = T.RandomHorizontalFlip()
def aug(img):
    return hflip(colour(affine(img)))


def cutmix(img_a, img_b, alpha=1.0):
    _, H, W = img_a.shape
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1. - lam)
    ch, cw  = int(H * cut_rat), int(W * cut_rat)
    cx, cy  = random.randint(0, W), random.randint(0, H)
    x1, y1  = np.clip(cx - cw // 2, 0, W), np.clip(cy - ch // 2, 0, H)
    x2, y2  = np.clip(cx + cw // 2, 0, W), np.clip(cy + ch // 2, 0, H)
    mixed = img_a.clone()
    mixed[:, y1:y2, x1:x2] = img_b[:, y1:y2, x1:x2]
    return mixed, (x1, x2, y1, y2)

def dice_loss(logits, target, smooth=1.0):
    probs  = torch.softmax(logits, dim=1)
    tgt_1h = F.one_hot(target, num_classes=2).permute(0,3,1,2).float()
    inter  = (probs * tgt_1h).sum((2,3))
    union  = probs.sum((2,3)) + tgt_1h.sum((2,3))
    dice   = (2*inter + smooth) / (union + smooth)
    return 1 - dice.mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--ckpt",  type=str)
    args = ap.parse_args()

    ckpt_path = args.ckpt or f"simulate_job/app_server/checkpoints/round-{args.round}.pt"
    state_dict = unwrap_state_dict(torch.load(ckpt_path, map_location="cpu"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = itunet_2d(n_channels=3, n_classes=2,
                       image_size=(384,384), transformer_depth=18).to(device)
    model.load_state_dict(state_dict)
    model.train()

    # synthetic fine-tune
    N_SYN, BATCH = 32, 4
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(N_SYN):
        imgs  = torch.randn(BATCH, 3, 384, 384).clamp_(-1,1)
        masks = torch.zeros(BATCH, 384, 384, dtype=torch.long)

        for i in range(BATCH):
            if random.random() < 0.5:
                imgs[i] = aug(imgs[i])
            else:
                j = random.randint(0, BATCH-1)
                imgs[i], (x1,x2,y1,y2) = cutmix(imgs[i], imgs[j])
                with torch.no_grad():
                    out = model(imgs[i:i+1].to(device))
                    if isinstance(out, (list,tuple)):          
                        out = out[0]                          # first element is logits
                    pseudo = out.argmax(1).cpu()[0]          
                masks[i,y1:y2,x1:x2] = pseudo[y1:y2,x1:x2]

        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs);  logits = logits[0] if isinstance(logits,(list,tuple)) else logits
        loss = F.cross_entropy(logits, masks) + dice_loss(logits, masks)

        opt.zero_grad(); loss.backward(); opt.step()
        if step % 4 == 0:
            print(f"[DENSE Gen] R{args.round} step {step:02d}/{N_SYN}  loss={loss:.4f}")

    # save
    out_dir = "simulate_job/app_server/checkpoints"; os.makedirs(out_dir, exist_ok=True)
    out_ckpt = f"{out_dir}/dense_gen_round-{args.round}.pt"
    torch.save({"model": model.state_dict()}, out_ckpt)

    
    print("Saved synthetic-tuned weights", out_ckpt)

if __name__ == "__main__":
    main()
