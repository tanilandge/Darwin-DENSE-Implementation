# prostate_2D/app/dense_distill.py
import os, argparse, torch
from custom.models.itunet import itunet_2d

def unwrap(blob):
    """If persistor wrapped weights in {'model': …}, unwrap them."""
    if isinstance(blob, dict):
        return blob.get("model", blob.get("state_dict", blob))
    return blob

def load_itunet(state_dict, device):
    net = itunet_2d(
        n_channels=3, n_classes=2,
        image_size=(384,384), transformer_depth=18
    ).to(device)
    net.load_state_dict(state_dict, strict=False)
    return net

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--ckpt",  type=str, help="Path to global model (teacher 2)")
    args = ap.parse_args()

    # paths
    r = args.round
    base = "simulate_job/app_server/checkpoints"
    global_ckpt = args.ckpt or f"{base}/round-{r}.pt"
    synth_ckpt  = f"{base}/dense_gen_round-{r}.pt"

    # devices
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # load teacher (synthetic) & student (global)
    teacher_sd = unwrap(torch.load(synth_ckpt,  map_location="cpu"))
    student_sd = unwrap(torch.load(global_ckpt, map_location="cpu"))

    # param-space distillation
    distilled = {}
    for k in student_sd.keys():
        t = teacher_sd.get(k, student_sd[k])
        s = student_sd[k]
        if t.shape == s.shape:
            distilled[k] = 0.5 * t.to(torch.float32) + 0.5 * s.to(torch.float32)
        else:                       # mismatch (rare) – keep student value
            distilled[k] = s
    print(f"[DENSE Distill] Averaged {len(distilled)} tensors for round {r}")

    # save
    out = f"{base}/distilled_round-{r}.pt"
    torch.save({"model": distilled}, out)
    print("Saved distilled weights ➜", out)

if __name__ == "__main__":
    main()
