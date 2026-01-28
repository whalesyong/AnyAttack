import matplotlib.pyplot as plt
import os, argparse, itertools, timm
from datetime import datetime 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import CLIPEncoder, Decoder
from robo_acc_dataset import AccidentTargetDataset, BddBaseDataset


class BiContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features  = F.normalize(text_features, p=2, dim=1)

        cos_sim = torch.matmul(image_features, text_features.t()) / self.temperature
        labels = torch.eye(cos_sim.size(0), device=cos_sim.device)

        loss_i2t = -torch.sum(labels * F.log_softmax(cos_sim, dim=1), dim=1).mean()
        loss_t2i = -torch.sum(labels * F.log_softmax(cos_sim.t(), dim=1), dim=1).mean()
        loss = (loss_i2t + loss_t2i) / 2

        avg_similarity = torch.diag(torch.matmul(image_features, text_features.t())).mean()
        return loss, avg_similarity


def make_dataloaders(args):
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_acc_dataset = AccidentTargetDataset(args.acc_data_dir, split=args.train_split, transform=train_tf, margin=0.10)
    train_bdd_dataset = BddBaseDataset(args.bdd_data_dir, split=args.train_split, transform=train_tf)

    val_acc_dataset = AccidentTargetDataset(args.acc_data_dir, split=args.val_split, transform=val_tf, margin=0.10)
    val_bdd_dataset = BddBaseDataset(args.bdd_data_dir, split=args.val_split, transform=val_tf)

    train_acc_loader = DataLoader(
        train_acc_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    train_bdd_loader = DataLoader(
        train_bdd_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_acc_loader = DataLoader(
        val_acc_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_bdd_loader = DataLoader(
        val_bdd_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_acc_loader, train_bdd_loader, val_acc_loader, val_bdd_loader


@torch.no_grad()
def validate(
    args,
    dev0, dev1,
    clip_encoder,
    eva_encoder,
    imagenet_encoder,
    decoder,
    criterion,
    val_acc_loader,
    val_bdd_loader,
):
    decoder.eval()
    clip_encoder.eval()
    eva_encoder.eval()
    imagenet_encoder.eval()

    bdd_iter = iter(itertools.cycle(val_bdd_loader))

    total_loss = 0.0
    total_sim_clip = 0.0
    total_sim_eva = 0.0
    total_sim_imnet = 0.0
    steps = 0

    pbar = tqdm(val_acc_loader, desc="Valid", leave=False)
    for (x_t, _meta_t) in pbar:
        x_t = x_t.to(dev0, non_blocking=True)
        x_r, _meta_r = next(bdd_iter)
        x_r = x_r.to(dev0, non_blocking=True)

        B = x_t.size(0)
        x_r = x_r[:B] 
        # --- target embeddings ---
        with torch.inference_mode(), autocast("cuda", enabled=args.amp, dtype=args.amp_dtype):
            z_t_clip = clip_encoder.encode_img(x_t)       # dev0
            z_t_imnet = imagenet_encoder(x_t)             # dev0

            # EVA on dev1
            x_t_dev1 = x_t.to(dev1, non_blocking=True)
            x_t_eva = F.interpolate(x_t_dev1, size=(448, 448), mode="bilinear", align_corners=False)
            z_t_eva = eva_encoder(x_t_eva).to(dev0, non_blocking=True)

        # --- delta + adv ---
        with autocast("cuda", enabled=args.amp, dtype=args.amp_dtype):
            delta = decoder(z_t_clip)
            delta = torch.clamp(delta, -args.eps, args.eps)

        x_adv = torch.clamp(x_r + delta, 0.0, 1.0)        # dev0

        # --- adv embeddings ---
        with torch.inference_mode(), autocast("cuda", enabled=args.amp, dtype=args.amp_dtype):
            z_adv_clip = clip_encoder.encode_img(x_adv)   # dev0
            z_adv_imnet = imagenet_encoder(x_adv)         # dev0

            x_adv_dev1 = x_adv.to(dev1, non_blocking=True)
            x_adv_eva = F.interpolate(x_adv_dev1, size=(448, 448), mode="bilinear", align_corners=False)
            z_adv_eva = eva_encoder(x_adv_eva).to(dev0, non_blocking=True)

        loss_clip, sim_clip = criterion(z_adv_clip, z_t_clip)
        loss_eva, sim_eva = criterion(z_adv_eva, z_t_eva)
        loss_imnet, sim_imnet = criterion(z_adv_imnet, z_t_imnet)

        loss_total = (args.w_clip * loss_clip) + (args.w_eva * loss_eva) + (args.w_imnet * loss_imnet)

        total_loss += float(loss_total.item())
        total_sim_clip += float(sim_clip.item())
        total_sim_eva += float(sim_eva.item())
        total_sim_imnet += float(sim_imnet.item())
        steps += 1

        pbar.set_postfix({
            "loss": total_loss / steps,
            "sim_clip": total_sim_clip / steps,
            "sim_eva": total_sim_eva / steps,
            "sim_imnet": total_sim_imnet / steps,
        })

    return {
        "loss": total_loss / max(steps, 1),
        "sim_clip": total_sim_clip / max(steps, 1),
        "sim_eva": total_sim_eva / max(steps, 1),
        "sim_imnet": total_sim_imnet / max(steps, 1),
    }


def train(args):
    assert torch.cuda.is_available(), "CUDA not available"
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs for EVA-on-dev1 setup"

    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")

    # --- models ---
    clip_encoder = CLIPEncoder("ViT-B/32").to(dev0).eval()

    # ONLY EVA on dev1
    eva_encoder = timm.create_model(
        "hf_hub:timm/eva02_large_patch14_448.mim_m38m_ft_in1k",
        num_classes=0,
        pretrained=True
    ).to(dev1).eval()

    # ImageNet ViT stays on dev0
    imagenet_encoder = torchvision.models.vit_b_16(pretrained=True).to(dev0).eval()
    imagenet_encoder.head = torch.nn.Identity()

    decoder = Decoder(embed_dim=512).to(dev0)

    # freeze encoders (saves memory + avoids accidental grads)
    for m in [clip_encoder, eva_encoder, imagenet_encoder]:
        for p in m.parameters():
            p.requires_grad_(False)

    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=dev0)
        decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
        print(f"Loaded decoder checkpoint: {args.ckpt}")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=1)
    scaler = GradScaler(enabled=args.amp)

    criterion = BiContrastiveLoss(temperature=args.temp).to(dev0)
    train_acc_loader, train_bdd_loader, val_acc_loader, val_bdd_loader = make_dataloaders(args)

    bdd_iter = iter(itertools.cycle(train_bdd_loader))

    os.makedirs(args.out_dir, exist_ok=True)
    global_step = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_sim_clip": [],
        "train_sim_eva": [],
        "train_sim_imnet": [],
        "val_sim_clip": [],
        "val_sim_eva": [],
        "val_sim_imnet": [],
    }

    
    for epoch in range(args.epoch):
        decoder.train()

        running_loss = 0.0
        running_sim_clip = 0.0
        running_sim_eva = 0.0
        running_sim_imnet = 0.0
        steps = 0

        pbar = tqdm(train_acc_loader, desc=f"Train E{epoch:02d}", leave=True)

        for batch_idx, (x_t, _meta_t) in enumerate(pbar):
            x_t = x_t.to(dev0, non_blocking=True)

            # --- target embeddings (no grad) ---
            with torch.inference_mode(), autocast("cuda", enabled=args.amp, dtype=args.amp_dtype):
                z_t_clip = clip_encoder.encode_img(x_t)   # dev0
                z_t_imnet = imagenet_encoder(x_t)         # dev0

                # EVA on dev1
                x_t_dev1 = x_t.to(dev1, non_blocking=True)
                x_t_eva = F.interpolate(x_t_dev1, size=(448, 448), mode="bilinear", align_corners=False)
                z_t_eva = eva_encoder(x_t_eva).to(dev0, non_blocking=True)
            # Detach to be extra safe
            z_t_clip = z_t_clip.detach()
            z_t_eva = z_t_eva.detach()
            z_t_imnet = z_t_imnet.detach()
            optimizer.zero_grad(set_to_none=True)

            # --- delta from decoder (grad flows only through decoder) ---
            with autocast("cuda", enabled=args.amp, dtype=args.amp_dtype):
                delta = decoder(z_t_clip)
                delta = torch.clamp(delta, -args.eps, args.eps)

                loss_accum = 0.0
                sim_clip_accum = 0.0
                sim_eva_accum = 0.0
                sim_imnet_accum = 0.0

                for _k in range(args.k_aug):
                    x_r, _meta_r = next(bdd_iter)
                    x_r = x_r.to(dev0, non_blocking=True)

                    x_adv = torch.clamp(x_r + delta, 0.0, 1.0)

                    # --- adv embeddings ---
                    with autocast("cuda", enabled=args.amp, dtype=args.amp_dtype):
                        z_adv_clip = clip_encoder.encode_img(x_adv)  # dev0
                        z_adv_imnet = imagenet_encoder(x_adv)        # dev0

                        x_adv_dev1 = x_adv.to(dev1, non_blocking=True)
                        x_adv_eva = F.interpolate(x_adv_dev1, size=(448, 448), mode="bilinear", align_corners=False)
                        z_adv_eva = eva_encoder(x_adv_eva).to(dev0, non_blocking=True)

                    loss_clip, sim_clip = criterion(z_adv_clip, z_t_clip)
                    loss_eva, sim_eva = criterion(z_adv_eva, z_t_eva)
                    loss_imnet, sim_imnet = criterion(z_adv_imnet, z_t_imnet)

                    loss_total = (args.w_clip * loss_clip) + (args.w_eva * loss_eva) + (args.w_imnet * loss_imnet)

                    loss_accum = loss_accum + loss_total
                    sim_clip_accum = sim_clip_accum + sim_clip
                    sim_eva_accum = sim_eva_accum + sim_eva
                    sim_imnet_accum = sim_imnet_accum + sim_imnet

                loss_accum = loss_accum / args.k_aug
                sim_clip_accum = sim_clip_accum / args.k_aug
                sim_eva_accum = sim_eva_accum / args.k_aug
                sim_imnet_accum = sim_imnet_accum / args.k_aug

                if args.delta_l2 > 0:
                    loss_accum = loss_accum + args.delta_l2 * (delta.pow(2).mean())

            scaler.scale(loss_accum).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step(epoch + batch_idx / max(len(train_acc_loader), 1))

            running_loss += float(loss_accum.item())
            running_sim_clip += float(sim_clip_accum.item())
            running_sim_eva += float(sim_eva_accum.item())
            running_sim_imnet += float(sim_imnet_accum.item())
            steps += 1
            global_step += 1

            pbar.set_postfix({
                "loss": running_loss / steps,
                "sim_clip": running_sim_clip / steps,
                "sim_eva": running_sim_eva / steps,
                "sim_imnet": running_sim_imnet / steps,
                "lr": optimizer.param_groups[0]["lr"],
            })

            if args.log_every > 0 and (global_step % args.log_every == 0):
                print(
                    f"[E{epoch:02d} S{global_step:06d}] "
                    f"loss={running_loss/steps:.4f} "
                    f"sim_clip={running_sim_clip/steps:.4f} "
                    f"sim_eva={running_sim_eva/steps:.4f} "
                    f"sim_imnet={running_sim_imnet/steps:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        # epoch-level train averages
        train_loss_epoch = running_loss / max(steps, 1)
        train_sim_clip_epoch = running_sim_clip / max(steps, 1)
        train_sim_eva_epoch = running_sim_eva / max(steps, 1)
        train_sim_imnet_epoch = running_sim_imnet / max(steps, 1)
        
        history["train_loss"].append(train_loss_epoch)
        history["train_sim_clip"].append(train_sim_clip_epoch)
        history["train_sim_eva"].append(train_sim_eva_epoch)
        history["train_sim_imnet"].append(train_sim_imnet_epoch)

        # --- validation (same EVA-on-dev1 logic) ---
        val_stats = validate(
            args, dev0, dev1,
            clip_encoder, eva_encoder, imagenet_encoder,
            decoder, criterion,
            val_acc_loader, val_bdd_loader
        )
        print(f"Epoch {epoch} VAL: {val_stats}")
        history["val_loss"].append(val_stats["loss"])
        history["val_sim_clip"].append(val_stats["sim_clip"])
        history["val_sim_eva"].append(val_stats["sim_eva"])
        history["val_sim_imnet"].append(val_stats["sim_imnet"])


        ckpt_path = os.path.join(args.out_dir, f"decoder_epoch_{epoch:02d}_loss_{val_stats['loss']:.4f}.pt")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    if args.test:
        print("Starting testing...")

        test_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        test_acc_dataset = AccidentTargetDataset(args.acc_data_dir, split="test", transform=test_tf, margin=0.10)
        test_bdd_dataset = BddBaseDataset(args.bdd_data_dir, split="test", transform=test_tf)

        test_acc_loader = DataLoader(test_acc_dataset, args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True, drop_last=False)
        test_bdd_loader = DataLoader(test_bdd_dataset, args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True, drop_last=False)

        test_stats = validate(
            args, dev0, dev1,
            clip_encoder, eva_encoder, imagenet_encoder,
            decoder, criterion,
            test_acc_loader, test_bdd_loader
        )
        print(f"Final test stats: {test_stats}")
    save_epoch_plots(history, args.out_dir)
    print(f"Saved plots to {args.out_dir}/loss_epoch.png and {args.out_dir}/sims_epoch.png")


def save_epoch_plots(history: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(len(history["train_loss"])))

    # --- Loss plot ---
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_epoch.png"), dpi=200)
    plt.close()

    # --- Similarities plot (3 encoders) ---
    plt.figure()
    plt.plot(epochs, history["train_sim_clip"], label="train sim_clip")
    plt.plot(epochs, history["val_sim_clip"], label="val sim_clip")
    plt.plot(epochs, history["train_sim_eva"], label="train sim_eva")
    plt.plot(epochs, history["val_sim_eva"], label="val sim_eva")
    plt.plot(epochs, history["train_sim_imnet"], label="train sim_imnet")
    plt.plot(epochs, history["val_sim_imnet"], label="val sim_imnet")
    plt.xlabel("epoch")
    plt.ylabel("avg similarity")
    plt.title("Encoder Similarities vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sims_epoch.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc_data_dir", type=str, required=True)
    parser.add_argument("--bdd_data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="")

    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--t0", type=float, default=5)

    parser.add_argument("--eps", type=float, default=16 / 255)
    parser.add_argument("--k_aug", type=int, default=4)

    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--w_clip", type=float, default=1.0)
    parser.add_argument("--w_eva", type=float, default=1.0)
    parser.add_argument("--w_imnet", type=float, default=1.0)

    parser.add_argument("--delta_l2", type=float, default=0.0)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=200)

    parser.add_argument("--amp", action="store_true")
    # bf16 is great on H100; change to torch.float16 if needed
    parser.add_argument("--amp_dtype", type=lambda s: {"bf16": torch.bfloat16, "fp16": torch.float16}[s],
                        default=torch.bfloat16, choices=["bf16", "fp16"])

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="valid")  # change to "val" if your folder uses val
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    train(args)
