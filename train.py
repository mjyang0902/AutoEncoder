import os
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np
from utils.data import load_data
from model import SimpleCNN, AutoEncoder
import wandb


def build_backbone_meta(model: nn.Module):
    names, shapes, sizes = [], [], []
    for name, p in model.named_parameters():
        if name.startswith('backbone.') and p.requires_grad:
            names.append(name)
            shapes.append(p.shape)
            sizes.append(p.numel())
    total_dim = int(sum(sizes))
    if total_dim == 0:
         raise RuntimeError("No backbone parameters found. Checking SimpleCNN has parameters backbone.*, and requires_grad=True")
    return names, shapes, sizes, total_dim

@torch.no_grad()
def flatten_backbone_grads(model: nn.Module, names, sizes, device=None, dtype=None):
    vecs = []
    for name, sz in zip(names, sizes):
        p = dict(model.named_parameters())[name]
        if p.grad is None:
            v = torch.zeros(sz, device=device or p.device, dtype=dtype or p.dtype)
        else:
            v = p.grad.detach().reshape(-1).to(device or p.grad.device, dtype or p.grad.dtype)
        vecs.append(v)
    return torch.cat(vecs, dim=0)  

def rms_normalize(x: torch.Tensor, eps: float = 1e-8):
    s = torch.sqrt(torch.mean(x.float()**2)) + eps
    x_norm = (x / s).to(x.dtype)
    return x_norm, s

def train(args, train_loader, test_loader, criterion, device, AE, AE_optimizer, names, shapes, sizes, D, run):
    for time in range(5):
        num_classes = 1000 if args.data == 'ImageNet' else 10
        if args.model == 'SimpleCNN':
            model = SimpleCNN(in_ch=3, num_classes=num_classes).to(device)
        else:
            raise ValueError(f"Unsupported model: {args.model}")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        train_losses = []
        valid_accs = []
        AE_train_losses = []

        total_steps = args.epochs * len(train_loader)
        global_step = 0

        for epoch in tqdm(range(args.epochs)):
            model.train()
            epoch_losses = []
            AE_epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                with torch.no_grad():
                    g = flatten_backbone_grads(model, names, sizes, device=device, dtype=torch.float32)

                g_tilde, scale = rms_normalize(g)
                global_step += 1
                t = torch.tensor([global_step / total_steps], device=device, dtype=torch.float32)

                AE.train()
                AE_optimizer.zero_grad(set_to_none=True)
                x_hat, _ = AE(g_tilde.unsqueeze(0), t)
                ae_loss = torch.mean((x_hat.squeeze(0) - g_tilde)**2)
                ae_loss.backward()
                AE_optimizer.step()


                optimizer.step()
                epoch_losses.append(loss.item())
                AE_epoch_losses.append(ae_loss.item())


            avg_loss = np.mean(epoch_losses)
            avg_AE_loss = np.mean(AE_epoch_losses)
            train_losses.append(avg_loss)
            AE_train_losses.append(avg_AE_loss)

            valid_acc = test(model, test_loader, device)
            valid_accs.append(valid_acc)
            if time == 0: 
                run.log({"loss": avg_loss, "valid_acc": valid_acc, "AE_loss":avg_AE_loss})

    return AE

        
def test(model, test_loader, device):
    model.eval()
    acc = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = torch.argmax(model(data), dim=1)
            acc += torch.sum(output==target).cpu().item()
            count += target.size(0)

    model.train()
    return 100 * acc / count

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_data(data_name=args.data, data_path=args.store_path, batch_size=args.batch_size)

    names, shapes, sizes, D = build_backbone_meta(model)
    AE = AutoEncoder(input_dim=D,
                     latent_dim=128,
                     hidden_dims=[512, 128],
                     emb_dim=64,
                     cond_dim=256,
                     dropout=0.0).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    AE_optimizer = torch.optim.Adam(AE.parameters(), lr=1e-3)

    run = wandb.init(
        entity=None,
        mode=args.wandb_mode,
        project='Imagenet for autoencoder pretrain',
        config={
            "learning_rate": args.lr,
            "architecture": args.model,
            "dataset": args.data,
            "epochs": args.epochs,},
    )

    AE = train(args, train_loader, test_loader, criterion, device, AE, AE_optimizer, names, shapes, sizes, D, run)
    save_dir = os.path.join(args.store_path, "mjyang", "AutoEncoder")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "ae_state_dict": AE.state_dict(),
        "backbone_names": names,
        "backbone_shapes": shapes,
        "backbone_sizes": sizes,
        "D": D,
        "ae_cfg": {"latent_dim": 128, "hidden_dims": (512,128), "emb_dim": 64, "cond_dim": 256}
    }, os.path.join(save_dir, "timecond_ae_pretrain.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Autoencoder pretraining on ImageNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--batch-size', type=int, default=128, metavar='BATCH_SIZE', help='Batch size for training')
    parser.add_argument('--cuda-device', type=str, default='0', metavar='CUDA_DEVICE', help='CUDA device to use')
    parser.add_argument('--data', type=str, default='ImageNet', metavar='Data', help='Data')
    parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS', help='Number of epochs to train')
    parser.add_argument('--model', type = str, default = 'SimpleCNN', metavar = "Model", help = "Model",)
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR', help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=0, metavar='W', help='Weight decay')
    parser.add_argument('--wandb-mode', type = str, default = 'online', help = "mode of wandb",)
    parser.add_argument('--store-path', type = str, default = '../../../tmp2',)

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    main(args)
    torch.cuda.empty_cache()



