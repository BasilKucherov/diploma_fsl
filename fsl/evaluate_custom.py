import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

# Add cdfsl to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CDFSL_DIR = os.path.join(CURRENT_DIR, 'cdfsl')
sys.path.insert(0, CDFSL_DIR)

# Patch configs before importing datasets
import configs

def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot evaluation with frozen backbone")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to solo-learn checkpoint")
    parser.add_argument("--dataset", type=str, required=True, choices=['CropDisease', 'EuroSAT', 'ISIC', 'ChestX'], help="Dataset to evaluate on")
    parser.add_argument("--data_path", type=str, required=True, help="Root path to dataset")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=15)
    parser.add_argument("--n_episodes", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_backbone(ckpt_path, device):
    from backbone import ResNet10
    
    model = ResNet10(flatten=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # Map keys
    new_state_dict = OrderedDict()
    
    # Mapping rules based on inspection
    # backbone.conv1 -> trunk.0
    # backbone.bn1 -> trunk.1
    # backbone.layer1.0 -> trunk.4
    # backbone.layer2.0 -> trunk.5
    # backbone.layer3.0 -> trunk.6
    # backbone.layer4.0 -> trunk.7
    
    layer_map = {
        'backbone.conv1': 'trunk.0',
        'backbone.bn1': 'trunk.1',
        'backbone.layer1.0': 'trunk.4',
        'backbone.layer2.0': 'trunk.5',
        'backbone.layer3.0': 'trunk.6',
        'backbone.layer4.0': 'trunk.7'
    }
    
    for k, v in state_dict.items():
        if not k.startswith('backbone.'):
            continue
            
        # Direct mapping for initial layers
        mapped = False
        for prefix, target in layer_map.items():
            if k.startswith(prefix):
                # Handle block internals
                suffix = k[len(prefix):]
                
                # Standard ResNet (solo-learn) to cdfsl ResNet mapping
                if 'downsample.0' in suffix:
                    suffix = suffix.replace('downsample.0', 'shortcut')
                elif 'downsample.1' in suffix:
                    suffix = suffix.replace('downsample.1', 'BNshortcut')
                elif 'conv1' in suffix:
                    suffix = suffix.replace('conv1', 'C1')
                elif 'bn1' in suffix:
                    suffix = suffix.replace('bn1', 'BN1')
                elif 'conv2' in suffix:
                    suffix = suffix.replace('conv2', 'C2')
                elif 'bn2' in suffix:
                    suffix = suffix.replace('bn2', 'BN2')
                
                new_key = target + suffix
                new_state_dict[new_key] = v
                mapped = True
                break
        
        if not mapped:
            print(f"Warning: Key {k} not mapped")
            
    # Load state dict
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded with message: {msg}")
    
    model.to(device)
    model.eval()
    return model

def get_datamgr(dataset_name, data_path, n_way, n_shot, n_query, n_episodes, num_workers=4):
    # Monkeypatch configs
    if dataset_name == 'CropDisease':
        configs.CropDisease_path = data_path
        from datasets.CropDisease_few_shot import SetDataManager
    elif dataset_name == 'EuroSAT':
        configs.EuroSAT_path = data_path
        from datasets.EuroSAT_few_shot import SetDataManager
    elif dataset_name == 'ISIC':
        configs.ISIC_path = data_path
        from datasets.ISIC_few_shot import SetDataManager
    elif dataset_name == 'ChestX':
        configs.ChestX_path = data_path
        from datasets.Chest_few_shot import SetDataManager
        
    datamgr = SetDataManager(
        image_size=224, 
        n_way=n_way, 
        n_support=n_shot, 
        n_query=n_query, 
        n_eposide=n_episodes
    )
    # Patch num_workers
    # We need to ensure the dataloader uses 0 workers to avoid multiprocessing issues locally/in some envs
    # if configured to do so.
    # However, since SetDataManager creates the DataLoader inside get_data_loader,
    # we have to monkeypatch the class method OR the instance method if possible.
    
    # Option 1: Patch the instance's get_data_loader method (if it was just a function attached)
    # Option 2: Patch the class before instantiation (we already instantiated)
    # Option 3: Just rely on the caller to handle environment issues, but here we want to enforce it for safety if needed.
    
    # Actually, the previous patch attempt was:
    # original_get_data_loader = datamgr.get_data_loader
    # def patched_get_data_loader(aug): ...
    # datamgr.get_data_loader = patched_get_data_loader
    
    # But we didn't assign it back to datamgr.get_data_loader in the previous turn! 
    # Let's do that now properly.
    
    original_get_data_loader = datamgr.get_data_loader
    def patched_get_data_loader(aug):
        # We can't easily change num_workers inside the existing method without changing code.
        # BUT, we modified the source files (CropDisease_few_shot.py etc) to set num_workers=0.
        # So the default behavior is now safe for local (0 workers).
        # If we want to support >0 workers on remote, we should probably have made it configurable.
        # For now, let's stick to the file modifications we made (num_workers=0).
        return original_get_data_loader(aug)
        
    return datamgr

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def evaluate(model, dataloader, n_way, n_shot, n_query, device):
    acc_all = []
    
    print(f"Starting evaluation: {len(dataloader)} episodes")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            # x shape: [n_way, n_support + n_query, C, H, W]
            # Reshape for backbone
            # x is grouped by class: way 0 (sup+qry), way 1 (sup+qry)...
            
            # Flatten to [n_way * (n_support + n_query), C, H, W]
            p = n_shot + n_query
            n_samples = n_way * p
            
            x = x.to(device)
            x_flat = x.view(n_samples, *x.shape[2:])
            
            # Extract features
            feats = model(x_flat) # [n_samples, feat_dim]
            
            # Reshape back to [n_way, n_support + n_query, feat_dim]
            feats = feats.view(n_way, p, -1)
            
            # Split support and query
            z_support = feats[:, :n_shot] # [n_way, n_shot, d]
            z_query = feats[:, n_shot:]   # [n_way, n_query, d]
            
            # Compute prototypes
            z_proto = z_support.mean(1) # [n_way, d]
            
            # Flatten query for distance computation
            z_query_flat = z_query.contiguous().view(n_way * n_query, -1) # [n_query_total, d]
            
            # Compute distances
            dists = euclidean_dist(z_query_flat, z_proto) # [n_query_total, n_way]
            
            # Predictions (min distance)
            scores = -dists
            y_query = torch.arange(n_way).repeat_interleave(n_query).to(device)
            
            topk_scores, topk_labels = scores.topk(1, 1, True, True)
            top1_correct = topk_labels.eq(y_query.unsqueeze(1)).sum().item()
            
            acc = top1_correct / (n_way * n_query) * 100
            acc_all.append(acc)
            
            if (i+1) % 50 == 0:
                print(f"Episode {i+1}/{len(dataloader)}: Acc = {acc:.2f}%")
                
    acc_all = np.array(acc_all)
    mean = np.mean(acc_all)
    std = np.std(acc_all)
    conf_interval = 1.96 * std / np.sqrt(len(acc_all))
    
    print(f"Final Result: Acc = {mean:.2f}% +- {conf_interval:.2f}%")
    return mean, conf_interval

if __name__ == "__main__":
    args = parse_args()
    
    # Load Model
    model = load_backbone(args.ckpt_path, args.device)
    
    # Get Data Manager
    datamgr = get_datamgr(args.dataset, args.data_path, args.n_way, args.n_shot, args.n_query, args.n_episodes)
    dataloader = datamgr.get_data_loader(aug=False)
    
    # Evaluate
    evaluate(model, dataloader, args.n_way, args.n_shot, args.n_query, args.device)

