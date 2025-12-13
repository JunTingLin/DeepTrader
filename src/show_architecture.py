#!/usr/bin/env python3
"""
Show ASU and MSU architecture using torchinfo
Usage: python show_architecture.py
"""
import torch
import json
from torchinfo import summary
from model.ASU import ASU
from model.MSU import MSU

# Load hyperparameters from hyper.json
with open('hyper.json', 'r') as f:
    args = json.load(f)

print("=" * 80)
print("Configuration from hyper.json:")
print("=" * 80)
print(f"  num_assets: {args['num_assets']}")
print(f"  window_len: {args['window_len']}")
print(f"  hidden_dim: {args['hidden_dim']}")
print(f"  num_blocks (layers): {args['num_blocks']}")
print(f"  kernel_size: {args['kernel_size']}")
print(f"  batch_size: {args['batch_size']}")
print()
print("ASU Configuration:")
print(f"  - in_features: {args['in_features'][0]}")
print(f"  - transformer_asu_bool: {args['transformer_asu_bool']} {'✓ Using 2D Transformer (TE)' if args['transformer_asu_bool'] else '✗ Using GCN'}")
print(f"  - spatial_bool: {args['spatial_bool']} {'✓ Using Spatial Attention' if args['spatial_bool'] else '✗ Disabled'}")
print(f"  - addaptiveadj: {args['addaptiveadj']}")
print()
print("MSU Configuration:")
print(f"  - in_features: {args['in_features'][1]}")
print(f"  - transformer_msu_bool: {args['transformer_msu_bool']} {'✓ Using 1D Transformer (TE_1D)' if args['transformer_msu_bool'] else '✗ Using LSTM'}")
print(f"  - temporal_attention_bool: {args['temporal_attention_bool']} {'✓ Using Temporal Attention' if args['temporal_attention_bool'] else '✗ Using Mean Pooling'}")
print("=" * 80)

# Create dummy supports for ASU (industry relationship matrix)
# In real training, this would be loaded from relation_file
supports = [torch.eye(args['num_assets'])]

print("\n" + "=" * 80)
print("ASU (Asset Selection Unit) Architecture")
print("=" * 80)

# Create ASU model
asu_model = ASU(
    num_nodes=args['num_assets'],
    in_features=args['in_features'][0],
    hidden_dim=args['hidden_dim'],
    window_len=args['window_len'],
    dropout=args['dropout'],
    kernel_size=args['kernel_size'],
    layers=args['num_blocks'],
    supports=supports,
    spatial_bool=args['spatial_bool'],
    addaptiveadj=args['addaptiveadj'],
    transformer_asu_bool=args['transformer_asu_bool'],
    num_assets=args['num_assets']
)

# Print ASU summary
# Input shape: [batch_size, num_assets, window_len, in_features]
print()
print("ASU Architecture Details:")
print("-" * 80)
print(f"Input: [batch={args['batch_size']}, assets={args['num_assets']}, time={args['window_len']}, features={args['in_features'][0]}]")
print()
print("Layer Structure:")
print("  1. Start Conv (1x1): features 34 → hidden_dim 128")
print("  2. Padding: time 13 → 16 (receptive_field requirement)")
print()
for layer in range(args['num_blocks']):
    dilation = 2 ** layer
    print(f"  Layer {layer}:")
    print(f"    - TCN: Conv2d(kernel=(1,2), dilation={dilation})")
    if args['transformer_asu_bool']:
        print(f"    - 2D Transformer (TE): depth=2, heads=4")
    else:
        print(f"    - GCN: support_len={len(supports)}")
    if args['spatial_bool']:
        print(f"    - Spatial Attention: (30×30) attention matrix")
    print()

print("  Final: Linear(128 → 1) + Sigmoid → Stock scores")
print("-" * 80)
print()

summary(
    asu_model,
    input_data=[
        torch.randn(args['batch_size'], args['num_assets'], args['window_len'], args['in_features'][0]),  # inputs
        torch.zeros(args['batch_size'], args['num_assets'], dtype=torch.bool)  # mask
    ],
    col_names=['input_size', 'output_size', 'num_params', 'trainable'],
    depth=3,
    row_settings=['var_names'],
    verbose=1
)

print("\n" + "=" * 80)
print("MSU (Market Situation Unit) Architecture")
print("=" * 80)

# Create MSU model
msu_model = MSU(
    in_features=args['in_features'][1],
    window_len=args['window_len'],
    hidden_dim=args['hidden_dim'],
    transformer_msu_bool=args['transformer_msu_bool'],
    temporal_attention_bool=args['temporal_attention_bool']
)

# Print MSU summary
# Input shape: [batch_size, window_len, in_features]
print()
print("MSU Architecture Details:")
print("-" * 80)
print(f"Input: [batch={args['batch_size']}, time={args['window_len']}, features={args['in_features'][1]}]")
print()
print("Layer Structure:")
if args['transformer_msu_bool']:
    print("  1. TE_1D (1D Transformer):")
    print(f"    - Input embedding: features {args['in_features'][1]} → hidden_dim {args['hidden_dim']}")
    print(f"    - Positional Encoding: Sinusoidal (time={args['window_len']})")
    print(f"    - Transformer: depth=2, heads=4")
    print(f"    - Output: [batch, time={args['window_len']}, dim={args['hidden_dim']}]")
else:
    print("  1. LSTM:")
    print(f"    - Input: features {args['in_features'][1]} → hidden {args['hidden_dim']}")
    print(f"    - Output: [time={args['window_len']}, batch, hidden={args['hidden_dim']}]")

print()
if args['temporal_attention_bool']:
    print(f"  2. Temporal Attention:")
    print(f"    - Attend over {args['window_len']} time steps")
    print(f"    - Output: weighted sum → [batch, {args['hidden_dim']}]")
else:
    print(f"  2. Mean Pooling:")
    print(f"    - Average over {args['window_len']} time steps")
    print(f"    - Output: [batch, {args['hidden_dim']}]")

print()
print("  3. MLP Head:")
print(f"    - Linear({args['hidden_dim']} → {args['hidden_dim']}) + ReLU + BN")
print(f"    - Linear({args['hidden_dim']} → 2) → [mu, sigma] for rho distribution")
print("-" * 80)
print()

summary(
    msu_model,
    input_size=(args['batch_size'], args['window_len'], args['in_features'][1]),
    col_names=['input_size', 'output_size', 'num_params', 'trainable'],
    depth=3,
    row_settings=['var_names'],
    verbose=1
)

print("\n" + "=" * 80)
print("Model Summary")
print("=" * 80)

# Count total parameters
asu_params = sum(p.numel() for p in asu_model.parameters())
asu_trainable_params = sum(p.numel() for p in asu_model.parameters() if p.requires_grad)
msu_params = sum(p.numel() for p in msu_model.parameters())
msu_trainable_params = sum(p.numel() for p in msu_model.parameters() if p.requires_grad)

print(f"ASU Total Parameters: {asu_params:,}")
print(f"ASU Trainable Parameters: {asu_trainable_params:,}")
print(f"\nMSU Total Parameters: {msu_params:,}")
print(f"MSU Trainable Parameters: {msu_trainable_params:,}")
print(f"\nTotal Parameters: {asu_params + msu_params:,}")
print(f"Total Trainable Parameters: {asu_trainable_params + msu_trainable_params:,}")
print("=" * 80)