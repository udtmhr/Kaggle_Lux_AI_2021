import torch
ckpt = torch.load('outputs/survival_strategic/2026-08-15/23-58-20/1500160_weights.pt', map_location='cpu')
has_nan = False
for k, v in ckpt['model_state_dict'].items():
    if torch.isnan(v).any().item():
        print(f"NaN found in {k}")
        has_nan = True
if not has_nan:
    print("No NaNs found in weights.")
