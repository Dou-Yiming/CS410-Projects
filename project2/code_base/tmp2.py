import torch
import torch.nn as nn

# encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8)
# transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
# src = torch.rand(9, 128, 32)
# out = transformer_encoder(src)

encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8).cuda()
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).cuda()
h = torch.randn(9, 128, 32).cuda()
out = transformer_encoder(h)
print(out.shape)