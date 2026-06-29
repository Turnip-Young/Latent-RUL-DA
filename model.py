import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class mymodel(nn.Module):
    def __init__(
        self,
        d_model=24,
        dropout=0.1,
        nhead=8,
        nlayers=2,
        max_len=500,
        use_latent=False,
        latent_dim=12,
        latent_hidden=64,
        latent_split=True,
        deg_latent_dim=0,
        fault_latent_dim=0,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.model_type = 'Transformer'
        self.use_latent = use_latent
        self.latent_split = latent_split
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, 512, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        if use_latent:
            if latent_split:
                if deg_latent_dim <= 0 and fault_latent_dim <= 0:
                    deg_latent_dim = max(1, int(round(latent_dim * 0.75)))
                    fault_latent_dim = max(1, latent_dim - deg_latent_dim)
                elif deg_latent_dim <= 0:
                    deg_latent_dim = max(1, latent_dim - fault_latent_dim)
                elif fault_latent_dim <= 0:
                    fault_latent_dim = max(1, latent_dim - deg_latent_dim)
                latent_dim = deg_latent_dim + fault_latent_dim
                self.latent_dim = latent_dim
                self.deg_latent_dim = deg_latent_dim
                self.fault_latent_dim = fault_latent_dim
                self.latent_encoder = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, latent_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.deg_projector = nn.Sequential(
                    nn.Linear(latent_hidden, deg_latent_dim),
                    nn.LayerNorm(deg_latent_dim),
                )
                self.fault_projector = nn.Sequential(
                    nn.Linear(latent_hidden, fault_latent_dim),
                    nn.LayerNorm(fault_latent_dim),
                )
                self.latent_decoder = nn.Sequential(
                    nn.Linear(latent_dim, latent_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_hidden, d_model),
                )
                self.deg_ranker = nn.Sequential(nn.Linear(deg_latent_dim, 1))
                self.decoder = nn.Linear(deg_latent_dim, 1)
            else:
                self.latent_dim = latent_dim
                self.deg_latent_dim = latent_dim
                self.fault_latent_dim = 0
                self.latent_encoder = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, latent_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_hidden, latent_dim),
                    nn.LayerNorm(latent_dim),
                )
                self.latent_decoder = nn.Sequential(
                    nn.Linear(latent_dim, latent_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_hidden, d_model),
                )
                self.deg_ranker = nn.Sequential(nn.Linear(latent_dim, 1))
                self.decoder = nn.Linear(latent_dim, 1)
        else:
            self.decoder = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.use_latent:
            modules = [self.latent_encoder, self.latent_decoder]
            if self.latent_split:
                modules += [self.deg_projector, self.fault_projector]
            modules.append(self.deg_ranker)
            for module in modules:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        layer.bias.data.zero_()
                        layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, key_msk, attn_msk=None, return_latent=False) -> Tensor:
        """
        return:
            output1: Tensor, extracted features
            output2: Tensor, predicted series
        """
        src = self.pos_encoder(src)
        output1 = self.transformer_encoder(src, attn_msk, key_msk)
        output1 = self.dropout(output1)
        if self.use_latent:
            latent_base = self.latent_encoder(output1)
            if self.latent_split:
                deg_latent = self.deg_projector(latent_base)
                fault_latent = self.fault_projector(latent_base)
                latent = torch.cat([deg_latent, fault_latent], dim=-1)
                reconstructed = self.latent_decoder(latent)
                deg_score = self.deg_ranker(deg_latent).squeeze(-1)
                output2 = self.decoder(deg_latent)
            else:
                latent = latent_base
                deg_latent = latent
                fault_latent = None
                reconstructed = self.latent_decoder(latent)
                deg_score = self.deg_ranker(deg_latent).squeeze(-1)
                output2 = self.decoder(deg_latent)
            if return_latent:
                return output1, output2, {
                    "z": latent,
                    "z_deg": deg_latent,
                    "z_fault": fault_latent,
                    "deg_score": deg_score,
                    "recon": reconstructed,
                    "features": output1,
                }
            return output1, output2
        output2 = self.decoder(output1)
        if return_latent:
            return output1, output2, None
        return output1, output2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, feature_num]
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)
        

class Discriminator(nn.Module): #D_y
    def __init__(self, in_features=24) -> None:
        super().__init__()
        self.in_features = in_features
        self.li = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: Tensor, shape [bts, in_features]
        """
        x = ReverseLayer.apply(x, 1)
        if x.size(0) == 1:
            pad = torch.zeros(1, self.in_features).cuda()
            x = torch.cat((x, pad), 0)
            y = self.li(x)[0].unsqueeze(0)
            return y
        return self.li(x)


class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
        

class backboneDiscriminator(nn.Module): #D_f
    def __init__(self, seq_len, d=24) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.li1 = nn.Linear(d, 1)
        self.li2 = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = ReverseLayer.apply(x, 1)
        out1 = self.li1(x).squeeze(2)
        if x.size(0) == 1:
            pad = torch.zeros(1, self.seq_len).cuda()
            out1 = torch.cat((out1, pad), 0)
            out2 = self.li2(out1)[0].unsqueeze(0)
            return out2
        out2 = self.li2(out1)
        return out2


if __name__ == "__main__":
    pass
    
