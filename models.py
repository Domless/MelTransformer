import math
import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class MelTransformer(nn.Module):
    def __init__(self, mel_dim=80, hidden_dim=256, ideal_dim=256, num_layers=4, nhead=4, is_mel_ideal = False):
        super().__init__()

        self.mel_proj = spectral_norm(nn.Linear(mel_dim, hidden_dim))
        #self.pos_enc = PositionalEncoding(hidden_dim)
        #self.conv_pre = weight_norm(Conv1d(80, hidden_dim, 7, 1, padding=3))
        #self.speaker_proj = nn.Linear(speaker_emb_dim, hidden_dim)
        if is_mel_ideal:
            self.ideal_proj = spectral_norm(nn.Linear(mel_dim, hidden_dim))
        else:
            self.ideal_proj = spectral_norm(nn.Linear(ideal_dim, hidden_dim))

        #self.pos_enc = nn.Parameter(torch.randn(1, 500, hidden_dim))  # если нужна позиция

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True, norm_first=True, dim_feedforward=hidden_dim*4
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True, norm_first=True, dim_feedforward=hidden_dim*4
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, mel_dim)
        #self.conv_pre = weight_norm(Conv1d(hidden_dim, 80, 7, 1, padding=3))

    def forward(self, mel_spec, ideal_mel):#speaker_embedding):
        """
        mel_spec: [B, T, mel_dim]
        speaker_embedding: [B, speaker_emb_dim]
        """
        # B, T, _ = mel_spec.shape

        # mel -> hidden + position
        #x = self.pos_enc(self.mel_proj(mel_spec))
        x = self.mel_proj(mel_spec)
        
        # Encode mel sequence
        encoded = self.encoder(x)  # [B, T, H]

        memory = self.ideal_proj(ideal_mel)#.unsqueeze(1)
        # Project speaker embedding -> memory [B, 1, H]
        #memory = self.speaker_proj(speaker_embedding).unsqueeze(1)

        # Decode: input is encoder output (tgt), context is speaker
        decoded = self.decoder(tgt=encoded, memory=memory)  # [B, T, H]

        # Project to mel-dim
        out = self.output_proj(decoded)  # [B, T, mel_dim]

        return out

class MelTransformer2(nn.Module):
    def __init__(self, mel_dim=80, hidden_dim=256, ideal_dim=256, num_layers=4, nhead=4, is_mel_ideal = False):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mel_dim, nhead=nhead, batch_first=True, norm_first=True, dim_feedforward=mel_dim*2
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=mel_dim, nhead=nhead, batch_first=True, norm_first=True, dim_feedforward=mel_dim*2
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, mel_spec, ideal_mel):#speaker_embedding):
        """
        mel_spec: [B, T, mel_dim]
        speaker_embedding: [B, speaker_emb_dim]
        """
        # B, T, _ = mel_spec.shape

        # mel -> hidden + position
        
        # Encode mel sequence
        encoded = self.encoder(mel_spec)  # [B, T, H]
        # Decode: input is encoder output (tgt), context is speaker
        decoded = self.decoder(tgt=encoded, memory=ideal_mel)  # [B, T, H]
        return decoded


class MelPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                Conv1d(80, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
        ])
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False

    def forward(self, real, pred):
        # [B, T, 80] → [B, 80, T]
        real = real.transpose(1, 2)
        pred = pred.transpose(1, 2)
        loss = 0.0
        for block in self.blocks:
            real = block(real)
            pred = block(pred)
            loss += F.l1_loss(pred, real)
        return loss
    
class MelDiscriminator(nn.Module):
    def __init__(self, mel_dim=80):
        super().__init__()
        self.model = nn.Sequential(
            Conv1d(mel_dim, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            Conv1d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            Conv1d(512, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x.transpose(1, 2))  # [B, T, mel] → [B, mel, T]

class StyleEncoder(nn.Module):
    def __init__(self, mel_dim=80, style_dim=128, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1),  # (B, 1, 80, T) → (B, hidden_dim, 40, T/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),  # → (B, hidden_dim, 20, T/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),  # → (B, hidden_dim, 10, T/8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),  # → (B, hidden_dim, 10, T/8)
            nn.ReLU()
        )
        #self.pool = nn.AdaptiveAvgPool2d((1, 1))  # глобальное усреднение
        self.fc = nn.Linear(hidden_dim, hidden_dim)  # вектор стиля

    def forward(self, mel_spec):
        """
        mel_spec: [B, T, mel_dim] → нужно будет перед подачей добавить канал
        """
        x = mel_spec.permute(0, 2, 1).unsqueeze(1)  # [B, 1, mel_dim, T]
        x = self.conv(x).squeeze(3).permute(0, 2, 1)
        #x = self.pool(x)
        #x = x.view(x.size(0), -1)
        style_vector = self.fc(x)  # [B, style_dim]
        return style_vector

LRELU_SLOPE = 0.1

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)
    
class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

