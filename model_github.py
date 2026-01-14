import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


class StableLayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig = x.dtype
        return super().forward(x.float()).to(orig)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.ln1 = StableLayerNorm(d_model)
        self.ln2 = StableLayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        kpm = None
        if attn_mask is not None:
            kpm = (attn_mask == 0).bool()
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=kpm)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class FluctuationExtractor(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.alpha_logits = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.proj = nn.Linear(d_model, out_dim)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, X: torch.Tensor, attn_mask: torch.Tensor = None):
        B, L, D = X.shape
        if L <= 1:
            z = X.mean(dim=1)
            return self.proj(z)

        S = X[:, 1:, :]
        if attn_mask is None:
            lengths = torch.full((B,), S.size(1), device=X.device, dtype=torch.long)
        else:
            lengths = attn_mask[:, 1:].sum(dim=1).to(torch.long)

        alpha = F.softmax(self.alpha_logits, dim=0)
        a1, a2 = alpha[0], alpha[1]

        z_list = []
        for i in range(B):
            tlen = int(lengths[i].item())
            if tlen <= 0:
                z_i = torch.zeros(D, device=X.device, dtype=X.dtype)
                z_list.append(z_i.unsqueeze(0))
                continue

            real_seq = S[i, :tlen, :]  # [tlen, D]
            if tlen == 1:
                z_i = real_seq.mean(dim=0)
                z_list.append(z_i.unsqueeze(0))
                continue

            diff1 = real_seq[1:, :] - real_seq[:-1, :]  # [tlen-1, D]
            if tlen >= 3:
                diff2 = real_seq[2:, :] - real_seq[:-2, :]  # [tlen-2, D]
                pad = diff1.size(0) - diff2.size(0)
                if pad > 0:
                    diff2 = F.pad(diff2, (0, 0, 0, pad), value=0.0)
                else:
                    diff2 = diff2[: diff1.size(0), :]
            else:
                diff2 = torch.zeros_like(diff1)

            z_t = a1 * diff1 + a2 * diff2
            z_i = z_t.mean(dim=0)
            z_list.append(z_i.unsqueeze(0))

        Z = torch.cat(z_list, dim=0)  # [B, D]
        return self.proj(Z)


class EchoEnhanceNet(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, lstm_hidden: int = None):
        super().__init__()
        D = d_model
        H = lstm_hidden or D
        self.d_model = D
        self.hid = H

        self.proj_h0 = nn.Linear(D, H)
        self.proj_c0 = nn.Linear(D, H)

        self.lstm = nn.LSTM(
            input_size=D,
            hidden_size=H,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.seq_proj = nn.Linear(2 * H, D)
        self.cls_gate = nn.Parameter(torch.zeros(D))

        self.attn = ResidualAttentionBlock(D, n_head, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(D, 2 * D),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * D, D),
        )
        self.drop = nn.Dropout(dropout)
        self.ln_out = StableLayerNorm(D)

        self._init()

    def _init(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        for m in [self.proj_h0, self.proj_c0, self.seq_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        nn.init.zeros_(self.cls_gate)

    def forward(self, X: torch.Tensor, attn_mask: torch.Tensor = None):
        B, L, D = X.shape
        cls = X[:, 0:1, :]
        S = X[:, 1:, :]

        cls_flat = cls.squeeze(1)
        h0 = self.proj_h0(cls_flat).unsqueeze(0).repeat(2, 1, 1)
        c0 = self.proj_c0(cls_flat).unsqueeze(0).repeat(2, 1, 1)

        if S.size(1) == 0:
            X_fused = cls
        else:
            if attn_mask is not None:
                seq_mask = attn_mask[:, 1:]
                lengths = seq_mask.sum(dim=1).to(torch.long).cpu()
                lengths = torch.clamp(lengths, min=1)
                packed = nn.utils.rnn.pack_padded_sequence(S, lengths, batch_first=True, enforce_sorted=False)
                packed_out, (hn, _) = self.lstm(packed, (h0, c0))
                lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_out, batch_first=True, total_length=S.size(1)
                )
            else:
                lstm_out, (hn, _) = self.lstm(S, (h0, c0))

            S_prime = self.seq_proj(lstm_out)
            hn_cat = torch.cat([hn[0], hn[1]], dim=-1)
            x_hat_cls = self.seq_proj(hn_cat).unsqueeze(1)

            g = torch.sigmoid(self.cls_gate).view(1, 1, -1)
            x_tilde_cls = cls * g + x_hat_cls * (1.0 - g)

            X_fused = torch.cat([x_tilde_cls, S_prime], dim=1)

        A = self.attn(X_fused, attn_mask)
        Y = self.ln_out(A + self.drop(self.ffn(A)))
        return Y


class FluctuationAwareConv1d(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, K: int, fluct_dim: int, dropout: float):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.K = K

        self.z_to_in = nn.Linear(fluct_dim, in_planes)
        self.z_to_w = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_planes // 2, K),
        )

        self.weight_bank = nn.Parameter(torch.randn(K, out_planes, in_planes, kernel_size))
        self.bias_bank = nn.Parameter(torch.zeros(K, out_planes))

        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.z_to_in.weight)
        nn.init.zeros_(self.z_to_in.bias)
        for k in range(self.K):
            nn.init.kaiming_uniform_(self.weight_bank[k])
        nn.init.zeros_(self.bias_bank)

    def forward(self, X: torch.Tensor, Z: torch.Tensor):
        proj = self.z_to_in(Z)
        logits = self.z_to_w(proj)
        omega = F.softmax(logits, dim=1)

        B, C, L = X.shape
        W = self.weight_bank.view(self.K, -1)
        W_tilde = torch.mm(omega, W).view(B * self.out_planes, self.in_planes, self.kernel_size)
        b_tilde = torch.mm(omega, self.bias_bank).view(-1)

        Xg = X.view(1, -1, L)
        Y = F.conv1d(
            Xg, W_tilde, b_tilde, stride=1, padding=self.kernel_size // 2, groups=B
        )
        return Y.view(B, self.out_planes, Y.size(-1))


class EchoDynamicsNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(config["text_encoder_name"])
        self.num_classes = int(config.get("class_num", 8))
        self.dropout = float(config.get("dropout", 0.1))
        self.n_head = int(config.get("head_num", 12))

        self.use_een = bool(config.get("use_een", True))
        self.use_fac = bool(config.get("use_fac", True))

        if bool(config.get("freeze_encoder", True)):
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        D = int(self.text_encoder.config.hidden_size)
        self.d_model = D
        fluct_dim = int(config.get("fluct_dim", 2 * D))

        if self.use_een:
            self.een = EchoEnhanceNet(D, self.n_head, self.dropout, lstm_hidden=D)

        self.mgfe = FluctuationExtractor(D, fluct_dim)

        if self.use_fac:
            conv_dim = int(config.get("conv_dim", 256))
            self.conv_proj = nn.Sequential(
                nn.Linear(D, conv_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            self.fac = FluctuationAwareConv1d(
                in_planes=conv_dim,
                out_planes=conv_dim,
                kernel_size=int(config.get("kernel_size", 5)),
                K=int(config.get("K", 2)),
                fluct_dim=fluct_dim,
                dropout=self.dropout,
            )
            self.conv_back = nn.Linear(conv_dim, D)
            self.fusion_gate = nn.Parameter(torch.zeros(D))
            self.ln_fuse = StableLayerNorm(D)

            for m in self.conv_proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            nn.init.xavier_uniform_(self.conv_back.weight)
            nn.init.zeros_(self.conv_back.bias)

        self.classifier = nn.Sequential(
            nn.Linear(D, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes),
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        enc = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        X = enc.last_hidden_state  # [B,L,D]

        X_een = self.een(X, attention_mask) if self.use_een else X
        Z = self.mgfe(X_een, attention_mask)

        if self.use_fac:
            V = self.conv_proj(X_een)  # [B,L,C]
            if attention_mask is not None:
                V = V * attention_mask.unsqueeze(-1).type_as(V)
            V = V.transpose(1, 2).contiguous()  # [B,C,L]
            Y = self.fac(V, Z).transpose(1, 2).contiguous()  # [B,L,C]
            X_conv = self.ln_fuse(self.conv_back(Y))  # [B,L,D]

            g = torch.sigmoid(self.fusion_gate).view(1, 1, -1)
            X_fused = X_een * g + X_conv * (1.0 - g)
        else:
            X_fused = X_een

        x_cls = X_fused[:, 0, :]
        logits = self.classifier(x_cls)
        return logits


if __name__ == "__main__":
    cfg = {
        "text_encoder_name": "bert-base-uncased",
        "class_num": 8,
        "head_num": 12,
        "dropout": 0.1,
        "freeze_encoder": False,
        "use_een": True,
        "use_fac": True,
        "K": 2,
        "kernel_size": 5,
        "conv_dim": 256,
    }
    m = EchoDynamicsNet(cfg)
    B, L = 4, 16
    ids = torch.randint(0, 30000, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    with torch.no_grad():
        out = m(ids, mask)
    print(out.shape)
