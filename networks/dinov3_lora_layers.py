import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3LinearLoRA(nn.Module):
    """Frozen Linear layer plus low-rank trainable update used for LoRA fine-tuning."""

    def __init__(self, base_linear, r=8, lora_alpha=8, dropout_rate=0.0, train_bias=False):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("DINOv3LinearLoRA expects an nn.Linear module.")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = int(r)
        self.scaling = float(lora_alpha) / max(1, self.r)
        self.dropout = nn.Dropout(dropout_rate) if float(dropout_rate) > 0 else nn.Identity()

        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=bool(train_bias))
        else:
            self.register_parameter("bias", None)

        if self.r <= 0:
            raise ValueError("DINOv3LinearLoRA requires rank r > 0.")

        self.w_lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
        self.w_lora_B = nn.Parameter(torch.empty(self.out_features, self.r))
        self.reset_parameters()
        self.use_lora = True

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.w_lora_B)

    def forward(self, x):
        base_output = F.linear(x, self.weight, self.bias)
        if not getattr(self, "use_lora", True):
            return base_output
        lora_input = self.dropout(x) if self.training else x
        lora_hidden = F.linear(lora_input, self.w_lora_A, bias=None)
        lora_output = F.linear(lora_hidden, self.w_lora_B, bias=None)
        return base_output + lora_output * self.scaling
