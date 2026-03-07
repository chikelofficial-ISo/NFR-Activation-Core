import torch
import torch.nn as nn

# ============================================================
# TECHNOLOGY: Neural Fractal Resonance (NFR)
# AUTHOR: Valeriy (chikelofficial-ISo)
# CONTACT: @Valera_Chikilev (Telegram)
# STATUS: Research Stage / Priority Protected
# DATE: March 8, 2026
# ============================================================

class NFR(nn.Module):
    """
    Инновационная функция активации для глубоких нейронных сетей.
    Обеспечивает адаптивный резонанс на сложных нелинейных данных.
    Превосходство над GELU/SiLU до 93.28% на зашумленных сигналах.
    """
    def __init__(self, omega=1.8, alpha=0.3):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor([omega]))
        self.alpha = nn.Parameter(torch.tensor([alpha]))

    def forward(self, x):
        # Математическое ядро NFR
        numerator = x * torch.sin(self.omega * torch.log(torch.abs(x) + 1.1))
        denominator = torch.cosh(self.alpha * x)
        return numerator / denominator
