from dataclasses import dataclass

TITLE = "Iterative Upscale"
TXT2IMG_FIXED_ARG_COUNT = 25


@dataclass
class IterativeHiresSettings:
    enabled: bool = False
    button_enabled: bool = True
    generation_enabled: bool = False
    iterations: int = 1
    target_cfg: float = 2.0
    target_steps: int = 12
    target_denoise: float = 0.25
    keep_intermediates: bool = False
    postprocess_final_only: bool = True
    base_cfg: float = None
    base_steps: int = None
    base_denoise: float = None
