import torch

from modules.ui import sRound


def lerp(start, end, index, count):
    if count <= 1:
        return start

    ratio = index / (count - 1)
    return start + (end - start) * ratio


def stage_parameters(p, settings, index, count):
    base_cfg = float(
        settings.base_cfg if settings.base_cfg is not None else getattr(p, "hr_cfg", getattr(p, "cfg_scale", 1.0))
    )
    base_steps = int(
        settings.base_steps
        if settings.base_steps is not None
        else (getattr(p, "hr_second_pass_steps", 0) or getattr(p, "steps", 1))
    )
    base_denoise = float(
        settings.base_denoise if settings.base_denoise is not None else getattr(p, "denoising_strength", 0.0)
    )

    cfg = float(lerp(base_cfg, float(settings.target_cfg), index, count))
    steps = max(1, int(round(lerp(base_steps, int(settings.target_steps), index, count))))
    denoise = max(0.0, min(1.0, float(lerp(base_denoise, float(settings.target_denoise), index, count))))

    return cfg, steps, denoise


def final_hires_size(width, height, hr_scale, hr_resize_x, hr_resize_y):
    if hr_resize_x == 0 and hr_resize_y == 0:
        return sRound(width * hr_scale), sRound(height * hr_scale)

    if hr_resize_y == 0:
        target_width = hr_resize_x
        target_height = hr_resize_x * (height / width)
    elif hr_resize_x == 0:
        target_width = hr_resize_y * (width / height)
        target_height = hr_resize_y
    else:
        target_width = hr_resize_x
        target_height = hr_resize_y

    return sRound(target_width), sRound(target_height)


def stage_sizes(width, height, final_width, final_height, count):
    if count <= 1:
        return [(final_width, final_height)]

    scale_x = final_width / width
    scale_y = final_height / height

    sizes = []
    for index in range(count):
        ratio = (index + 1) / count
        stage_width = sRound(width * (scale_x**ratio))
        stage_height = sRound(height * (scale_y**ratio))
        sizes.append((stage_width, stage_height))

    sizes[-1] = (final_width, final_height)
    return sizes


def apply_stage_to_processing(p, settings, index, count, width, height):
    cfg, steps, denoise = stage_parameters(p, settings, index, count)

    p.hr_cfg = cfg
    p.hr_second_pass_steps = steps
    p.denoising_strength = denoise
    p.hr_resize_x = int(width)
    p.hr_resize_y = int(height)
    p.hr_scale = 1.0
    p.hr_upscale_to_x = int(width)
    p.hr_upscale_to_y = int(height)
    p.hr_c = None
    p.hr_uc = None


def decoded_to_unit_tensor(decoded_samples):
    if isinstance(decoded_samples, torch.Tensor):
        decoded = decoded_samples.float()
    else:
        decoded = torch.stack(list(decoded_samples)).float()

    return torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)


def stage_step_counts(p, settings, count):
    return [stage_parameters(p, settings, index, count)[1] for index in range(count)]
