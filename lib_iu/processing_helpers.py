from contextlib import closing

from lib_iu.settings import TITLE, IterativeHiresSettings
from lib_iu.stages import stage_parameters
from modules import processing, scripts, shared
from modules.shared import state


def selected_processed_image(processed):
    if not processed.images:
        return None

    index = min(processed.index_of_first_image, len(processed.images) - 1)
    image = processed.images[index]
    if isinstance(image, tuple) and len(image) == 2:
        return image[0]

    return image


def find_script():
    runner = getattr(scripts, "scripts_txt2img", None)
    if runner is None:
        return None

    for script in runner.alwayson_scripts:
        try:
            if script.title() == TITLE:
                return script
        except Exception:
            continue

    return None


def settings_from_script_args(script_args):
    script = find_script()
    if script is None or script.args_from is None or script.args_to is None:
        return IterativeHiresSettings()

    values = list(script_args[script.args_from : script.args_to])
    if len(values) not in (8, 9):
        return IterativeHiresSettings()

    return IterativeHiresSettings(
        enabled=bool(values[0]),
        button_enabled=bool(values[1]),
        generation_enabled=bool(values[2]),
        iterations=max(1, int(values[3] or 1)),
        target_cfg=float(values[4]),
        target_steps=max(1, int(values[5] or 1)),
        target_denoise=float(values[6]),
        keep_intermediates=bool(values[7]),
        postprocess_final_only=bool(values[8]) if len(values) > 8 else True,
    )


def add_generation_metadata(p, settings, sizes):
    if settings.iterations <= 1:
        return

    p.extra_generation_params["Iterative Hires"] = f"{settings.iterations} passes"
    p.extra_generation_params["Iterative Hires Target CFG"] = settings.target_cfg
    p.extra_generation_params["Iterative Hires Target Steps"] = int(settings.target_steps)
    p.extra_generation_params["Iterative Hires Target Denoise"] = settings.target_denoise
    p.extra_generation_params["Iterative Hires Resolution Progression"] = "Geometric"
    p.extra_generation_params["Iterative Hires Sizes"] = " -> ".join(f"{w}x{h}" for w, h in sizes)


def set_stage_metadata(p, index, count):
    if count <= 1:
        return

    p.extra_generation_params["Iterative Hires Pass"] = f"{index + 1}/{count}"


def set_progress_totals(job_count, total_steps):
    state.job_count = max(1, int(job_count))
    state.processing_has_refined_job_count = True
    shared.total_tqdm.updateTotal(max(1, int(total_steps)))


def set_progress_stage(p, settings, index, count, width, height):
    cfg, steps, denoise = stage_parameters(p, settings, index, count)
    prefix = (
        f"Batch {getattr(p, 'iteration', 0) + 1}/{getattr(p, 'n_iter', 1)} - " if getattr(p, "n_iter", 1) > 1 else ""
    )
    label = (
        f"{prefix}Iterative Hires pass {index + 1}/{count}: "
        f"{int(width)}x{int(height)}, {steps} steps, CFG {cfg:g}, denoise {denoise:g}"
    )
    state.job = label
    state.textinfo = label


def run_processing(p):
    with closing(p):
        processed = scripts.scripts_txt2img.run(p, *p.script_args)
        if processed is None:
            processed = processing.process_images(p)

    return processed


class SkipPostprocessScriptRunner:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def postprocess_batch(self, *args, **kwargs):
        pass

    def postprocess_batch_list(self, *args, **kwargs):
        pass

    def postprocess_image(self, *args, **kwargs):
        pass

    def postprocess_maskoverlay(self, *args, **kwargs):
        pass

    def postprocess_image_after_composite(self, *args, **kwargs):
        pass
