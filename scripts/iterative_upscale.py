import json
from contextlib import closing
from dataclasses import dataclass

import gradio as gr
import torch

from modules import devices, infotext_utils, processing, script_callbacks, scripts, shared
from modules.shared import opts, state
from modules.ui import plaintext_to_html, sRound
from modules.ui_components import FormRow, InputAccordion


TITLE = "Iterative Hires"
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


def _lerp(start, end, index, count):
    if count <= 1:
        return start

    ratio = index / (count - 1)
    return start + (end - start) * ratio


def _stage_parameters(p, settings, index, count):
    base_cfg = float(settings.base_cfg if settings.base_cfg is not None else getattr(p, "hr_cfg", getattr(p, "cfg_scale", 1.0)))
    base_steps = int(settings.base_steps if settings.base_steps is not None else (getattr(p, "hr_second_pass_steps", 0) or getattr(p, "steps", 1)))
    base_denoise = float(settings.base_denoise if settings.base_denoise is not None else getattr(p, "denoising_strength", 0.0))

    cfg = float(_lerp(base_cfg, float(settings.target_cfg), index, count))
    steps = max(1, int(round(_lerp(base_steps, int(settings.target_steps), index, count))))
    denoise = max(0.0, min(1.0, float(_lerp(base_denoise, float(settings.target_denoise), index, count))))

    return cfg, steps, denoise


def _final_hires_size(width, height, hr_scale, hr_resize_x, hr_resize_y):
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


def _stage_sizes(width, height, final_width, final_height, count):
    if count <= 1:
        return [(final_width, final_height)]

    scale_x = final_width / width
    scale_y = final_height / height

    sizes = []
    for index in range(count):
        ratio = (index + 1) / count
        stage_width = sRound(width * (scale_x ** ratio))
        stage_height = sRound(height * (scale_y ** ratio))
        sizes.append((stage_width, stage_height))

    sizes[-1] = (final_width, final_height)
    return sizes


def _apply_stage_to_processing(p, settings, index, count, width, height):
    cfg, steps, denoise = _stage_parameters(p, settings, index, count)

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


def _decoded_to_unit_tensor(decoded_samples):
    if isinstance(decoded_samples, torch.Tensor):
        decoded = decoded_samples.float()
    else:
        decoded = torch.stack(list(decoded_samples)).float()

    return torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)


def _selected_processed_image(processed):
    if not processed.images:
        return None

    index = min(processed.index_of_first_image, len(processed.images) - 1)
    image = processed.images[index]
    if isinstance(image, tuple) and len(image) == 2:
        return image[0]

    return image


def _find_script():
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


def _settings_from_script_args(script_args):
    script = _find_script()
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


def _add_generation_metadata(p, settings, sizes):
    if settings.iterations <= 1:
        return

    p.extra_generation_params["Iterative Hires"] = f"{settings.iterations} passes"
    p.extra_generation_params["Iterative Hires Target CFG"] = settings.target_cfg
    p.extra_generation_params["Iterative Hires Target Steps"] = int(settings.target_steps)
    p.extra_generation_params["Iterative Hires Target Denoise"] = settings.target_denoise
    p.extra_generation_params["Iterative Hires Resolution Progression"] = "Geometric"
    p.extra_generation_params["Iterative Hires Sizes"] = " -> ".join(f"{w}x{h}" for w, h in sizes)


def _set_stage_metadata(p, index, count):
    if count <= 1:
        return

    p.extra_generation_params["Iterative Hires Pass"] = f"{index + 1}/{count}"


def _stage_step_counts(p, settings, count):
    return [_stage_parameters(p, settings, index, count)[1] for index in range(count)]


def _set_progress_totals(job_count, total_steps):
    state.job_count = max(1, int(job_count))
    state.processing_has_refined_job_count = True
    shared.total_tqdm.updateTotal(max(1, int(total_steps)))


def _set_progress_stage(p, settings, index, count, width, height):
    cfg, steps, denoise = _stage_parameters(p, settings, index, count)
    prefix = f"Batch {getattr(p, 'iteration', 0) + 1}/{getattr(p, 'n_iter', 1)} - " if getattr(p, "n_iter", 1) > 1 else ""
    label = (
        f"{prefix}Iterative Hires pass {index + 1}/{count}: "
        f"{int(width)}x{int(height)}, {steps} steps, CFG {cfg:g}, denoise {denoise:g}"
    )
    state.job = label
    state.textinfo = label


def _run_processing(p):
    with closing(p):
        processed = scripts.scripts_txt2img.run(p, *p.script_args)
        if processed is None:
            processed = processing.process_images(p)

    return processed


class _SkipPostprocessScriptRunner:
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

def _iterative_txt2img_upscale_function(id_task, request, gallery, gallery_index, generation_info, *args):
    import modules.txt2img as txt2img

    settings = _settings_from_script_args(args[TXT2IMG_FIXED_ARG_COUNT:])
    original = getattr(txt2img, "_iterative_hires_original_upscale_function", None)

    if original is None or not settings.enabled or not settings.button_enabled or settings.iterations <= 1:
        return original(id_task, request, gallery, gallery_index, generation_info, *args)

    assert len(gallery) > 0, "No image to upscale"

    if gallery_index < 0 or gallery_index >= len(gallery):
        return gallery, generation_info, f"Bad image index: {gallery_index}", ""

    geninfo = json.loads(generation_info)
    first_image_index = geninfo.get("index_of_first_image", 0)
    count_images = len(geninfo.get("infotexts"))
    if len(gallery) > 1 and (gallery_index < first_image_index or gallery_index >= count_images):
        return gallery, generation_info, "Unable to upscale grid or control images.", ""

    current_image = infotext_utils.image_from_url_text(gallery[gallery_index])
    source_width, source_height = current_image.size

    final_width, final_height = _final_hires_size(
        source_width,
        source_height,
        float(args[11]),
        int(args[14]),
        int(args[15]),
    )
    sizes = _stage_sizes(source_width, source_height, final_width, final_height, settings.iterations)

    stage_outputs = []
    final_processed = None
    progress_initialized = False

    for index, (stage_width, stage_height) in enumerate(sizes):
        p = txt2img.txt2img_create_processing(id_task, request, *args, force_enable_hr=True)
        p.batch_size = 1
        p.n_iter = 1
        p.txt2img_upscale = True
        p._iterative_hires_button_loop = True
        p.firstpass_image = current_image
        p.width, p.height = current_image.size
        p.extra_generation_params["Original Size"] = f"{source_width}x{source_height}"
        p.override_settings["save_images_before_highres_fix"] = False

        if settings.postprocess_final_only and index < len(sizes) - 1:
            p.scripts = _SkipPostprocessScriptRunner(p.scripts)

        if not progress_initialized:
            _set_progress_totals(
                job_count=settings.iterations * 2,
                total_steps=sum(_stage_step_counts(p, settings, settings.iterations)),
            )
            progress_initialized = True

        _set_progress_stage(p, settings, index, settings.iterations, stage_width, stage_height)
        _apply_stage_to_processing(p, settings, index, settings.iterations, stage_width, stage_height)
        _add_generation_metadata(p, settings, sizes)
        _set_stage_metadata(p, index, settings.iterations)

        if not settings.keep_intermediates and index < len(sizes) - 1:
            p.do_not_save_samples = True
            p.do_not_save_grid = True

        processed = _run_processing(p)
        final_processed = processed

        current_image = _selected_processed_image(processed)
        if current_image is None:
            break

        if settings.keep_intermediates or index == len(sizes) - 1:
            stage_outputs.append((processed.images, processed.infotexts))

    shared.total_tqdm.clear()

    if final_processed is None:
        return gallery, generation_info, "Iterative Hires produced no images.", ""

    if settings.keep_intermediates:
        # Forge selects the first newly inserted image after the upscale click,
        # so put the final pass first and keep earlier passes after it.
        stage_outputs = list(reversed(stage_outputs))

    processed_to_insert = []
    infotexts_to_insert = []
    for images_list, infotexts_list in stage_outputs:
        processed_to_insert.extend(images_list)
        infotexts_to_insert.extend(infotexts_list)

    insert = getattr(shared.opts, "hires_button_gallery_insert", False)
    new_gallery = []
    new_infotexts = []

    for i, image in enumerate(gallery):
        if insert or i != gallery_index:
            image[0].already_saved_as = image[0].filename.rsplit("?", 1)[0]
            new_gallery.append(image)
            if i >= len(geninfo["infotexts"]):
                new_infotexts.append(None)
            else:
                new_infotexts.append(geninfo["infotexts"][i])
        if i == gallery_index:
            new_gallery.extend(processed_to_insert)
            new_infotexts.extend(infotexts_to_insert)

    geninfo["infotexts"] = new_infotexts

    return (
        gr.update(value=new_gallery, selected_index=gallery_index),
        json.dumps(geninfo),
        plaintext_to_html(final_processed.infotexts[0]),
        plaintext_to_html(final_processed.comments, classname="comments"),
    )


_iterative_txt2img_upscale_function._iterative_hires_wrapper = True


def _patch_txt2img_upscale_function():
    import modules.txt2img as txt2img

    if getattr(txt2img, "_iterative_hires_original_upscale_function", None) is None:
        txt2img._iterative_hires_original_upscale_function = txt2img.txt2img_upscale_function

    txt2img.txt2img_upscale_function = _iterative_txt2img_upscale_function


def _restore_txt2img_upscale_function():
    import modules.txt2img as txt2img

    original = getattr(txt2img, "_iterative_hires_original_upscale_function", None)
    if original is not None:
        txt2img.txt2img_upscale_function = original
        txt2img._iterative_hires_original_upscale_function = None


class Script(scripts.Script):
    section = "accordions"
    create_group = False
    sorting_priority = 3

    def title(self):
        return TITLE

    def show(self, is_img2img):
        return False if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):
        _patch_txt2img_upscale_function()

        with InputAccordion(False, label=TITLE, elem_id=self.elem_id("enabled")) as enabled:
            with FormRow(variant="compact"):
                button_enabled = gr.Checkbox(
                    value=True,
                    label="Iterate txt2img upscale button",
                    info="Loops the selected gallery image through the existing Hires Fix upscale button.",
                    elem_id=self.elem_id("button_enabled"),
                )
                generation_enabled = gr.Checkbox(
                    value=False,
                    label="Iterate normal Hires Fix generation",
                    info="Turns base -> hires into base -> hires -> hires when Hires Fix is enabled.",
                    elem_id=self.elem_id("generation_enabled"),
                )

            with FormRow(variant="compact"):
                iterations = gr.Slider(
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=1,
                    label="Iterations",
                    info="1 is stock Hires Fix. Target values are used only when this is 2 or higher.",
                    elem_id=self.elem_id("iterations"),
                )
                keep_intermediates = gr.Checkbox(
                    value=False,
                    label="Keep intermediate images",
                    info="For the txt2img upscale button loop. If off, only the final result is returned/saved.",
                    elem_id=self.elem_id("keep_intermediates"),
                )
                postprocess_final_only = gr.Checkbox(
                    value=True,
                    label="Run image postprocess hooks only on final button pass",
                    info="Prevents ADetailer and other post-sampling image mutators from running after intermediate txt2img upscale-button iterations. Cleanup-style postprocess hooks still run.",
                    elem_id=self.elem_id("postprocess_final_only"),
                )

            with FormRow(variant="compact"):
                target_cfg = gr.Slider(
                    minimum=1.0,
                    maximum=24.0,
                    step=0.5,
                    value=2.0,
                    label="Target Hires CFG",
                    elem_id=self.elem_id("target_cfg"),
                )
                target_steps = gr.Slider(
                    minimum=1,
                    maximum=150,
                    step=1,
                    value=12,
                    label="Target Hires steps",
                    elem_id=self.elem_id("target_steps"),
                )
                target_denoise = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                    label="Target denoising strength",
                    elem_id=self.elem_id("target_denoise"),
                )

            with gr.Accordion("How iteration is calculated", open=False):
                gr.Markdown(
                    "The first iterative pass uses the normal Hires Fix values already set above. "
                    "When Iterations is 2 or higher, each later pass linearly approaches the target CFG, steps, and denoising strength.\n\n"
                    "Resolution is handled as one total geometric upscale: the Hires Fix Scale by / Resize to values are treated as the final target, "
                    "and each pass uses a similar relative scale jump until the final requested size is reached."
                )

        return [
            enabled,
            button_enabled,
            generation_enabled,
            iterations,
            target_cfg,
            target_steps,
            target_denoise,
            keep_intermediates,
            postprocess_final_only,
        ]

    def before_process(
        self,
        p,
        enabled,
        button_enabled,
        generation_enabled,
        iterations,
        target_cfg,
        target_steps,
        target_denoise,
        keep_intermediates,
        postprocess_final_only,
    ):
        if not enabled or not generation_enabled:
            return
        if int(iterations or 1) <= 1:
            return
        if not isinstance(p, processing.StableDiffusionProcessingTxt2Img):
            return
        if not getattr(p, "enable_hr", False):
            return
        if getattr(p, "txt2img_upscale", False) or getattr(p, "_iterative_hires_wrapped", False):
            return

        settings = IterativeHiresSettings(
            enabled=enabled,
            button_enabled=button_enabled,
            generation_enabled=generation_enabled,
            iterations=max(1, int(iterations or 1)),
            target_cfg=float(target_cfg),
            target_steps=max(1, int(target_steps or 1)),
            target_denoise=float(target_denoise),
            keep_intermediates=keep_intermediates,
            postprocess_final_only=postprocess_final_only,
            base_cfg=float(getattr(p, "hr_cfg", getattr(p, "cfg_scale", 1.0))),
            base_steps=int(getattr(p, "hr_second_pass_steps", 0) or getattr(p, "steps", 1)),
            base_denoise=float(getattr(p, "denoising_strength", 0.0)),
        )

        original_sample_hr_pass = p.sample_hr_pass
        original_hr_scale = p.hr_scale
        original_hr_resize_x = p.hr_resize_x
        original_hr_resize_y = p.hr_resize_y
        stage_steps = _stage_step_counts(p, settings, settings.iterations)
        _set_progress_totals(
            job_count=p.n_iter * (settings.iterations + 1),
            total_steps=p.n_iter * (int(getattr(p, "steps", 1)) + sum(stage_steps)),
        )
        state.job = "Iterative Hires base pass"
        state.textinfo = "Iterative Hires base pass"
        p._iterative_hires_wrapped = True

        def iterative_sample_hr_pass(samples, decoded_samples, seeds, subseeds, subseed_strength, prompts):
            source_width = int(p.width)
            source_height = int(p.height)
            final_width = int(getattr(p, "hr_upscale_to_x", 0) or 0)
            final_height = int(getattr(p, "hr_upscale_to_y", 0) or 0)
            if final_width <= 0 or final_height <= 0:
                final_width, final_height = _final_hires_size(
                    source_width,
                    source_height,
                    float(original_hr_scale),
                    int(original_hr_resize_x),
                    int(original_hr_resize_y),
                )
            sizes = _stage_sizes(source_width, source_height, final_width, final_height, settings.iterations)
            current_samples = samples
            current_decoded = decoded_samples
            current_width = source_width
            current_height = source_height
            result = None

            _add_generation_metadata(p, settings, sizes)

            for index, (stage_width, stage_height) in enumerate(sizes):
                if state.interrupted:
                    break

                _set_progress_stage(p, settings, index, settings.iterations, stage_width, stage_height)
                p.width = current_width
                p.height = current_height
                _apply_stage_to_processing(p, settings, index, settings.iterations, stage_width, stage_height)
                _set_stage_metadata(p, index, settings.iterations)

                result = original_sample_hr_pass(
                    current_samples,
                    current_decoded,
                    seeds,
                    subseeds,
                    subseed_strength,
                    prompts,
                )

                if index == len(sizes) - 1:
                    break

                unit_decoded = _decoded_to_unit_tensor(result)
                current_width, current_height = stage_width, stage_height

                if p.latent_scale_mode is not None:
                    current_samples = processing.images_tensor_to_samples(
                        unit_decoded,
                        processing.approximation_indexes.get(opts.sd_vae_encode_method),
                        p.sd_model,
                    )
                    current_decoded = None
                else:
                    current_samples = None
                    current_decoded = unit_decoded

                devices.torch_gc()

            p.width = source_width
            p.height = source_height
            p.hr_scale = original_hr_scale
            p.hr_resize_x = original_hr_resize_x
            p.hr_resize_y = original_hr_resize_y
            p.hr_upscale_to_x = final_width
            p.hr_upscale_to_y = final_height
            p.hr_c = None
            p.hr_uc = None

            return result

        p.sample_hr_pass = iterative_sample_hr_pass


try:
    _patch_txt2img_upscale_function()
    script_callbacks.on_script_unloaded(_restore_txt2img_upscale_function)
except Exception:
    pass
