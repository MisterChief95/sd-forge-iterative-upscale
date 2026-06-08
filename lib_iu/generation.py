from lib_iu.processing_helpers import (
    add_generation_metadata,
    set_progress_stage,
    set_progress_totals,
    set_stage_metadata,
)
from lib_iu.settings import IterativeHiresSettings
from lib_iu.stages import (
    apply_stage_to_processing,
    decoded_to_unit_tensor,
    final_hires_size,
    stage_sizes,
    stage_step_counts,
)
from modules import devices, processing
from modules.shared import opts, state


def make_settings(
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
    return IterativeHiresSettings(
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


def should_wrap_generation(p, enabled, generation_enabled, iterations):
    if not enabled or not generation_enabled:
        return False
    if int(iterations or 1) <= 1:
        return False
    if not isinstance(p, processing.StableDiffusionProcessingTxt2Img):
        return False
    if not getattr(p, "enable_hr", False):
        return False
    if getattr(p, "txt2img_upscale", False) or getattr(p, "_iterative_hires_wrapped", False):
        return False

    return True


def install_generation_wrapper(p, settings):
    original_sample_hr_pass = p.sample_hr_pass
    original_hr_scale = p.hr_scale
    original_hr_resize_x = p.hr_resize_x
    original_hr_resize_y = p.hr_resize_y
    stage_steps = stage_step_counts(p, settings, settings.iterations)
    set_progress_totals(
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
            final_width, final_height = final_hires_size(
                source_width,
                source_height,
                float(original_hr_scale),
                int(original_hr_resize_x),
                int(original_hr_resize_y),
            )
        sizes = stage_sizes(source_width, source_height, final_width, final_height, settings.iterations)
        current_samples = samples
        current_decoded = decoded_samples
        current_width = source_width
        current_height = source_height
        result = None

        add_generation_metadata(p, settings, sizes)

        for index, (stage_width, stage_height) in enumerate(sizes):
            if state.interrupted:
                break

            set_progress_stage(p, settings, index, settings.iterations, stage_width, stage_height)
            p.width = current_width
            p.height = current_height
            apply_stage_to_processing(p, settings, index, settings.iterations, stage_width, stage_height)
            set_stage_metadata(p, index, settings.iterations)

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

            unit_decoded = decoded_to_unit_tensor(result)
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
