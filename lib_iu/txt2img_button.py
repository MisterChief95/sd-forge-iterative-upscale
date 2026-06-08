import json

import gradio as gr

from lib_iu.processing_helpers import (
    SkipPostprocessScriptRunner,
    add_generation_metadata,
    run_processing,
    selected_processed_image,
    set_progress_stage,
    set_progress_totals,
    set_stage_metadata,
    settings_from_script_args,
)
from lib_iu.settings import TXT2IMG_FIXED_ARG_COUNT
from lib_iu.stages import apply_stage_to_processing, final_hires_size, stage_sizes, stage_step_counts
from modules import infotext_utils, script_callbacks, shared
from modules.ui import plaintext_to_html


def iterative_txt2img_upscale_function(id_task, request, gallery, gallery_index, generation_info, *args):
    import modules.txt2img as txt2img

    settings = settings_from_script_args(args[TXT2IMG_FIXED_ARG_COUNT:])
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

    final_width, final_height = final_hires_size(
        source_width,
        source_height,
        float(args[11]),
        int(args[14]),
        int(args[15]),
    )
    sizes = stage_sizes(source_width, source_height, final_width, final_height, settings.iterations)

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
            p.scripts = SkipPostprocessScriptRunner(p.scripts)

        if not progress_initialized:
            set_progress_totals(
                job_count=settings.iterations * 2,
                total_steps=sum(stage_step_counts(p, settings, settings.iterations)),
            )
            progress_initialized = True

        set_progress_stage(p, settings, index, settings.iterations, stage_width, stage_height)
        apply_stage_to_processing(p, settings, index, settings.iterations, stage_width, stage_height)
        add_generation_metadata(p, settings, sizes)
        set_stage_metadata(p, index, settings.iterations)

        if not settings.keep_intermediates and index < len(sizes) - 1:
            p.do_not_save_samples = True
            p.do_not_save_grid = True

        processed = run_processing(p)
        final_processed = processed

        current_image = selected_processed_image(processed)
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


iterative_txt2img_upscale_function._iterative_hires_wrapper = True


def patch_txt2img_upscale_function():
    import modules.txt2img as txt2img

    if getattr(txt2img, "_iterative_hires_original_upscale_function", None) is None:
        txt2img._iterative_hires_original_upscale_function = txt2img.txt2img_upscale_function

    txt2img.txt2img_upscale_function = iterative_txt2img_upscale_function


def restore_txt2img_upscale_function():
    import modules.txt2img as txt2img

    original = getattr(txt2img, "_iterative_hires_original_upscale_function", None)
    if original is not None:
        txt2img.txt2img_upscale_function = original
        txt2img._iterative_hires_original_upscale_function = None


def register_txt2img_patch():
    patch_txt2img_upscale_function()
    script_callbacks.on_script_unloaded(restore_txt2img_upscale_function)
