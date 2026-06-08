import gradio as gr

from lib_iu.generation import install_generation_wrapper, make_settings, should_wrap_generation
from lib_iu.settings import TITLE
from lib_iu.txt2img_button import patch_txt2img_upscale_function, register_txt2img_patch
from modules import scripts
from modules.ui_components import FormRow, InputAccordion


class Script(scripts.Script):
    section = "accordions"
    create_group = False
    sorting_priority = -1

    def title(self):
        return TITLE

    def show(self, is_img2img):
        return False if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):
        patch_txt2img_upscale_function()

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
        if not should_wrap_generation(p, enabled, generation_enabled, iterations):
            return

        settings = make_settings(
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
        )
        install_generation_wrapper(p, settings)


try:
    register_txt2img_patch()
except Exception:
    pass
