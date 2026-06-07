# SD Forge Iterative Hires

Iterative Hires is for Stable Diffusion WebUI Forge Neo that turns a single Hires Fix upscale into multiple smaller hires passes.

Instead of jumping directly from the base image to the final Hires Fix resolution, the extension builds a geometric resolution progression and runs each stage through Forge's normal hires pipeline. This can make large upscales easier to tune because later passes can use different CFG, step count, and denoising values.

## Features

- Works with both direct HR Fix, and quick-upscale HR Fix (sparkle button.)
- Treats Hires Fix `Scale by` / `Resize to` as the final target, not as a per-pass multiplier.
- Progresses resolution geometrically from the current image size to the final hires size.
- Progresses hires CFG, hires steps, and denoising strength linearly toward target values.
- Can return only the final image or keep intermediate pass images.
- Can suppress image postprocess hooks on intermediate button-upscale passes so tools such as ADetailer only mutate the final image.
- Writes pass count, target values, and stage sizes into generation metadata.

## Installation

Clone or copy this extension into your Forge `extensions` directory:

```text
stable-diffusion-webui-forge/extensions/sd-forge-iterative-hires
```

Restart Forge, then open txt2img and expand the `Iterative Hires` accordion.

## Controls

`Iterative Hires`
: Master enable switch.

`Iterate txt2img upscale button`
: Applies iterative hires when you click Forge's Hires Fix upscale button on an existing txt2img gallery image.

`Iterate normal Hires Fix generation`
: Applies iterative hires during ordinary txt2img generation when Hires Fix is enabled.

`Iterations`
: Total number of hires passes. `1` behaves like stock Hires Fix. Values above `1` enable staged upscaling.

`Target Hires CFG`
: CFG value used by the final iterative pass.

`Target Hires steps`
: Hires step count used by the final iterative pass.

`Target denoising strength`
: Denoising strength used by the final iterative pass.

`Keep intermediate images`
: For the txt2img upscale button mode, returns each intermediate pass image as well as the final result.

`Run image postprocess hooks only on final button pass`
: For the txt2img upscale button mode, skips image-mutating postprocess hooks on intermediate passes. This is useful when extensions such as ADetailer should only run once, after the final upscale.

## How Sizing Works

The extension uses the current image size as the source size and Forge's Hires Fix settings as the final target.

For example, with a `1024x1024` source image, `Scale by 2`, and `3` iterations, the final image is still `2048x2048`. The extension chooses intermediate stage sizes between `1024x1024` and `2048x2048`; it does not apply `2x` three separate times.

When `Resize to` is used instead of `Scale by`, the final iterative pass lands on the same target dimensions that Forge's normal Hires Fix would use.

## How Parameters Work

The first iterative pass uses your normal Hires Fix values. Later passes move linearly toward the target controls:

- Hires CFG moves from the current hires CFG to `Target Hires CFG`.
- Hires steps move from the current hires step count to `Target Hires steps`.
- Denoising strength moves from the current denoising strength to `Target denoising strength`.

With `2` iterations, the first pass uses the normal values and the second pass uses the target values. With more iterations, the middle passes use values between those endpoints.

## Modes

Can optionally be enabled for either Hires Fix control

### Txt2img Upscale Button

This mode starts from the selected gallery image. The selected image's actual dimensions are used as the source dimensions, so it works naturally with images that were generated at random or non-square aspect ratios.

This mode can optionally keep intermediate images and can suppress intermediate postprocess hooks.

### Normal Hires Fix Generation

This mode wraps Forge's normal `sample_hr_pass` during txt2img generation. It starts from the dimensions of the current generated batch, then runs the configured number of hires passes until it reaches Forge's calculated final Hires Fix target.

This is useful when you want every new txt2img generation to use staged hires automatically.

## Compatibility Notes

- The txt2img upscale button mode uses the selected gallery image dimensions and the image metadata path Forge already uses for quick hires upscales.
- Intermediate postprocess suppression only applies to the txt2img upscale button loop. Normal generation keeps Forge's usual script lifecycle.
- The extension is txt2img-only. It does not expose controls in img2img.

## Metadata

When more than one iteration is used, generated images include metadata such as:

- `Iterative Hires`
- `Iterative Hires Target CFG`
- `Iterative Hires Target Steps`
- `Iterative Hires Target Denoise`
- `Iterative Hires Resolution Progression`
- `Iterative Hires Sizes`
- `Iterative Hires Pass`

These fields make it easier to audit which stage sizes and target values were used for a result.

## Practical Tips

- Use `Iterations = 2` as a starting point. It gives you one normal hires pass and one target pass.
- Lower final denoising can help preserve composition on later passes.
- Lower final CFG can help reduce overcooking during large upscales.
- If another extension makes visible image edits, enable final-only postprocess hooks in button mode so those edits happen once.

## Requirements

- Stable Diffusion WebUI Forge.
- Hires Fix enabled for normal-generation mode.
- A selected txt2img gallery image for the upscale button mode.
