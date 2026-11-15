"""Local background remover. Runs offline after first model download.
Full-resolution output: the model predicts a mask, which is applied to the
ORIGINAL pixels (never resized), saved as lossless PNG.
"""
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps
from rembg import new_session, remove

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

MODELS = {
    "Best quality (BiRefNet) - any background, fine edges": "birefnet-general",
    "Balanced (IS-Net) - fast, good for most images": "isnet-general-use",
    "People / portraits (BiRefNet Portrait)": "birefnet-portrait",
    "Fastest (U2Net)": "u2net",
}
_sessions = {}


def get_session(model_id):
    if model_id not in _sessions:
        _sessions[model_id] = new_session(model_id)
    return _sessions[model_id]


def process(files, model_label, refine, bg_color, progress=gr.Progress()):
    if not files:
        raise gr.Error("Upload at least one image.")
    session = get_session(MODELS[model_label])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    for i, f in enumerate(files):
        progress((i, len(files)), desc=f"Processing {i + 1}/{len(files)}")
        path = Path(f if isinstance(f, str) else f.name)
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        cut = remove(
            img,
            session=session,
            alpha_matting=refine,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )
        suffix = ""
        if bg_color and bg_color.lower() not in ("transparent", ""):
            base = Image.new("RGBA", cut.size, bg_color)
            base.alpha_composite(cut)
            cut, suffix = base.convert("RGB"), "_solid"
        ext = ".jpg" if suffix else ".png"
        out = OUT_DIR / f"{path.stem}_nobg{suffix}_{stamp}{ext}"
        if ext == ".png":
            cut.save(out, "PNG", compress_level=3)
        else:
            cut.save(out, "JPEG", quality=100, subsampling=0)
        results.append(str(out))
    downloads = list(results)
    if len(results) > 1:
        zpath = OUT_DIR / f"batch_{stamp}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for r in results:
                z.write(r, Path(r).name)
        downloads.append(str(zpath))
    return results, downloads


blue = gr.themes.Color(
    c50="#EEF8FD", c100="#D6EFFB", c200="#B0E0F5", c300="#71C5E8",
    c400="#4A9FEA", c500="#307FE2", c600="#0857C3", c700="#06459C",
    c800="#042A61", c900="#031F49", c950="#021633",
)
theme = gr.themes.Soft(
    primary_hue=blue, font=[gr.themes.GoogleFont("Inter"), "Aptos", "sans-serif"]
)

with gr.Blocks(title="Background Remover", theme=theme) as demo:
    gr.Markdown("# Background Remover\nLocal and private. Output keeps original resolution (PNG with transparency).")
    with gr.Row():
        with gr.Column():
            files = gr.File(
                label="Drop images here (single or batch)",
                file_count="multiple",
                file_types=["image"],
            )
            model = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="Mode")
            refine = gr.Checkbox(
                label="Edge refinement (hair, fur, soft edges; slower on large images)",
                value=False,
            )
            bg = gr.ColorPicker(label="Fill background with color (optional)", value=None)
            run = gr.Button("Remove background", variant="primary")
        with gr.Column():
            gallery = gr.Gallery(label="Result", columns=2, object_fit="contain", height=480)
            dl = gr.File(label="Download", file_count="multiple")
    run.click(process, [files, model, refine, bg], [gallery, dl])
    gr.Markdown(f"Saved to: `{OUT_DIR}`")

if __name__ == "__main__":
    demo.launch(inbrowser=True)
