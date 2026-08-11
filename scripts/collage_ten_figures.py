#!/usr/bin/env python3
"""Honest 2x2 A–D collages for PanODE-Topic ten-figures pass.

No fabricated data: only crops / rearrangements of verified PNG/PDF assets.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "benchmarks" / "paper_figures" / "topic"
FIG6 = TOPIC / "fig6"
EXP = ROOT / "experiments" / "results" / "topic"
OUT = TOPIC / "tenfig_labeled"
PAD = 12
LABEL_PAD = 8


def _font(size: int = 56) -> ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ):
        p = Path(name)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def _open(path: Path) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        # Rasterize first page via pdftoppm-equivalent: Pillow cannot read PDF;
        # use temporary PNG sibling or convert with subprocess.
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            prefix = Path(td) / "page"
            subprocess.check_call(
                ["pdftoppm", "-png", "-r", "150", "-singlefile", str(path), str(prefix)]
            )
            png = prefix.with_suffix(".png")
            return Image.open(png).convert("RGB")
    return Image.open(path).convert("RGB")


def _fit(im: Image.Image, box: tuple[int, int]) -> Image.Image:
    bw, bh = box
    im = im.copy()
    im.thumbnail((bw, bh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (bw, bh), (255, 255, 255))
    x = (bw - im.width) // 2
    y = (bh - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def _label(im: Image.Image, letter: str, font: ImageFont.ImageFont) -> Image.Image:
    draw = ImageDraw.Draw(im)
    text = letter
    # white halo + black letter
    x, y = LABEL_PAD, LABEL_PAD
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
        draw.text((x + dx, y + dy), text, font=font, fill=(255, 255, 255))
    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    return im


def collage_2x2(
    panels: list[tuple[str, Image.Image]],
    out_path: Path,
    cell: tuple[int, int] = (1400, 1050),
) -> None:
    assert len(panels) == 4
    font = _font(64)
    cw, ch = cell
    W = PAD * 3 + cw * 2
    H = PAD * 3 + ch * 2
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    positions = [
        (PAD, PAD),
        (PAD * 2 + cw, PAD),
        (PAD, PAD * 2 + ch),
        (PAD * 2 + cw, PAD * 2 + ch),
    ]
    for (letter, im), (x, y) in zip(panels, positions):
        tile = _fit(im, (cw, ch))
        _label(tile, letter, font)
        canvas.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)
    pdf_path = out_path.with_suffix(".pdf")
    canvas.save(pdf_path, "PDF", resolution=150.0)
    print(f"wrote {out_path} + {pdf_path.name} ({canvas.size[0]}x{canvas.size[1]})")


def crop_rows(im: Image.Image, n_rows: int, which: slice) -> Image.Image:
    """Crop a contiguous row-block assuming equal-height rows."""
    h = im.height // n_rows
    y0 = which.start * h
    y1 = which.stop * h
    # last slice may absorb remainder
    if which.stop == n_rows:
        y1 = im.height
    return im.crop((0, y0, im.width, y1))


def crop_frac(
    im: Image.Image,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> Image.Image:
    w, h = im.size
    return im.crop(
        (int(left * w), int(top * h), int(right * w), int(bottom * h))
    )


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- fig:bio — verified fig6 subpanels ---
    collage_2x2(
        [
            ("A", _open(FIG6 / "heat_setty_Topic-FM-Base.png")),
            ("B", _open(FIG6 / "heat_setty_Topic-FM-Transformer.png")),
            ("C", _open(FIG6 / "beta_setty_Topic-FM-Base.png")),
            ("D", _open(FIG6 / "beta_setty_Topic-FM-Contrastive.png")),
        ],
        OUT / "Fig6_biological_topic_4panel.png",
    )

    # --- fig:sensitivity — four row-blocks of Fig3 (9 HP rows) ---
    sens = _open(TOPIC / "Fig3_sensitivity_topic.png")
    collage_2x2(
        [
            ("A", crop_rows(sens, 9, slice(0, 2))),   # kl_weight, latent_dim
            ("B", crop_rows(sens, 9, slice(2, 4))),   # encoder_size, dropout
            ("C", crop_rows(sens, 9, slice(4, 6))),   # lr, epochs
            ("D", crop_rows(sens, 9, slice(6, 9))),   # batch, wd, hvg
        ],
        OUT / "Fig3_sensitivity_topic.png",
        cell=(1500, 1100),
    )

    # --- fig:ablation — four suite panels from ablation figures ---
    abl = EXP / "ablation" / "figures"
    collage_2x2(
        [
            ("A", _open(abl / "clustering.png")),
            ("B", _open(abl / "dre_umap.png")),
            ("C", _open(abl / "dre_tsne.png")),
            ("D", _open(abl / "lse_intrinsic.png")),
        ],
        OUT / "Fig_ablation_topic_4panel.png",
        cell=(1600, 900),
    )

    # --- fig:external — proposed / classical / deep / latent ---
    vex = EXP / "vs_external" / "figures"
    collage_2x2(
        [
            ("A", _open(vex / "proposed" / "composed_metrics.png")),
            ("B", _open(vex / "classical" / "composed_metrics.png")),
            ("C", _open(vex / "deep" / "composed_metrics.png")),
            ("D", _open(vex / "proposed" / "composed_latent_structure.png")),
        ],
        OUT / "Fig10_external_topic_4panel.png",
        cell=(1400, 1100),
    )

    # --- fig:training — keep Fig4 (already A–J); copy with PDF sibling ---
    train = _open(TOPIC / "Fig4_training_topic.png")
    train_out = OUT / "Fig4_training_topic.png"
    train.save(train_out, optimize=True)
    train.save(train_out.with_suffix(".pdf"), "PDF", resolution=150.0)
    print(f"copied {train_out}")

    # --- fig:correlation — crop Fig7 into workflow + 3 dataset rows ---
    corr = _open(TOPIC / "Fig7_correlation_topic.png")
    # Empirically: workflow ~ top 18%; heatmap grid ~ 18%–100% in 3 equal rows
    collage_2x2(
        [
            ("A", crop_frac(corr, 0.0, 0.0, 1.0, 0.20)),
            ("B", crop_frac(corr, 0.0, 0.20, 1.0, 0.47)),  # setty
            ("C", crop_frac(corr, 0.0, 0.47, 1.0, 0.74)),  # endo
            ("D", crop_frac(corr, 0.0, 0.74, 1.0, 1.0)),   # dentate
        ],
        OUT / "Fig7_correlation_topic.png",
        cell=(1500, 900),
    )

    # --- fig:umap — crop Fig8 into workflow + first three dataset blocks ---
    umap = _open(TOPIC / "Fig8_umap_topic.png")
    collage_2x2(
        [
            ("A", crop_frac(umap, 0.0, 0.0, 1.0, 0.14)),          # workflow
            ("B", crop_frac(umap, 0.0, 0.14, 1.0, 0.42)),         # setty B/C
            ("C", crop_frac(umap, 0.0, 0.42, 1.0, 0.70)),         # endo D/E
            ("D", crop_frac(umap, 0.0, 0.70, 1.0, 1.0)),          # dentate F/G
        ],
        OUT / "Fig8_umap_topic.png",
        cell=(1500, 950),
    )

    # --- fig:enrichment — workflow + pert grid + two beta plots ---
    enr = _open(TOPIC / "Fig9_enrichment_topic.png")
    collage_2x2(
        [
            ("A", crop_frac(enr, 0.0, 0.0, 1.0, 0.16)),           # workflow
            ("B", crop_frac(enr, 0.0, 0.16, 1.0, 0.72)),          # pert 3x3
            ("C", crop_frac(enr, 0.0, 0.72, 0.50, 1.0)),          # beta endo
            ("D", crop_frac(enr, 0.50, 0.72, 1.0, 1.0)),          # beta dentate
        ],
        OUT / "Fig9_enrichment_topic.png",
        cell=(1500, 1000),
    )

    # --- fig:crossdataset — 4 scatters already; overlay A–D on PNG for float option ---
    cross = _open(TOPIC / "Fig5_crossdataset_topic.png")
    collage_2x2(
        [
            ("A", crop_frac(cross, 0.0, 0.0, 0.5, 0.5)),
            ("B", crop_frac(cross, 0.5, 0.0, 1.0, 0.5)),
            ("C", crop_frac(cross, 0.0, 0.5, 0.5, 1.0)),
            ("D", crop_frac(cross, 0.5, 0.5, 1.0, 1.0)),
        ],
        OUT / "Fig5_crossdataset_topic_4panel.png",
        cell=(1200, 900),
    )

    # Manifest
    lines = ["# tenfig_labeled manifest\n"]
    for p in sorted(OUT.glob("*.png")) + sorted(OUT.glob("*.pdf")):
        import hashlib

        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{p.name}\t{p.stat().st_size}\t{h}\n")
    (OUT / "MANIFEST.tsv").write_text("".join(lines))
    print("manifest written")


if __name__ == "__main__":
    build()
