"""
src/data/augmentation.py

Albumentations pipeline that simulates real-world receipt scanning noise.
Applied only to the training split.
"""
import albumentations as A
from albumentations.core.composition import Compose


def build_augmentation_pipeline(p: float = 0.8) -> Compose:
    """
    Returns an Albumentations pipeline that mimics receipt scan degradation.

    Each transform is independently probabilistic so augmented images
    vary naturally rather than all receiving the same set of distortions.

    Args:
        p: Overall pipeline probability (default 0.8 — 20% stay clean).
    """
    return A.Compose(
        [
            # ── Geometric ──────────────────────────────────────────────────
            A.Rotate(limit=5, border_mode=0, p=0.5),          # slight tilt
            A.Perspective(scale=(0.01, 0.04), p=0.3),         # minor warp

            # ── Blur / sharpness ───────────────────────────────────────────
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0), # out-of-focus
                    A.MotionBlur(blur_limit=5, p=1.0),        # camera shake
                ],
                p=0.3,
            ),

            # ── Lighting & contrast ────────────────────────────────────────
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2), p=0.2),

            # ── Noise ──────────────────────────────────────────────────────
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),

            # ── JPEG compression artifact ──────────────────────────────────
            # Simulates phone camera saves and re-scans
            A.ImageCompression(quality_range=(65, 90), p=0.4),

            # ── Colour shift (receipt paper aging) ────────────────────────
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.3
            ),
        ],
        p=p,
    )
