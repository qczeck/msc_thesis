"""RoPART helper functions.

Early-stage, deliberately small utilities. ``sampling`` ports the off-grid,
constant-size patch-sampling path from upstream PART; ``viz`` provides matplotlib
helpers to see what it produces.
"""

from helpers.sampling import (
    crop_patches,
    load_image,
    retile_patches,
    sample_offgrid_patches,
)
from helpers.targets import (
    check_translation_invariants,
    patch_positions,
    relative_translation,
)
from helpers.viz import (
    draw_boxes,
    show_image,
    show_pair,
    show_patch_grid,
    show_sampling_overview,
)

__all__ = [
    "crop_patches",
    "load_image",
    "retile_patches",
    "sample_offgrid_patches",
    "check_translation_invariants",
    "patch_positions",
    "relative_translation",
    "draw_boxes",
    "show_image",
    "show_pair",
    "show_patch_grid",
    "show_sampling_overview",
]
