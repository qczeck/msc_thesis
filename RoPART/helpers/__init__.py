"""RoPART helper functions.

Early-stage, deliberately small utilities. ``sampling`` ports the off-grid,
constant-size patch-sampling path from upstream PART (plus the RoPART rotation
extension); ``targets`` builds the relative pretext targets; ``viz`` provides
matplotlib helpers to see what they produce.
"""

from helpers.sampling import (
    crop_patches,
    crop_patches_rotated,
    gaussian_blur_patches,
    load_image,
    retile_patches,
    rotation_margin,
    rotation_window_size,
    sample_offgrid_patches,
    sample_rotation_angles,
)
from helpers.targets import (
    check_orientation_invariants,
    check_translation_invariants,
    patch_positions,
    relative_orientation,
    relative_translation,
)
from helpers.viz import (
    draw_boxes,
    draw_rotated_boxes,
    show_image,
    show_orientation_pair,
    show_pair,
    show_patch_grid,
    show_rotation_sweep,
    show_sampling_overview,
)

__all__ = [
    # sampling
    "load_image",
    "sample_offgrid_patches",
    "crop_patches",
    "retile_patches",
    "rotation_window_size",
    "rotation_margin",
    "sample_rotation_angles",
    "crop_patches_rotated",
    "gaussian_blur_patches",
    # targets
    "patch_positions",
    "relative_translation",
    "check_translation_invariants",
    "relative_orientation",
    "check_orientation_invariants",
    # viz
    "draw_boxes",
    "show_image",
    "show_patch_grid",
    "show_pair",
    "show_sampling_overview",
    "draw_rotated_boxes",
    "show_rotation_sweep",
    "show_orientation_pair",
]
