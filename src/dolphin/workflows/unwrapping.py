"""Stitch burst interferograms (optional) and unwrap them."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from dolphin import io, stitching, unwrap
from dolphin._log import get_log, log_runtime
from dolphin._types import PathOrStr

from .config import UnwrapOptions

logger = get_log(__name__)


@log_runtime
def run(
    ifg_file_list: Sequence[Path],
    cor_file_list: Sequence[Path],
    nlooks: float,
    unwrap_options: UnwrapOptions,
    mask_file: PathOrStr | None = None,
) -> tuple[list[Path], list[Path]]:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    ifg_file_list : Sequence[Path]
        Sequence interferograms files to unwrap.
    cor_file_list : Sequence[Path]
        Sequence interferometric correlation files, one per file in `ifg_file_list`
    nlooks : float
        Effective number of looks used to form the input correlation data.
    unwrap_options : UnwrapOptions
        [`UnwrapOptions`][dolphin.workflows.config.UnwrapOptions] config object
        with parameters for running unwrapping jobs.
    mask_file : PathOrStr, optional
        Path to boolean mask indicating nodata areas.
        1 indicates valid data, 0 indicates missing data.

    Returns
    -------
    unwrapped_paths : list[Path]
        list of Paths to unwrapped interferograms created.
    conncomp_paths : list[Path]
        list of Paths to connected component files created.

    """
    if len(ifg_file_list) != len(cor_file_list):
        msg = f"{len(ifg_file_list) = } != {len(cor_file_list) = }"
        raise ValueError(msg)

    output_path = unwrap_options._directory
    output_path.mkdir(exist_ok=True, parents=True)
    if mask_file is not None:
        output_mask = _get_matching_mask(
            mask_file=mask_file,
            output_dir=output_path,
            match_file=ifg_file_list[0],
        )
    else:
        output_mask = None

    logger.info(f"Unwrapping {len(ifg_file_list)} interferograms")

    # Make a scratch directory for unwrapping
    unwrap_scratchdir = unwrap_options._directory / "scratch"
    unwrap_scratchdir.mkdir(exist_ok=True, parents=True)

    unwrapped_paths, conncomp_paths = unwrap.run(
        ifg_filenames=ifg_file_list,
        cor_filenames=cor_file_list,
        output_path=output_path,
        nlooks=nlooks,
        mask_file=output_mask,
        max_jobs=unwrap_options.n_parallel_jobs,
        ntiles=unwrap_options.ntiles,
        downsample_factor=unwrap_options.downsample_factor,
        unwrap_method=unwrap_options.unwrap_method,
        scratchdir=unwrap_scratchdir,
    )

    return (unwrapped_paths, conncomp_paths)


def _get_matching_mask(
    mask_file: PathOrStr, output_dir: Path, match_file: PathOrStr
) -> Path:
    """Create a mask with the same size/projection as `match_file`."""
    # Check that the input mask is the same size as the ifgs:
    if io.get_raster_xysize(mask_file) == match_file:
        logger.info(f"Using {mask_file} to mask during unwrapping")
        output_mask = Path(mask_file)
    else:
        logger.info(f"Warping {mask_file} to match size of interferograms")
        output_mask = output_dir / "warped_mask.tif"
        if output_mask.exists():
            logger.info(f"Mask already exists at {output_mask}")
        else:
            stitching.warp_to_match(
                input_file=mask_file,
                match_file=match_file,
                output_file=output_mask,
            )
    return output_mask
