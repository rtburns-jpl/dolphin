"""Stitch burst interferograms (optional) and unwrap them."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from dolphin import stitching
from dolphin._log import get_log, log_runtime
from dolphin._types import Bbox
from dolphin.interferogram import estimate_interferometric_correlations

from .config import OutputOptions

logger = get_log(__name__)


@log_runtime
def run(
    ifg_file_list: Sequence[Path],
    temp_coh_file_list: Sequence[Path],
    ps_file_list: Sequence[Path],
    stitched_ifg_dir: Path,
    output_options: OutputOptions,
    file_date_fmt: str = "%Y%m%d",
    corr_window_size: tuple[int, int] = (11, 11),
) -> tuple[list[Path], list[Path], Path, Path]:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    ifg_file_list : Sequence[Path]
        Sequence of burst-wise interferograms files to stitch.
    temp_coh_file_list : Sequence[Path]
        Sequence of paths to the burst-wise temporal coherence files.
    ps_file_list : Sequence[Path]
        Sequence of paths to the (looked) burst-wise ps mask files.
    stitched_ifg_dir : Path
        Location to store the output stitched ifgs and correlations
    output_options : OutputOptions
        [`UnwrapWorkflow`][dolphin.workflows.config.OutputOptions] object
        for with parameters for the input/output options
    file_date_fmt : str
        Format of dates contained in filenames.
        default = "%Y%m%d"
    corr_window_size : tuple[int, int]
        Size of moving window (rows, cols) to use for estimating correlation.
        Default = (11, 11)

    Returns
    -------
    stitched_ifg_paths : list[Path]
        list of Paths to the stitched interferograms.
    interferometric_corr_paths : list[Path]
        list of Paths to interferometric correlation files created.
    stitched_temp_coh_file : Path
        Path to temporal correlation file created.
    stitched_ps_file : Path
        Path to ps mask file created.

    """
    stitched_ifg_dir.mkdir(exist_ok=True, parents=True)
    # Also preps for snaphu, which needs binary format with no nans
    logger.info("Stitching interferograms by date.")
    out_bounds = Bbox(*output_options.bounds) if output_options.bounds else None
    date_to_ifg_path = stitching.merge_by_date(
        image_file_list=ifg_file_list,
        file_date_fmt=file_date_fmt,
        output_dir=stitched_ifg_dir,
        output_suffix=".int.tif",
        driver="GTiff",
        out_bounds=out_bounds,
        out_bounds_epsg=output_options.bounds_epsg,
    )
    stitched_ifg_paths = list(date_to_ifg_path.values())

    # Estimate the interferometric correlation from the stitched interferogram
    interferometric_corr_paths = estimate_interferometric_correlations(
        stitched_ifg_paths, window_size=corr_window_size
    )

    # Stitch the correlation files
    stitched_temp_coh_file = stitched_ifg_dir / "temporal_coherence.tif"
    stitching.merge_images(
        temp_coh_file_list,
        outfile=stitched_temp_coh_file,
        driver="GTiff",
        out_bounds=out_bounds,
        out_bounds_epsg=output_options.bounds_epsg,
    )

    # Stitch the looked PS files
    stitched_ps_file = stitched_ifg_dir / "ps_mask_looked.tif"
    stitching.merge_images(
        ps_file_list,
        outfile=stitched_ps_file,
        out_nodata=255,
        driver="GTiff",
        resample_alg="nearest",
        out_bounds=out_bounds,
        out_bounds_epsg=output_options.bounds_epsg,
    )

    return (
        stitched_ifg_paths,
        interferometric_corr_paths,
        stitched_temp_coh_file,
        stitched_ps_file,
    )
