from __future__ import annotations

from pathlib import Path

from dolphin._log import get_log
from dolphin._types import Filename
from dolphin.utils import full_suffix

from ._constants import CONNCOMP_SUFFIX
from ._utils import _zero_from_mask

logger = get_log(__name__)


def unwrap_snaphu_py(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    nlooks: float,
    ntiles: tuple[int, int] = (1, 1),
    nproc: int = 1,
    tile_overlap: tuple[int, int] = (0, 0),
    mask_file: Filename | None = None,
    zero_where_masked: bool = True,
    nodata: str | float | None = None,
    init_method: str = "mst",
    scratchdir: Filename | None = None,
) -> tuple[Path, Path]:
    """Unwrap an interferogram using at multiple scales using `tophu`.

    Parameters
    ----------
    ifg_filename : Filename
        Path to input interferogram.
    corr_filename : Filename
        Path to input correlation file.
    unw_filename : Filename
        Path to output unwrapped phase file.
    downsample_factor : tuple[int, int]
        Downsample the interferograms by this factor to unwrap faster, then upsample
    nlooks : float
        Effective number of looks used to form the input correlation data.
    ntiles : tuple[int, int], optional
        Number of (row, column) tiles to split for full image into.
        If `ntiles` is an int, will use `(ntiles, ntiles)`
    tile_overlap : tuple[int, int], optional
        Number of pixels to overlap in the (row, col) direction.
        Default = (0, 0)
    nproc : int, optional
        If specifying `ntiles`, number of processes to spawn to unwrap the
        tiles in parallel.
        Default = 1, which unwraps each tile in serial.
    mask_file : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    zero_where_masked : bool, optional
        Set wrapped phase/correlation to 0 where mask is 0 before unwrapping.
        If not mask is provided, this is ignored.
        By default True.
    nodata : float | str, optional.
        If providing `unwrap_callback`, provide the nodata value for your
        unwrapping function.
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mst"
    scratchdir : Filename, optional
        If provided, uses a scratch directory to save the intermediate files
        during unwrapping.

    Returns
    -------
    unw_path : Path
        Path to output unwrapped phase file.
    conncomp_path : Path
        Path to output connected component label file.

    """
    import snaphu

    unw_suffix = full_suffix(unw_filename)
    cc_filename = str(unw_filename).replace(unw_suffix, CONNCOMP_SUFFIX)

    if zero_where_masked and mask_file is not None:
        logger.info(f"Zeroing phase/corr of pixels masked in {mask_file}")
        zeroed_ifg_file, zeroed_corr_file = _zero_from_mask(
            ifg_filename, corr_filename, mask_file
        )
        igram = snaphu.io.Raster(zeroed_ifg_file)
        corr = snaphu.io.Raster(zeroed_corr_file)
    else:
        igram = snaphu.io.Raster(ifg_filename)
        corr = snaphu.io.Raster(corr_filename)

    mask = None if mask_file is None else snaphu.io.Raster(mask_file)
    try:
        with (
            snaphu.io.Raster.create(
                unw_filename, like=igram, nodata=nodata, dtype="f4"
            ) as unw,
            snaphu.io.Raster.create(
                cc_filename, like=igram, nodata=nodata, dtype="u4"
            ) as conncomp,
        ):
            # Unwrap and store the results in the `unw` and `conncomp` rasters.
            snaphu.unwrap(
                igram,
                corr,
                nlooks=nlooks,
                init=init_method,
                mask=mask,
                unw=unw,
                conncomp=conncomp,
                ntiles=ntiles,
                tile_overlap=tile_overlap,
                nproc=nproc,
                scratchdir=scratchdir,
            )
    finally:
        igram.close()
        corr.close()
        if mask is not None:
            mask.close()

    return Path(unw_filename), Path(cc_filename)
