"""Estimate wrapped phase for one ministack of SLCs.

References
----------
    .. [1] Mirzaee, Sara, Falk Amelung, and Heresh Fattahi. "Non-linear phase
    linking using joined distributed and persistent scatterers." Computers &
    Geosciences (2022): 105291.


"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import DTypeLike

from dolphin import io, shp
from dolphin._log import get_log
from dolphin._types import Filename, HalfWindow, Strides
from dolphin.io import EagerLoader, StridedBlockManager, VRTStack
from dolphin.phase_link import PhaseLinkRuntimeError, compress, run_phase_linking
from dolphin.stack import MiniStackInfo

from .config import ShpMethod

logger = get_log(__name__)

__all__ = ["run_wrapped_phase_single"]


@dataclass
class OutputFile:
    filename: Path
    dtype: DTypeLike
    strides: Optional[dict[str, int]] = None


# @atomic_output(output_arg="output_folder", is_dir=True)
def run_wrapped_phase_single(
    *,
    slc_vrt_file: Filename,
    ministack: MiniStackInfo,
    output_folder: Filename,
    half_window: dict,
    strides: Optional[dict] = None,
    reference_idx: int = 0,
    beta: float = 0.01,
    use_evd: bool = False,
    mask_file: Optional[Filename] = None,
    ps_mask_file: Optional[Filename] = None,
    amp_mean_file: Optional[Filename] = None,
    amp_dispersion_file: Optional[Filename] = None,
    shp_method: ShpMethod = ShpMethod.NONE,
    shp_alpha: float = 0.05,
    shp_nslc: Optional[int] = None,
    block_shape: tuple[int, int] = (1024, 1024),
    # show_progress: bool = False,
):
    """Estimate wrapped phase for one ministack.

    Output files will all be placed in the provided `output_folder`.
    """
    # TODO: extract common stuff between here and sequential
    if strides is None:
        strides = {"x": 1, "y": 1}
    strides_tup = Strides(y=strides["y"], x=strides["x"])
    half_window_tup = HalfWindow(y=half_window["y"], x=half_window["x"])
    output_folder = Path(output_folder)
    vrt = VRTStack.from_vrt_file(slc_vrt_file)
    input_slc_files = ministack.file_list
    assert len(input_slc_files) == vrt.shape[0]

    # If we are using a different number of SLCs for the amplitude data,
    # we should note that for the SHP finding algorithms
    if shp_nslc is None:
        shp_nslc = len(input_slc_files)

    logger.info(f"{vrt}: from {ministack.dates[0]} {ministack.file_list[-1]}")

    nrows, ncols = vrt.shape[-2:]

    nodata_mask = _get_nodata_mask(mask_file, nrows, ncols)
    ps_mask = _get_ps_mask(ps_mask_file, nrows, ncols)
    amp_mean, amp_variance = _get_amp_mean_variance(amp_mean_file, amp_dispersion_file)
    amp_stack: Optional[np.ndarray] = None

    xhalf, yhalf = half_window["x"], half_window["y"]

    # If we were passed any compressed SLCs in `input_slc_files`,
    # then we want that index for when we create new compressed SLCs.
    # We skip the old compressed SLCs to create new ones
    first_real_slc_idx = ministack.first_real_slc_idx

    msg = (
        f"Processing {len(input_slc_files) - first_real_slc_idx} SLCs +"
        f" {first_real_slc_idx} compressed SLCs. "
    )
    logger.info(msg)

    # Create the background writer for this ministack
    writer = io.GdalWriter()

    logger.info(f"Total stack size (in pixels): {vrt.shape}")
    # Set up the output folder with empty files to write into
    phase_linked_slc_files = setup_output_folder(
        ministack=ministack,
        driver="GTiff",
        strides=strides,
        output_folder=output_folder,
        like_filename=vrt.outfile,
        nodata=0,
    )

    comp_slc_info = ministack.get_compressed_slc_info()

    # Use the real-SLC date range for output file naming
    start_end = ministack.real_slc_date_range_str
    output_files: list[OutputFile] = [
        OutputFile(output_folder / comp_slc_info.filename, np.complex64),
        OutputFile(
            output_folder / f"temporal_coherence_{start_end}.tif", np.float32, strides
        ),
        OutputFile(output_folder / f"avg_coh_{start_end}.tif", np.uint16, strides),
        OutputFile(output_folder / f"eigenvalues_{start_end}.tif", np.float32, strides),
        OutputFile(output_folder / f"shp_counts_{start_end}.tif", np.uint16, strides),
    ]
    for op in output_files:
        io.write_arr(
            arr=None,
            like_filename=vrt.outfile,
            output_name=op.filename,
            dtype=op.dtype,
            strides=op.strides,
            nbands=1,
            nodata=0,
        )

    # Iterate over the output grid
    block_manager = StridedBlockManager(
        arr_shape=(nrows, ncols),
        block_shape=block_shape,
        strides=strides_tup,
        half_window=half_window_tup,
    )
    # Set up the background loader
    loader = EagerLoader(reader=vrt, block_shape=block_shape)
    # Queue all input slices, skip ones that are all nodata
    blocks = []
    # Queue all input slices, skip ones that are all nodata
    for b in block_manager.iter_blocks():
        in_rows, in_cols = b[2]
        if nodata_mask[in_rows, in_cols].all():
            continue
        loader.queue_read(in_rows, in_cols)
        blocks.append(b)

    logger.info(f"Iterating over {block_shape} blocks, {len(blocks)} total")
    for (
        (out_rows, out_cols),
        (out_trim_rows, out_trim_cols),
        (in_rows, in_cols),
        (in_no_pad_rows, in_no_pad_cols),
        (in_trim_rows, in_trim_cols),
    ) in blocks:
        logger.debug(f"{out_rows = }, {out_cols = }, {in_rows = }, {in_no_pad_rows = }")

        cur_data, (read_rows, read_cols) = loader.get_data()
        if np.all(cur_data == 0) or np.isnan(cur_data).all():
            continue
        assert read_rows == in_rows and read_cols == in_cols

        cur_data = cur_data.astype(np.complex64)

        if shp_method == "ks":
            # Only actually compute if we need this one
            amp_stack = np.abs(cur_data)

        # Compute the neighbor_arrays for this block
        neighbor_arrays = shp.estimate_neighbors(
            halfwin_rowcol=(yhalf, xhalf),
            alpha=shp_alpha,
            strides=strides,
            mean=amp_mean[in_rows, in_cols] if amp_mean is not None else None,
            var=amp_variance[in_rows, in_cols] if amp_variance is not None else None,
            nslc=shp_nslc,
            amp_stack=amp_stack,
            method=shp_method,
        )
        # Run the phase linking process on the current ministack
        reference_idx = max(0, first_real_slc_idx - 1)
        try:
            pl_output = run_phase_linking(
                cur_data,
                half_window=half_window_tup,
                strides=strides_tup,
                use_evd=use_evd,
                beta=beta,
                reference_idx=reference_idx,
                nodata_mask=nodata_mask[in_rows, in_cols],
                ps_mask=ps_mask[in_rows, in_cols],
                neighbor_arrays=neighbor_arrays,
                avg_mag=amp_mean[in_rows, in_cols] if amp_mean is not None else None,
            )
        except PhaseLinkRuntimeError as e:
            # note: this is a warning instead of info, since it should
            # get caught at the "skip_empty" step
            msg = f"At block {in_rows.start}, {in_cols.start}: {e}"
            if "are all NaNs" in e.args[0]:
                # Some SLCs in the ministack are all NaNs
                # This happens from a shifting burst window near the edges,
                # and seems to cause no issues
                logger.debug(msg)
            else:
                logger.warning(msg)
            continue

        # Fill in the nan values with 0
        np.nan_to_num(pl_output.cpx_phase, copy=False)
        np.nan_to_num(pl_output.temp_coh, copy=False)

        # Save each of the MLE estimates (ignoring those corresponding to
        # compressed SLCs indexes)
        assert len(pl_output.cpx_phase[first_real_slc_idx:]) == len(
            phase_linked_slc_files
        )

        for img, f in zip(
            pl_output.cpx_phase[first_real_slc_idx:, out_trim_rows, out_trim_cols],
            phase_linked_slc_files,
        ):
            writer.queue_write(img, f, out_rows.start, out_cols.start)

        # Get the SHP counts for each pixel (if not using Rect window)
        if neighbor_arrays is None:
            shp_counts = np.zeros(pl_output.cpx_phase.shape[-2:], dtype=np.int16)
        else:
            shp_counts = np.sum(neighbor_arrays, axis=(-2, -1))

        # Compress the ministack using only the non-compressed SLCs
        cur_comp_slc = compress(
            # Get the inner portion of the full-res SLC data
            cur_data[first_real_slc_idx:, in_trim_rows, in_trim_cols],
            pl_output.cpx_phase[first_real_slc_idx:, out_trim_rows, out_trim_cols],
        )

        # ### Save results ###

        # Save the compressed SLC block
        writer.queue_write(
            cur_comp_slc,
            output_files[0].filename,
            in_no_pad_rows.start,
            in_no_pad_cols.start,
        )

        # All other outputs are strided (smaller in size)
        out_datas = [pl_output.temp_coh, pl_output.avg_coh, shp_counts]
        for data, output_file in zip(out_datas, output_files[1:]):
            if data is None:  # May choose to skip some outputs, e.g. "avg_coh"
                continue
            writer.queue_write(
                data[out_trim_rows, out_trim_cols],
                output_file.filename,
                out_rows.start,
                out_cols.start,
            )

    loader.notify_finished()
    # Block until all the writers for this ministack have finished
    logger.info(f"Waiting to write {writer.num_queued} blocks of data.")
    writer.notify_finished()
    logger.info(f"Finished ministack of size {vrt.shape}.")

    written_comp_slc = output_files[0]

    ccslc_info = ministack.get_compressed_slc_info()
    ccslc_info.write_metadata(output_file=written_comp_slc.filename)
    # TODO: Does it make sense to return anything from this?
    # or just allow user to search through the `output_folder` they provided?


def _get_nodata_mask(
    mask_file: Optional[Filename],
    nrows: int,
    ncols: int,
) -> np.ndarray:
    if mask_file is not None:
        # The mask file will by -2s at invalid data, 1s at good
        nodata_mask = io.load_gdal(mask_file, masked=True).astype(bool).filled(False)
        # invert the mask so -1s are the missing data pixels
        nodata_mask = ~nodata_mask
        # check middle pixel
        if nodata_mask[nrows // 2, ncols // 2]:
            logger.warning(f"{mask_file} is True at {nrows//2, ncols//2}")
            logger.warning("Proceeding without the nodata mask.")
            nodata_mask = np.zeros((nrows, ncols), dtype=bool)
    else:
        nodata_mask = np.zeros((nrows, ncols), dtype=bool)
    return nodata_mask


def _get_ps_mask(
    ps_mask_file: Optional[Filename], nrows: int, ncols: int
) -> np.ndarray:
    if ps_mask_file is not None:
        ps_mask = io.load_gdal(ps_mask_file, masked=True)
        # Fill the nodata values with false
        ps_mask = ps_mask.astype(bool).filled(False)
    else:
        ps_mask = np.zeros((nrows, ncols), dtype=bool)
    return ps_mask


def _get_amp_mean_variance(
    amp_mean_file: Optional[Filename],
    amp_dispersion_file: Optional[Filename],
) -> tuple[np.ndarray, np.ndarray]:
    if amp_mean_file is not None and amp_dispersion_file is not None:
        # Note: have to fill, since numba (as of 0.57) can't do masked arrays
        amp_mean = io.load_gdal(amp_mean_file, masked=True).filled(np.nan)
        amp_dispersion = io.load_gdal(amp_dispersion_file, masked=True).filled(np.nan)
        # convert back to variance from dispersion: amp_disp = std_dev / mean
        amp_variance = (amp_dispersion * amp_mean) ** 2
    else:
        amp_mean = amp_variance = None

    return amp_mean, amp_variance


def setup_output_folder(
    ministack: MiniStackInfo,
    driver: str = "GTiff",
    dtype="complex64",
    like_filename: Optional[Filename] = None,
    strides: Optional[dict[str, int]] = None,
    nodata: Optional[float] = 0,
    output_folder: Optional[Path] = None,
) -> list[Path]:
    """Create empty output files for each band after `start_idx` in `vrt_stack`.

    Also creates an empty file for the compressed SLC.
    Used to prepare output for block processing.

    Parameters
    ----------
    ministack : MiniStackInfo
        [dolphin.stack.MiniStackInfo][] object for the current batch of SLCs
    driver : str, optional
        Name of GDAL driver, by default "GTiff"
    dtype : str, optional
        Numpy datatype of output files, by default "complex64"
    like_filename : Filename, optional
        Filename to use for getting the shape/GDAL metadata of the output files.
        If None, will use the first SLC in `vrt_stack`
    strides : dict[str, int], optional
        Strides to use when creating the empty files, by default {"y": 1, "x": 1}
        Larger strides will create smaller output files, computed using
        [dolphin.io.compute_out_shape][]
    nodata : float, optional
        Nodata value to use for the output files, by default 0.
    output_folder : Path, optional
        Path to output folder, by default None
        If None, will use the same folder as the first SLC in `vrt_stack`

    Returns
    -------
    list[Path]
        list of saved empty files for the outputs of phase linking

    """
    if strides is None:
        strides = {"y": 1, "x": 1}
    if output_folder is None:
        output_folder = ministack.output_folder
    # Note: DONT use the ministack.output_folder here, since that will
    # be the tempdir made by @atomic_output
    # # output_folder = Path(ministack.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    start_idx = ministack.first_real_slc_idx
    date_strs = ministack.get_date_str_list()[start_idx:]

    phase_linked_slc_files = []
    for filename in date_strs:
        slc_name = Path(filename).stem
        output_path = output_folder / f"{slc_name}.slc.tif"

        io.write_arr(
            arr=None,
            like_filename=like_filename,
            output_name=output_path,
            driver=driver,
            nbands=1,
            dtype=dtype,
            strides=strides,
            nodata=nodata,
        )

        phase_linked_slc_files.append(output_path)
    return phase_linked_slc_files
