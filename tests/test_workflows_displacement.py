from __future__ import annotations

import os
from pathlib import Path

import pytest
from opera_utils import group_by_burst

from dolphin.utils import flatten
from dolphin.workflows import config, displacement

pytestmark = pytest.mark.filterwarnings(
    "ignore::rasterio.errors.NotGeoreferencedWarning",
    "ignore:.*io.FileIO.*:pytest.PytestUnraisableExceptionWarning",
)


def test_displacement_run_single(
    opera_slc_files: list[Path],
    tmpdir,
):
    with tmpdir.as_cwd():
        cfg = config.DisplacementWorkflow(
            cslc_file_list=opera_slc_files,
            input_options={"subdataset": "/data/VV"},
            interferogram_network={
                "network_type": config.InterferogramNetworkType.MANUAL_INDEX,
                "indexes": [(0, -1)],
            },
            phase_linking={
                "ministack_size": 500,
            },
            worker_settings={
                "gpu_enabled": (os.environ.get("NUMBA_DISABLE_JIT") != "1")
            },
        )
        displacement.run(cfg)


def test_displacement_run_single_official_opera_naming(
    opera_slc_files_official: list[Path],
    weather_model_files: list[Path],
    tec_files: list[Path],
    dem_file: Path,
    opera_static_files_official: list[Path],
    tmpdir,
):
    with tmpdir.as_cwd():
        cfg = config.DisplacementWorkflow(
            cslc_file_list=opera_slc_files_official,
            input_options={"subdataset": "/data/VV"},
            interferogram_network={
                "network_type": config.InterferogramNetworkType.MANUAL_INDEX,
                "indexes": [(0, -1)],
            },
            phase_linking={
                "ministack_size": 500,
            },
            worker_settings={
                "gpu_enabled": (os.environ.get("NUMBA_DISABLE_JIT") != "1")
            },
            # TODO: Move to a disp-s1 test
            correction_options={
                "troposphere_files": weather_model_files,
                "ionosphere_files": tec_files,
                "dem_file": dem_file,
                "geometry_files": opera_static_files_official,
            },
            unwrap_options={"run_unwrap": False},
        )
        displacement.run(cfg)


def run_displacement_stack(
    path, file_list: list[Path], run_unwrap: bool = False, ministack_size: int = 500
):
    cfg = config.DisplacementWorkflow(
        cslc_file_list=file_list,
        input_options={"subdataset": "/data/VV"},
        work_directory=path,
        phase_linking={
            "ministack_size": ministack_size,
        },
        worker_settings={"gpu_enabled": (os.environ.get("NUMBA_DISABLE_JIT") != "1")},
        unwrap_options={"run_unwrap": run_unwrap},
        log_file=Path() / "dolphin.log",
    )
    displacement.run(cfg)


def test_stack_with_compressed(opera_slc_files, tmpdir):
    with tmpdir.as_cwd():
        p1 = Path("first_run")
        run_displacement_stack(p1, opera_slc_files)
        # Find the compressed SLC files
        new_comp_slcs = sorted(p1.rglob("compressed_*"))

        p2 = Path("second_run")
        # Add the first compressed SLC in place of first real one and run again
        by_burst = group_by_burst(opera_slc_files)
        new_real_slcs = list(flatten(v[1:] for v in by_burst.values()))
        new_file_list = new_comp_slcs + new_real_slcs

        run_displacement_stack(p2, new_file_list)

        # Now the results should be the same (for the file names)
        # check the ifg folders
        ifgs1 = sorted((p1 / "interferograms").glob("*.int.tif"))
        ifgs2 = sorted((p2 / "interferograms").glob("*.int.tif"))
        assert len(ifgs1) > 0
        assert [f.name for f in ifgs1] == [f.name for f in ifgs2]


def test_separate_workflow_runs(slc_file_list, tmp_path):
    """Check that manually running the workflow results in the same
    interferograms as one sequential run.
    """
    p_all = tmp_path / "all"
    run_displacement_stack(p_all, slc_file_list, ministack_size=10)
    all_ifgs = sorted((p_all / "interferograms").glob("*.int.tif"))
    assert len(all_ifgs) == 29

    p1 = tmp_path / Path("first")
    ms = 10
    # Split into batches of 10
    file_batches = [slc_file_list[i : i + ms] for i in range(0, len(slc_file_list), ms)]
    assert len(file_batches) == 3
    assert all(len(b) == 10 for b in file_batches)
    run_displacement_stack(p1, file_batches[0])
    new_comp_slcs1 = sorted((p1 / "linked_phase").glob("compressed_*"))
    assert len(new_comp_slcs1) == 1
    ifgs1 = sorted((p1 / "interferograms").glob("*.int.tif"))
    assert len(ifgs1) == 9

    p2 = tmp_path / Path("second")
    files2 = new_comp_slcs1 + file_batches[1]
    run_displacement_stack(p2, files2)
    new_comp_slcs2 = sorted((p2 / "linked_phase").glob("compressed_*"))
    assert len(new_comp_slcs2) == 1
    ifgs2 = sorted((p2 / "interferograms").glob("*.int.tif"))
    assert len(ifgs2) == 10

    p3 = tmp_path / Path("third")
    files3 = new_comp_slcs1 + new_comp_slcs2 + file_batches[2]
    run_displacement_stack(p3, files3)
    ifgs3 = sorted((p3 / "interferograms").glob("*.int.tif"))
    assert len(ifgs3) == 10

    all_ifgs_names = [f.name for f in all_ifgs]
    batched_names = [f.name for f in ifgs1 + ifgs2 + ifgs3]
    assert all_ifgs_names == batched_names

    # Last, try one where we dont have the first CCSLC
    # The metadata should still tell it what the reference date is,
    # So the outputs should be the same
    p3_b = tmp_path / Path("third")
    files3_b = new_comp_slcs2 + file_batches[2]
    run_displacement_stack(p3_b, files3_b)
    ifgs3_b = sorted((p3_b / "interferograms").glob("*.int.tif"))
    assert len(ifgs3_b) == 10
    # Names should be the same as the previous run
    assert [f.name for f in ifgs3_b] == [f.name for f in ifgs3]
