#!/usr/bin/env python
from __future__ import annotations

import warnings
from datetime import date, datetime
from os import fspath
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from dolphin._log import get_log
from dolphin._types import DateOrDatetime, Filename
from dolphin.io import DEFAULT_DATETIME_FORMAT
from dolphin.utils import _format_date_pair

logger = get_log(__name__)

# Sentinel value for when no reference date is provided
# Appeases mypy
_DUMMY_DATE = datetime(1900, 1, 1)


class BaseStack(BaseModel):
    """Base class for mini- and full stack classes."""

    file_list: list[Filename] = Field(
        ...,
        description="List of SLC filenames in the ministack.",
    )
    dates: list[Sequence[datetime]] = Field(
        ...,
        description="List of date sequences, one for each SLC in the ministack. "
        "Each item is a list/tuple of datetime.date or datetime.datetime objects, "
        "as returned by [opera_utils._dates.get_dates][].",
    )
    is_compressed: list[bool] = Field(
        ...,
        description="List of booleans indicating whether each "
        "SLC is compressed or real.",
    )
    reference_date: datetime = Field(
        _DUMMY_DATE,
        description=(
            "Reference date to be used for understanding output interferograms. "
            "Note that this may be different from `dates[reference_idx]` if the"
            " ministack starts with a compressed SLC which has an earlier 'base phase',"
            " which is used as the phase linking references. "
            "It will propagate across ministacks when we always use `reference_idx=0`."
        ),
        validate_default=True,
    )
    file_date_fmt: str = Field(
        DEFAULT_DATETIME_FORMAT,
        description="Format string for the dates/datetimes in the ministack filenames.",
    )
    output_folder: Path = Field(
        Path(),
        description="Folder/location where ministack will write outputs to.",
    )
    reference_idx: int = Field(
        0,
        description="Index of the SLC to use as reference during phase linking",
    )

    @field_validator("dates", mode="before")
    @classmethod
    def _check_if_not_tuples(cls, v):
        if isinstance(v[0], (date, datetime)):
            v = [[d] for d in v]
        return v

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.file_list) == 0:
            msg = "Cannot create empty ministack"
            raise ValueError(msg)
        elif len(self.file_list) == 1:
            warnings.warn("Creating ministack with only one SLC", stacklevel=2)
        if len(self.file_list) != len(self.is_compressed):
            lengths = f"{len(self.file_list)} and {len(self.is_compressed)}"
            msg = f"file_list and is_compressed must be the same length: Got {lengths}"
            raise ValueError(msg)
        if len(self.dates) != len(self.file_list):
            lengths = f"{len(self.dates)} and {len(self.file_list)}"
            msg = f"dates and file_list must be the same length. Got {lengths}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_unset_reference_date(self):
        if self.reference_date == _DUMMY_DATE:
            ref_date = self.dates[self.reference_idx][0]
            logger.debug("No reference date provided, using first date: %s", ref_date)
            self.reference_date = ref_date
        return self

    @property
    def full_date_range(self) -> tuple[DateOrDatetime, DateOrDatetime]:
        """Full date range of all SLCs in the ministack."""
        return (self.reference_date, self.dates[-1][-1])

    @property
    def full_date_range_str(self) -> str:
        """Full date range of the ministack as a string, e.g. '20210101_20210202'.

        Includes both compressed + normal SLCs in the range.
        """
        return _format_date_pair(*self.full_date_range, fmt=self.file_date_fmt)

    @property
    def first_real_slc_idx(self) -> int:
        """Index of the first real SLC in the ministack."""
        try:
            return np.where(~np.array(self.is_compressed))[0][0]
        except IndexError as e:
            msg = "No real SLCs in ministack"
            raise ValueError(msg) from e

    @property
    def real_slc_date_range(self) -> tuple[DateOrDatetime, DateOrDatetime]:
        """Date range of the real SLCs in the ministack."""
        return (self.dates[self.first_real_slc_idx][0], self.dates[-1][-1])

    @property
    def real_slc_date_range_str(self) -> str:
        """Date range of the real SLCs in the ministack."""
        return _format_date_pair(*self.real_slc_date_range, fmt=self.file_date_fmt)

    @property
    def compressed_slc_file_list(self) -> list[Filename]:
        """List of compressed SLCs for this ministack."""
        return [f for f, is_comp in zip(self.file_list, self.is_compressed) if is_comp]

    @property
    def real_slc_file_list(self) -> list[Filename]:
        """List of real SLCs for this ministack."""
        return [
            f for f, is_comp in zip(self.file_list, self.is_compressed) if not is_comp
        ]

    def get_date_str_list(self) -> list[str]:
        """Get a formated string for each date/date tuple in the ministack."""
        date_strs: list[str] = []
        for d in self.dates:
            if len(d) == 1:
                # normal SLC files will have a single date
                s = d[0].strftime(self.file_date_fmt)
            else:
                # Compressed SLCs will have 2 dates in name marking the start / end
                s = _format_date_pair(d[0], d[1], fmt=self.file_date_fmt)
            date_strs.append(s)
        return date_strs

    def __rich_repr__(self):
        yield "file_list", self.file_list
        yield "dates", self.dates
        yield "is_compressed", self.is_compressed
        yield "reference_date", self.reference_date
        yield "file_date_fmt", self.file_date_fmt
        yield "output_folder", self.output_folder
        yield "reference_idx", self.reference_idx


class CompressedSlcInfo(BaseModel):
    """Class for holding attributes about one compressed SLC."""

    real_slc_file_list: list[Filename] = Field(
        ...,
        description="List of real SLC filenames in the ministack.",
    )
    real_slc_dates: list[datetime] = Field(
        ...,
        description="List of date sequences, one for each SLC in the ministack. "
        "Each item is a list/tuple of datetime.date or datetime.datetime objects.",
    )
    compressed_slc_file_list: list[Filename] = Field(
        ...,
        description="List of compressed SLC filenames in the ministack.",
    )
    reference_date: datetime = Field(
        _DUMMY_DATE,
        description="Reference date for understanding output interferograms. "
        "Note that this may be different from `dates[reference_idx]` if the ministack "
        "starts with a compressed SLC which has an earlier 'base phase', which "
        "is used as the phase linking references. "
        "It will propagate across ministacks when we always use `reference_idx=0`.",
        validate_default=True,
    )
    file_date_fmt: str = Field(
        DEFAULT_DATETIME_FORMAT,
        description="Format string for the dates/datetimes in the ministack filenames.",
    )
    output_folder: Path = Field(
        Path(),
        description="Folder/location where ministack will write outputs to.",
    )

    @field_validator("real_slc_dates", mode="before")
    @classmethod
    def _untuple_dates(cls, v):
        """Make the dates not be tuples/lists of datetimes."""
        out = []
        for item in v:
            if hasattr(item, "__iter__"):
                # Make sure they didn't pass more than 1 date, implying
                # a compressed SLC
                # assert len(item) == 1
                if isinstance(item, str):
                    out.append(item)
                elif len(item) > 1:
                    msg = f"Cannot pass multiple dates for a compressed SLC. Got {item}"
                    raise ValueError(msg)
                else:
                    out.append(item[0])
            else:
                out.append(item)
        return out

    @model_validator(mode="after")
    def _check_lengths(self):
        rlen = len(self.real_slc_file_list)
        clen = len(self.real_slc_dates)
        if rlen != clen:
            lengths = f"{rlen} and {clen}"
            msg = (
                "real_slc_file_list and real_slc_dates must be the same length. "
                f"Got {lengths}"
            )
            raise ValueError(msg)
        return self

    @property
    def real_date_range(self) -> tuple[DateOrDatetime, DateOrDatetime]:
        """Date range of the real SLCs in the ministack."""
        return (self.real_slc_dates[0], self.real_slc_dates[-1])

    @property
    def filename(self) -> str:
        """The filename of the compressed SLC for this ministack."""
        date_str = _format_date_pair(*self.real_date_range, fmt=self.file_date_fmt)
        return f"compressed_{date_str}.tif"

    @property
    def path(self) -> Path:
        """The path of the compressed SLC for this ministack."""
        return self.output_folder / self.filename

    def write_metadata(
        self, domain: str = "DOLPHIN", output_file: Optional[Filename] = None
    ):
        """Write the metadata to the compressed SLC file.

        Parameters
        ----------
        domain : str, optional
            Domain to write the metadata to, by default "DOLPHIN".
        output_file : Optional[Filename], optional
            Path to the file to write the metadata to, by default None.
            If None, will use `self.path`.

        """
        from dolphin.io import set_raster_metadata

        out = self.path if output_file is None else Path(output_file)
        if not out.exists():
            msg = f"Must create {out} before writing metadata"
            raise FileNotFoundError(msg)

        set_raster_metadata(
            out,
            metadata=self.model_dump(mode="json"),
            domain=domain,
        )

    @classmethod
    def from_file_metadata(cls, filename: Filename) -> CompressedSlcInfo:
        """Try to parse the CCSLC metadata from `filename`."""
        import json

        from dolphin.io import get_raster_metadata

        domains = ["DOLPHIN", ""]
        for domain in domains:
            gdal_md = get_raster_metadata(filename, domain=domain)
            if not gdal_md:
                continue
            else:
                break
        else:
            msg = f"Could not find metadata in {filename}"
            raise ValueError(msg)
        # GDAL can write it weirdly and mess up the JSON
        cleaned = {}
        for k, v in gdal_md.items():
            try:
                # Swap the single quotes for double quotes to parse lists
                cleaned[k] = json.loads(v.replace("'", '"'))
            except json.JSONDecodeError:
                cleaned[k] = v
        # Parse the date/file lists from the metadata
        out = cls.model_validate(cleaned)
        # Overwrite the `output_folder` part- we may have moved it since
        # writing the metadata
        out.output_folder = Path(filename).parent
        return out

    def __fspath__(self):
        return fspath(self.path)


class MiniStackInfo(BaseStack):
    """Class for holding attributes about one mini-stack of SLCs.

    Used for planning the processing of a batch of SLCs.
    """

    def get_compressed_slc_info(self) -> CompressedSlcInfo:
        """Get the compressed SLC which will come from this ministack.

        Excludes the existing compressed SLCs during the compression.
        """
        real_slc_files: list[Filename] = []
        real_slc_dates: list[Sequence[DateOrDatetime]] = []
        comp_slc_files: list[Filename] = []
        for f, d, is_comp in zip(self.file_list, self.dates, self.is_compressed):
            if is_comp:
                comp_slc_files.append(f)
            else:
                real_slc_files.append(f)
                real_slc_dates.append(d)

        return CompressedSlcInfo(
            real_slc_file_list=real_slc_files,
            real_slc_dates=real_slc_dates,
            compressed_slc_file_list=comp_slc_files,
            reference_date=self.reference_date,
            file_date_fmt=self.file_date_fmt,
            output_folder=self.output_folder,
        )


class MiniStackPlanner(BaseStack):
    """Class for planning the processing of batches of SLCs."""

    max_num_compressed: int = 5

    def plan(self, ministack_size: int) -> list[MiniStackInfo]:
        """Create a list of ministacks to be processed."""
        if ministack_size < 2:
            msg = "Cannot create ministacks with size < 2"
            raise ValueError(msg)

        output_ministacks: list[MiniStackInfo] = []

        # Start of with any compressed SLCs that are passed in
        compressed_slc_infos: list[CompressedSlcInfo] = []
        for f in self.compressed_slc_file_list:
            # Note: these must actually exist to be used!
            compressed_slc_infos.append(CompressedSlcInfo.from_file_metadata(f))

        # Solve each ministack using current chunk (and the previous compressed SLCs)
        ministack_starts = range(
            self.first_real_slc_idx, len(self.file_list), ministack_size
        )

        for full_stack_idx in ministack_starts:
            cur_slice = slice(full_stack_idx, full_stack_idx + ministack_size)
            cur_files = list(self.file_list[cur_slice]).copy()
            cur_dates = list(self.dates[cur_slice]).copy()

            comp_slc_files = [c.path for c in compressed_slc_infos]
            # Add the existing compressed SLC files to the start, but
            # limit the num comp slcs `max_num_compressed`
            cur_comp_slc_files = comp_slc_files[-self.max_num_compressed :]
            combined_files = cur_comp_slc_files + cur_files

            combined_dates = [
                c.real_date_range
                for c in compressed_slc_infos[-self.max_num_compressed :]
            ] + cur_dates

            num_ccslc = len(cur_comp_slc_files)
            combined_is_compressed = num_ccslc * [True] + list(
                self.is_compressed[cur_slice]
            )
            # If there are any compressed SLCs, set the reference to the last one
            try:
                last_compressed_idx = np.where(combined_is_compressed)[0]
                reference_idx = last_compressed_idx[-1]
            except IndexError:
                reference_idx = 0

            # Make the current ministack output folder using the start/end dates
            new_date_str = _format_date_pair(
                cur_dates[0][0], cur_dates[-1][-1], fmt=self.file_date_fmt
            )
            cur_output_folder = self.output_folder / new_date_str
            cur_ministack = MiniStackInfo(
                file_list=combined_files,
                dates=combined_dates,
                is_compressed=combined_is_compressed,
                reference_idx=reference_idx,
                output_folder=cur_output_folder,
                reference_date=self.reference_date,
                # TODO: we'll need to alter logic here if we dont fix
                # reference_idx=0, since this will change the reference date
            )

            output_ministacks.append(cur_ministack)
            cur_comp_slc = cur_ministack.get_compressed_slc_info()
            compressed_slc_infos.append(cur_comp_slc)

        return output_ministacks
