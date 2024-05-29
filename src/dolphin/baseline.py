import datetime

import h5py
import isce3
import isce3.core
import numpy as np
from numpy.typing import ArrayLike
from opera_utils import get_radar_wavelength
from pyproj import CRS, Transformer

from dolphin._types import Filename


def compute(
    llh: ArrayLike,
    ref_pos: ArrayLike,
    sec_pos: ArrayLike,
    ref_rng: float,
    sec_rng: float,
    ref_vel: ArrayLike,
    ell: isce3.core.Ellipsoid,
):
    """Compute the perpendicular baseline at a given geographic position.

    Parameters
    ----------
    llh : ArrayLike
        Lon/Lat/Height vector specifying the target position.
    ref_pos : ArrayLike
        Reference position vector (x, y, z) in ECEF coordinates.
    sec_pos : ArrayLike
        Secondary position vector (x, y, z) in ECEF coordinates.
    ref_rng : float
        Range from the reference satellite to the target.
    sec_rng : float
        Range from the secondary satellite to the target.
    ref_vel : ArrayLike
        Velocity vector (vx, vy, vz) of the reference satellite in ECEF coordinates.
    ell : isce3.core.Ellipsoid
        Ellipsoid for the target surface.

    Returns
    -------
    float
        Perpendicular baseline, in meters.

    """
    # Difference in position between the two passes
    baseline = np.linalg.norm(sec_pos - ref_pos)

    # Compute angle between LOS vector and baseline vector
    # via the law of cosines
    costheta = (ref_rng**2 + baseline**2 - sec_rng**2) / (2 * ref_rng * baseline)

    sintheta = np.sqrt(1 - costheta**2)
    perp = baseline * sintheta
    # par = baseline * costheta

    targ_xyz = ell.lon_lat_to_xyz(llh)
    direction = np.sign(
        np.dot(np.cross(targ_xyz - ref_pos, sec_pos - ref_pos), ref_vel)
    )

    return direction * perp


def get_orbit_arrays(
    h5file: Filename,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, datetime.datetime]:
    """Parse orbit info from OPERA S1 CSLC HDF5 file into python types.

    Parameters
    ----------
    h5file : Filename
        Path to OPERA S1 CSLC HDF5 file.

    Returns
    -------
    times : np.ndarray
        Array of times in seconds since reference epoch.
    positions : np.ndarray
        Array of positions in meters.
    velocities : np.ndarray
        Array of velocities in meters per second.
    reference_epoch : datetime.datetime
        Reference epoch of orbit.

    """
    with h5py.File(h5file) as hf:
        orbit_group = hf["/metadata/orbit"]
        times = orbit_group["time"][:]
        positions = np.stack([orbit_group[f"position_{p}"] for p in ["x", "y", "z"]]).T
        velocities = np.stack([orbit_group[f"velocity_{p}"] for p in ["x", "y", "z"]]).T
        reference_epoch = datetime.datetime.fromisoformat(
            orbit_group["reference_epoch"][()].decode()
        )

    return times, positions, velocities, reference_epoch


def get_cslc_orbit(h5file: Filename) -> isce3.core.Orbit:
    """Parse orbit info from OPERA S1 CSLC HDF5 file into an isce3.core.Orbit.

    Parameters
    ----------
    h5file : Filename
        Path to OPERA S1 CSLC HDF5 file.

    Returns
    -------
    orbit : isce3.core.Orbit
        Orbit object.

    """
    times, positions, velocities, reference_epoch = get_orbit_arrays(h5file)
    orbit_svs = []
    for t, x, v in zip(times, positions, velocities):
        orbit_svs.append(
            isce3.core.StateVector(
                isce3.core.DateTime(reference_epoch + datetime.timedelta(seconds=t)),
                x,
                v,
            )
        )

    return isce3.core.Orbit(orbit_svs)


def get_xy_coords(h5file: Filename, subsample: int = 100):
    """Get x and y grid from OPERA S1 CSLC HDF5 file.

    Parameters
    ----------
    h5file : Filename
        Path to OPERA S1 CSLC HDF5 file.
    subsample : int, optional
        Subsampling factor, by default 100

    Returns
    -------
    x : np.ndarray
        Array of x coordinates in meters.
    y : np.ndarray
        Array of y coordinates in meters.
    projection : int
        EPSG code of projection.

    """
    with h5py.File(h5file) as hf:
        x = hf["/data/x_coordinates"][:]
        y = hf["/data/y_coordinates"][:]
        projection = hf["/data/projection"][()]

    return x[::subsample], y[::subsample], projection


def get_lonlat_grid(h5file: Filename, subsample: int = 100):
    """Get 2D latitude and longitude grid from OPERA S1 CSLC HDF5 file.

    Parameters
    ----------
    h5file : Filename
        Path to OPERA S1 CSLC HDF5 file.
    subsample : int, optional
        Subsampling factor, by default 100

    Returns
    -------
    lat : np.ndarray
        2D Array of latitude coordinates in degrees.
    lon : np.ndarray
        2D Array of longitude coordinates in degrees.
    projection : int
        EPSG code of projection.

    """
    x, y, projection = get_xy_coords(h5file, subsample)
    X, Y = np.meshgrid(x, y)
    xx = X.flatten()
    yy = Y.flatten()
    crs = CRS.from_epsg(projection)
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(xx=xx, yy=yy, radians=True)
    lon = lon.reshape(X.shape)
    lat = lat.reshape(Y.shape)
    return lon, lat


def compute_baselines(
    h5file_ref: Filename,
    h5file_sec: Filename,
    height: float = 0.0,
    latlon_subsample: int = 100,
    threshold: float = 1e-08,
    maxiter: int = 50,
    delta_range: float = 10.0,
):
    """Compute the perpendicular baseline at a subsampled grid for two CSLCs.

    Parameters
    ----------
    h5file_ref : Filename
        Path to reference OPERA S1 CSLC HDF5 file.
    h5file_sec : Filename
        Path to secondary OPERA S1 CSLC HDF5 file.
    height: float
        Target height to use for baseline computation.
        Default = 0.0
    latlon_subsample: int
        Factor by which to subsample the CSLC latitude/longitude grids.
        Default = 30
    threshold : float
        isce3 geo2rdr: azimuth time convergence threshold in meters
        Default = 1e-8
    maxiter : int
        isce3 geo2rdr: Maximum number of Newton-Raphson iterations
        Default = 50
    delta_range : float
        isce3 geo2rdr: Step size used for computing derivative of doppler
        Default = 10.0

    Returns
    -------
    lon : np.ndarray
        2D array of longitude coordinates in degrees.
    lat : np.ndarray
        2D array of latitude coordinates in degrees.
    baselines : np.ndarray
        2D array of perpendicular baselines

    """
    lon_grid, lat_grid = get_lonlat_grid(h5file_ref, subsample=latlon_subsample)
    lon_arr = lon_grid.ravel()
    lat_arr = lat_grid.ravel()

    ellipsoid = isce3.core.Ellipsoid()
    zero_doppler = isce3.core.LUT2d()
    wavelength = get_radar_wavelength(h5file_ref)
    side = isce3.core.LookSide.Right

    orbit_ref = get_cslc_orbit(h5file_ref)
    orbit_sec = get_cslc_orbit(h5file_sec)

    baselines = []
    for lon, lat in zip(lon_arr, lat_arr):
        llh_rad = np.array([lon, lat, height]).reshape((3, 1))
        az_time_ref, range_ref = isce3.geometry.geo2rdr(
            llh_rad,
            ellipsoid,
            orbit_ref,
            zero_doppler,
            wavelength,
            side,
            threshold=threshold,
            maxiter=maxiter,
            delta_range=delta_range,
        )
        az_time_sec, range_sec = isce3.geometry.geo2rdr(
            llh_rad,
            ellipsoid,
            orbit_sec,
            zero_doppler,
            wavelength,
            side,
            threshold=threshold,
            maxiter=maxiter,
            delta_range=delta_range,
        )

        pos_ref, velocity = orbit_ref.interpolate(az_time_ref)
        pos_sec, _ = orbit_sec.interpolate(az_time_sec)
        b = compute(
            llh_rad, pos_ref, pos_sec, range_ref, range_sec, velocity, ellipsoid
        )

        baselines.append(b)

    baseline_cube = np.array(baselines).reshape(lon_grid.shape)
    return lon_grid, lat_grid, baseline_cube
