import xarray as xr
import numpy as np
from scipy.ndimage import convolve


# --- Shared internal helper

def _compute_spatial_lag(clean_data: np.ndarray, is_valid_mask: np.ndarray, kernel: np.ndarray) -> tuple:
    """
    Compute the local weighted sum and spatial weight matrix (number of valid
    neighbours) via vectorised convolution.

    NaN handling strategy:
      1. Replace NaN with 0 before convolving (clean_data).
      2. Convolve the binary mask to count how many valid neighbours were
         actually present in each window (w_matrix).
      3. The caller divides local_sum / w_matrix to obtain unbiased means.

    Parameters
    ----------
    clean_data : np.ndarray
        2D array with NaN replaced by 0.
    is_valid_mask : np.ndarray
        Boolean mask: True where the value is valid.
    kernel : np.ndarray
        Convolution kernel defining the spatial neighbourhood.

    Returns
    -------
    tuple
        (local_sum, w_matrix) where local_sum is the weighted neighbour sum
        per pixel and w_matrix is the count of valid neighbours per pixel.
    """
    local_sum = convolve(clean_data, kernel, mode='constant', cval=0.0)
    w_matrix = convolve(is_valid_mask.astype(float), kernel, mode='constant', cval=0.0)
    return local_sum, w_matrix


# --- Getis-Ord Gi*

def calculate_getis_ord_gi_star(data_array: xr.DataArray, kernel_size: int = 3, min_valid_neighbors: int = 3) -> xr.Dataset:
    """
    Compute the local Getis-Ord Gi* spatial statistic on a 2D continuous map,
    with rigorous handling of vector mask edges (NaN).

    Gi* identifies where statistically high or low values are spatially
    clustered. Unlike the Gi without asterisk, it includes the target pixel
    in the local sum, making it more sensitive to the pixel's own extreme value.

    Methodological note:
        Gi* assumes spatial stationarity (constant mean and variance across the
        study area). This is reasonable for small, climatically homogeneous areas
        such as nature parks or catchments. For regional analyses, results should
        be interpreted with caution as the global mean may mask structural climatic
        heterogeneities (Getis & Ord, 1992).

    Ecological interpretation:
    - Applied to 'accumulated_deficit' or 'recovery_time':
        Hotspot (high Z)  -- high vulnerability or slow recovery zone.
        Coldspot (low Z)  -- climatic refugium or rapid recovery zone.
    - Applied to 'resilience' or 'resistance':
        Signs are inverted; see vegetation_metrics documentation.

    Parameters
    ----------
    data_array : xr.DataArray
        Static 2D map (e.g. metrics_ds['accumulated_deficit']).
        Must not have a time dimension.
    kernel_size : int
        Size of the square Queen kernel (must be odd: 3, 5, 7...).
        Defines the spatial neighbourhood of each pixel. At 1.1 km resolution:
        3x3 covers ~10 km2, 5x5 ~30 km2, 7x7 ~60 km2.
        Default 3, appropriate for local forest decline processes.
    min_valid_neighbors : int
        Minimum number of valid cells around a pixel to accept its result.
        Highly isolated pixels or extreme edge pixels are set to NaN.

    Returns
    -------
    xr.Dataset
        Two layers:
            'z_score'    : Continuous Gi* statistic (for fine cartography).
            'clustering' : Discrete classification into 5 confidence levels:
                            3 -> Hotspot  99% | 2 -> Hotspot  95%
                            0 -> Not significant
                           -2 -> Coldspot 95% | -3 -> Coldspot 99%
    """
    # Extract numpy array and build validity mask
    raw_data = data_array.compute().values
    is_valid_mask = ~np.isnan(raw_data)

    # Global statistics (valid pixels only)
    global_mean = float(np.nanmean(raw_data))
    global_std = float(np.nanstd(raw_data))
    n_valid = int(np.sum(is_valid_mask))

    if global_std < 1e-6:
        raise ValueError(
            "The map standard deviation is near zero. "
            "Insufficient spatial variability to apply Gi*."
        )
    if n_valid < 9:
        raise ValueError(
            f"Only {n_valid} valid pixels found. At least 9 are required "
            "to compute the statistic with a 3x3 kernel."
        )

    # Queen kernel -- Gi* includes the target pixel (weight = 1)
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd (3, 5, 7...). Received: {kernel_size}.")
    kernel = np.ones((kernel_size, kernel_size))
    clean_data = np.where(is_valid_mask, raw_data, 0.0)

    # Spatial lag (shared helper)
    local_sum, w_matrix = _compute_spatial_lag(clean_data, is_valid_mask, kernel)

    # Formal Getis-Ord Gi* formula:
    #   Numerator   : sum_local - (global_mean * W)
    #   Denominator : global_std * sqrt[(n * W - W^2) / (n - 1)]
    numerator = local_sum - (global_mean * w_matrix)

    inner_sqrt = (n_valid * w_matrix - w_matrix ** 2) / (n_valid - 1)
    inner_sqrt = np.where(inner_sqrt > 0, inner_sqrt, 0.0)
    denominator = global_std * np.sqrt(inner_sqrt)

    with np.errstate(divide='ignore', invalid='ignore'):
        z_score = numerator / denominator

    # Quality filters
    z_score = np.where(w_matrix >= min_valid_neighbors, z_score, np.nan)
    z_score = np.where(is_valid_mask, z_score, np.nan)

    # Discrete classification -- more restrictive conditions first
    # Critical values of the standard normal distribution:
    conditions = [
        z_score > 2.58,
        z_score > 1.96,
        z_score < -2.58,
        z_score < -1.96
    ]
    choices = [3, 2, -3, -2]
    clustering = np.select(conditions, choices, default=0).astype(float)
    clustering = np.where(is_valid_mask, clustering, np.nan)

    # Reconstruct as xarray, preserving spatial coordinates
    output_ds = xr.Dataset(
        {
            "z_score":    (["y", "x"], z_score),
            "clustering": (["y", "x"], clustering)
        },
        coords=data_array.coords
    )

    output_ds.attrs["statistical_method"] = "Getis-Ord Gi* Local Spatial Autocorrelation"
    output_ds.attrs["reference"] = "Getis & Ord (1992); Ord & Getis (1995)"
    output_ds.attrs["kernel"] = f"Queen {kernel_size}x{kernel_size} (includes target pixel)"
    output_ds.attrs["min_valid_neighbors_required"] = min_valid_neighbors
    output_ds.attrs["n_valid_pixels"] = n_valid
    output_ds.attrs["legend_clustering"] = {
         3: "Hotspot 99% confidence",
         2: "Hotspot 95% confidence",
         0: "Not significant",
        -2: "Coldspot 95% confidence",
        -3: "Coldspot 99% confidence"
    }

    return output_ds
