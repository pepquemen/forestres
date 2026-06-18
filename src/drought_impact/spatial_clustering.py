import xarray as xr
import numpy as np
from scipy.ndimage import convolve


# =============================================================================
# FUNCIÓN PRIVADA COMPARTIDA
# =============================================================================

def _compute_spatial_lag(clean_data: np.ndarray, is_valid_mask: np.ndarray, kernel: np.ndarray) -> tuple:
    """
    Función interna compartida. Calcula la suma local ponderada y la matriz de pesos
    espaciales (número de vecinos válidos) mediante convolución vectorizada.

    La estrategia para gestionar NaN es:
      1. Reemplazar NaN por 0 antes de convolucionar (clean_data).
      2. Convolucionar la máscara binaria para saber cuántos vecinos válidos
         había realmente en cada ventana (w_matrix).
      3. El llamador divide local_sum / w_matrix para obtener medias no sesgadas.

    Parámetros:
    -----------
    clean_data : np.ndarray
        Array 2D con NaN reemplazados por 0.
    is_valid_mask : np.ndarray
        Máscara booleana: True donde el dato es válido.
    kernel : np.ndarray
        Kernel de convolución que define la vecindad espacial.

    Retorna:
    --------
    tuple : (local_sum, w_matrix)
        local_sum : suma ponderada de los valores vecinos por píxel.
        w_matrix  : número de vecinos válidos por píxel.
    """
    local_sum = convolve(clean_data, kernel, mode='constant', cval=0.0)
    w_matrix = convolve(is_valid_mask.astype(float), kernel, mode='constant', cval=0.0)
    return local_sum, w_matrix


# =============================================================================
# GETIS-ORD GI*
# =============================================================================

def calculate_getis_ord_gi_star(data_array: xr.DataArray, kernel_size: int = 3, min_valid_neighbors: int = 3) -> xr.Dataset:
    """
    Calcula el estadístico espacial Getis-Ord Gi* local sobre un mapa 2D continuo,
    gestionando rigurosamente los bordes de máscaras vectoriales (NaN).

    El Gi* identifica dónde se concentran valores estadísticamente altos o bajos
    en el espacio (clusters). A diferencia del Gi sin asterisco, incluye el propio
    píxel en la suma local, lo que lo hace más sensible a los valores extremos
    del propio píxel analizado.

    NOTA DE LIMITACIÓN METODOLÓGICA:
        El estadístico Gi* asume estacionariedad espacial (media y varianza
        constantes en toda la zona de estudio). Esta asunción es razonable
        para áreas pequeñas y climáticamente homogéneas como parques naturales
        o cuencas. Para análisis a escala regional o peninsular, los resultados
        deben interpretarse con cautela ya que la media global puede enmascarar
        heterogeneidades climáticas estructurales (Getis & Ord, 1992).

    Interpretación ecológica recomendada:
    - Sobre 'accumulated_deficit' o 'recovery_time':
        Hotspot (Z alto)  → zona de alta vulnerabilidad o lenta recuperación.
        Coldspot (Z bajo) → refugio climático o zona de rápida recuperación.
    - Sobre 'resilience' o 'resistance':
        Los signos se invierten; consultar la documentación de vegetation_metrics.

    Parámetros:
    -----------
    data_array : xr.DataArray
        Mapa 2D estático (ej. metrics_ds['accumulated_deficit']).
        No debe tener dimensión temporal.
    kernel_size : int
        Tamaño del kernel Queen cuadrado (debe ser impar: 3, 5, 7...).
        Define el vecindario espacial de cada píxel. Con resolución de 1.1 km
        (datos CSIC): kernel 3x3 evalúa ~10 km², 5x5 evalúa ~30 km², 7x7 ~60 km².
        Por defecto 3, adecuado para procesos de decaimiento forestal local.
    min_valid_neighbors : int
        Número mínimo de celdas válidas alrededor del píxel para aceptar el
        resultado. Píxeles muy aislados o en bordes extremos quedan como NaN.

    Retorna:
    --------
    xr.Dataset con dos capas:
        'z_score'    : Valor continuo del estadístico Gi* (para cartografía fina).
        'clustering' : Clasificación discreta en 5 niveles de confianza:
                        3 → Hotspot  99% | 2 → Hotspot  95%
                        0 → No significativo
                       -2 → Coldspot 95% | -3 → Coldspot 99%
    """
    # 1. Extraer numpy y construir máscaras
    raw_data = data_array.compute().values
    is_valid_mask = ~np.isnan(raw_data)

    # 2. Estadísticos globales (solo píxeles válidos)
    global_mean = float(np.nanmean(raw_data))
    global_std = float(np.nanstd(raw_data))
    n_valid = int(np.sum(is_valid_mask))

    if global_std < 1e-6:
        raise ValueError(
            "La desviación estándar del mapa es cercana a cero. "
            "No hay variabilidad espacial suficiente para aplicar Gi*."
        )
    if n_valid < 9:
        raise ValueError(
            f"Solo hay {n_valid} píxeles válidos. Se necesitan al menos 9 para "
            "calcular el estadístico con un kernel 3x3."
        )

    # 3. Kernel Queen configurable — Gi* incluye el píxel central (peso = 1)
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size debe ser impar (3, 5, 7...). Recibido: {kernel_size}.")
    kernel = np.ones((kernel_size, kernel_size))
    clean_data = np.where(is_valid_mask, raw_data, 0.0)

    # 4. Lag espacial compartido
    local_sum, w_matrix = _compute_spatial_lag(clean_data, is_valid_mask, kernel)

    # 5. Fórmula formal de Getis-Ord Gi*
    #    Numerador   : ΣLocal - (μ_global × W)
    #    Denominador : σ_global × √[(n×W - W²) / (n-1)]
    numerator = local_sum - (global_mean * w_matrix)

    inner_sqrt = (n_valid * w_matrix - w_matrix ** 2) / (n_valid - 1)
    inner_sqrt = np.where(inner_sqrt > 0, inner_sqrt, 0.0)
    denominator = global_std * np.sqrt(inner_sqrt)

    with np.errstate(divide='ignore', invalid='ignore'):
        z_score = numerator / denominator

    # 6. Filtros de calidad analítica
    z_score = np.where(w_matrix >= min_valid_neighbors, z_score, np.nan)
    z_score = np.where(is_valid_mask, z_score, np.nan)

    # 7. Clasificación discreta robusta con np.select
    #    El orden importa: condiciones más restrictivas primero
    conditions = [
        z_score > 2.58,
        z_score > 1.96,
        z_score < -2.58,
        z_score < -1.96
    ]
    choices = [3, 2, -3, -2]
    clustering = np.select(conditions, choices, default=0).astype(float)
    clustering = np.where(is_valid_mask, clustering, np.nan)

    # 8. Reconstruir a xarray preservando coordenadas espaciales
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