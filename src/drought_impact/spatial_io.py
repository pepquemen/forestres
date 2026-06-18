import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import warnings
import logging
from shapely.geometry import box

logger = logging.getLogger(__name__)


def load_and_standardize_netcdf(
    file_path: str,
    var_mapping: dict = None,
    crs: str = "EPSG:4326",
    bbox: tuple = None
) -> xr.Dataset:
    """
    Carga un archivo NetCDF y estandariza sus variables y sistema de coordenadas.
    Esta es la capa de abstracción para que el resto de la librería no dependa de
    la fuente original.

    Si se proporciona un bbox, el dataset se recorta espacialmente ANTES de cargar
    los datos en memoria mediante selección por índices (isel). Esto reduce
    drásticamente el uso de memoria con índices de escala larga (SPI/SPEI), que de
    otro modo cargarían toda España (varios GB) antes de poder recortar.

    Parámetros:
    -----------
    file_path : str
        Ruta al archivo NetCDF original.
    var_mapping : dict, opcional
        Diccionario para renombrar las variables al estándar de la librería.
    crs : str, opcional
        Sistema de Coordenadas de Referencia por defecto si el archivo no lo tiene.
    bbox : tuple, opcional
        Bounding box (minx, miny, maxx, maxy) en las coordenadas del raster para
        recorte espacial temprano. Si es None, se carga el dataset completo.

    Retorna:
    --------
    xr.Dataset
        Dataset limpio, con nombres estandarizados y sistema de coordenadas validado.
    """
    # 1. Cargar el dataset con carga perezosa para no llenar memoria
    dataset = xr.open_dataset(
        file_path,
        decode_coords="all",
        engine="netcdf4",
        chunks={"time": 100}
    )

    # 2. Renombrar variables al estándar interno
    if var_mapping:
        rename_dict = {k: v for k, v in var_mapping.items() if k in dataset.variables}
        dataset = dataset.rename(rename_dict)

    # 3. Validar y asignar CRS
    if not dataset.rio.crs:
        warnings.warn(f"El archivo no tiene CRS definido. Asignando {crs} por defecto.")
        dataset = dataset.rio.write_crs(crs)

    # 4. Estandarizar dimensiones espaciales a 'x' e 'y'
    dim_mapping = {}
    if 'lon' in dataset.dims and 'lat' in dataset.dims:
        dim_mapping = {'lon': 'x', 'lat': 'y'}
    elif 'longitude' in dataset.dims and 'latitude' in dataset.dims:
        dim_mapping = {'longitude': 'x', 'latitude': 'y'}

    if dim_mapping:
        dataset = dataset.rename_dims(dim_mapping)
        dataset = dataset.rename(dim_mapping)
        dataset = dataset.assign_coords({
            "x": dataset.x,
            "y": dataset.y
        })

    # 5. Recorte espacial temprano por bbox (ANTES de cargar en memoria)
    # Selección por índices sobre las coordenadas, operación perezosa e instantánea.
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        x_vals = dataset.x.values
        y_vals = dataset.y.values

        x_mask = (x_vals >= minx) & (x_vals <= maxx)
        y_mask = (y_vals >= miny) & (y_vals <= maxy)

        x_idx = np.where(x_mask)[0]
        y_idx = np.where(y_mask)[0]

        if len(x_idx) == 0 or len(y_idx) == 0:
            raise ValueError(
                "El bbox del shapefile no intersecta con la extensión del raster. "
                "Verifica que el shapefile y los datos están en el mismo CRS."
            )

        dataset = dataset.isel(x=x_idx, y=y_idx)

    # 6. Confirmar dimensiones en rioxarray
    dataset = dataset.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    # 7. Validación de dimensiones
    if "x" not in dataset.dims or "y" not in dataset.dims:
        raise ValueError("No se pudieron estandarizar las dimensiones espaciales a 'x' e 'y'.")

    return dataset


def _get_shapefile_bbox(shapefile_path: str, target_crs) -> tuple:
    """
    Lee un shapefile y devuelve su bounding box reproyectado al CRS del raster.

    Parámetros:
    -----------
    shapefile_path : str
        Ruta al archivo vectorial.
    target_crs :
        CRS del raster al que reproyectar el bbox.

    Retorna:
    --------
    tuple (minx, miny, maxx, maxy) en las coordenadas del raster.
    """
    polygons = gpd.read_file(shapefile_path)
    if polygons.crs != target_crs:
        polygons = polygons.to_crs(target_crs)
    return tuple(polygons.total_bounds)  # (minx, miny, maxx, maxy)


def _align_time_nearest(ds_source: xr.Dataset, target_times: np.ndarray) -> xr.Dataset:
    """
    Alinea temporalmente un dataset a los timestamps de referencia usando
    búsqueda vectorizada del vecino más cercano con numpy.

    Este método es más rápido que xr.interp() porque opera directamente sobre
    los índices de tiempo como enteros (nanosegundos desde epoch), sin bucles
    Python ni operaciones de interpolación sobre datos espaciales.

    Funciona con cualquier resolución temporal de entrada (semanal, mensual,
    diaria, irregular) y es robusto frente a timestamps no uniformes.

    Parámetros:
    -----------
    ds_source : xr.Dataset
        Dataset cuya dimensión temporal se va a realinear.
    target_times : np.ndarray
        Array de timestamps de referencia (dtype datetime64).

    Retorna:
    --------
    xr.Dataset con la dimensión temporal reindexada a target_times.
    """
    # Convertir timestamps a enteros (nanosegundos desde epoch) para aritmética
    src_times_int = ds_source.time.values.astype("datetime64[ns]").astype(np.int64)
    tgt_times_int = target_times.astype("datetime64[ns]").astype(np.int64)

    # Para cada timestamp objetivo, encontrar el índice del más cercano en el origen
    # Matriz de distancias absolutas: shape (n_src, n_tgt)
    # argmin sobre el eje 0 → índice del timestamp origen más cercano a cada objetivo
    closest_idx = np.argmin(
        np.abs(src_times_int[:, None] - tgt_times_int[None, :]),
        axis=0
    )

    # Seleccionar por posición y reasignar los timestamps objetivo
    ds_aligned = ds_source.isel(time=closest_idx)
    ds_aligned["time"] = target_times

    return ds_aligned


def load_and_merge_datasets(
    drought_path: str,
    vegetation_path: str,
    drought_var: str,
    vegetation_var: str,
    shapefile_path: str,
    crs: str = "EPSG:4326"
) -> xr.Dataset:
    """
    Carga dos NetCDF (índice de sequía y vegetación), los recorta espacialmente al
    área de estudio, los alinea temporalmente y los fusiona en un único Dataset
    recortado y listo para el pipeline.

    FLUJO OPTIMIZADO:
        1. Lee el bounding box del shapefile.
        2. Recorta ambos NetCDF por bbox ANTES de cargarlos en memoria (isel).
        3. Recorta al período temporal común.
        4. Alinea el índice a los timestamps del SNDVI mediante búsqueda
           vectorizada del vecino más cercano (numpy argmin, <1 segundo).
        5. Fusiona ambas variables.
        6. Aplica el recorte exacto al contorno del shapefile.

    El recorte temprano por bbox es crítico para el rendimiento: los índices de
    escala larga (SPI/SPEI) tienen miles de pasos temporales y cargar toda España
    en memoria (varios GB) antes de recortar puede agotar la RAM. Recortando por
    bbox primero, todas las operaciones posteriores trabajan sobre el área de
    estudio (típicamente unos pocos MB).

    El índice de sequía del CSIC viene en resolución semanal (~8 días).
    El SNDVI viene en resolución semi-mensual (1 y 15 de cada mes).
    La alineación temporal se realiza por búsqueda vectorizada del vecino más
    cercano, preservando las propiedades estadísticas del índice estandarizado
    (media=0, std=1) sin promediar ni interpolar valores.

    Parámetros:
    -----------
    drought_path : str
        Ruta al NetCDF del índice de sequía (ej. spei03.nc).
    vegetation_path : str
        Ruta al NetCDF de vegetación (ej. SNDVI.nc).
    drought_var : str
        Nombre de la variable de sequía en el NetCDF original (ej. 'spei').
    vegetation_var : str
        Nombre de la variable de vegetación en el NetCDF original (ej. 'SNDVI').
    shapefile_path : str
        Ruta al shapefile del área de estudio. Define el recorte espacial y es
        obligatorio para garantizar un uso eficiente de memoria.
    crs : str
        CRS por defecto si los archivos no lo tienen definido.

    Retorna:
    --------
    xr.Dataset
        Dataset fusionado y recortado al área de estudio, con variables
        estandarizadas 'drought_index' y 'ndvi', listo para el pipeline.
    """
    # 1. Calcular el bbox del shapefile en el CRS de los datos
    logger.info(f"Leyendo área de estudio: {shapefile_path}")
    bbox = _get_shapefile_bbox(shapefile_path, crs)
    logger.info(f"Bounding box del área: {tuple(round(b, 2) for b in bbox)}")

    # 2. Cargar y recortar por bbox cada NetCDF (recorte temprano)
    logger.info(f"Cargando índice de sequía: {drought_path}")
    ds_drought = load_and_standardize_netcdf(
        drought_path,
        var_mapping={drought_var: "drought_index"},
        crs=crs,
        bbox=bbox
    )

    logger.info(f"Cargando vegetación: {vegetation_path}")
    ds_veg = load_and_standardize_netcdf(
        vegetation_path,
        var_mapping={vegetation_var: "ndvi"},
        crs=crs,
        bbox=bbox
    )

    # 3. Recortar al período temporal común
    t_start = max(ds_drought.time.values[0], ds_veg.time.values[0])
    t_end   = min(ds_drought.time.values[-1], ds_veg.time.values[-1])

    ds_drought = ds_drought.sel(time=slice(t_start, t_end))
    ds_veg     = ds_veg.sel(time=slice(t_start, t_end))

    logger.info(f"Período común: {str(t_start)[:10]} → {str(t_end)[:10]}")
    logger.info(f"Pasos temporales SNDVI en período común: {len(ds_veg.time)}")

    # 4. Alinear temporalmente el índice a los timestamps del SNDVI
    # Búsqueda vectorizada del vecino más cercano con numpy argmin.
    # Más rápido que xr.interp() porque opera sobre enteros (ns desde epoch)
    # sin bucles Python ni operaciones sobre datos espaciales.
    # Robusto con cualquier resolución temporal de entrada.
    logger.info("Alineando índice de sequía a timestamps del SNDVI (argmin vectorizado)...")
    ds_drought = _align_time_nearest(
        ds_drought[["drought_index"]],
        ds_veg.time.values
    )

    # 5. Verificar alineación temporal
    n = min(len(ds_drought.time), len(ds_veg.time))
    if not np.all(ds_drought.time.values[:n] == ds_veg.time.values[:n]):
        warnings.warn(
            "Los timestamps no están perfectamente alineados tras la alineación. "
            "Verifica la compatibilidad temporal de los NetCDF de entrada."
        )

    # 6. Fusionar en un único Dataset (compat explícito para evitar FutureWarning)
    ds_merged = xr.merge([ds_drought, ds_veg[["ndvi"]]], compat="override")

    # 7. Cargar en memoria el resultado ya recortado por bbox (pequeño)
    ds_merged = ds_merged.compute()

    # 8. Recorte exacto al contorno del shapefile sobre el array pequeño
    ds_clipped = clip_dataset_to_polygon(ds_merged, shapefile_path)

    logger.info(f"Dataset fusionado y recortado: {dict(ds_clipped.sizes)}")
    return ds_clipped


def clip_dataset_to_polygon(dataset: xr.Dataset, shapefile_path: str) -> xr.Dataset:
    """
    Recorta un dataset multidimensional usando un polígono vectorial, validando
    la intersección.

    Esta función se aplica internamente al final de load_and_merge_datasets(), pero
    se mantiene pública para permitir recortar un dataset ya cargado a un shapefile
    diferente (por ejemplo, para subdividir un análisis por zonas).

    Parámetros:
    -----------
    dataset : xr.Dataset
        El dataset estandarizado.
    shapefile_path : str
        Ruta al archivo vectorial.

    Retorna:
    --------
    xr.Dataset
        Dataset recortado al contorno exacto del polígono.
    """
    # 1. Leer el vector
    polygons = gpd.read_file(shapefile_path)

    # 2. Alinear CRS
    if polygons.crs != dataset.rio.crs:
        polygons = polygons.to_crs(dataset.rio.crs)

    # 3. Disolver geometrías para evitar errores con FeatureCollection
    polygons = polygons.dissolve().reset_index(drop=True)

    # 4. Verificar solapamiento geográfico
    raster_bounds = box(*dataset.rio.bounds())
    try:
        union = polygons.union_all()
    except AttributeError:
        union = polygons.unary_union  # geopandas < 1.0

    if not union.intersects(raster_bounds):
        raise ValueError(
            "Error de topología: El polígono vectorial no se solapa con el área del raster."
        )

    # 5. Recortar al contorno exacto
    clipped_dataset = dataset.rio.clip(
        polygons.geometry, polygons.crs, all_touched=True, drop=True
    )

    return clipped_dataset