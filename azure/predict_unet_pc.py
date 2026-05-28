import argparse
import os
# import glob
from os.path import join
import sys
from io import BytesIO
import multiprocessing
# from datetime import datetime
from pathlib import Path
import json
import numpy as np
from azure.storage.blob import ContainerClient

from osgeo import gdal
from shapely.geometry import Polygon, Point
import geopandas as gpd
# import dask_geopandas
import dask.dataframe as dd
# import zarr
import pandas as pd
import xarray as xr
import rioxarray
import rasterio as rio
from rasterio.transform import xy, Affine
from rasterio.windows import Window, from_bounds, transform, shape
from rasterio.vrt import WarpedVRT
from pyproj import CRS
# import planetary_computer
# import pystac_client
# import stac_vrt
# import tempfile
from scipy.ndimage import zoom
# from tensorflow.keras import models
import tensorflow as tf

print('contents of root dir', os.listdir('.'))

DIR = Path().resolve()
sys.path.append(str(DIR/'scv'))

from utils import model_tools, pc_tools, raster_tools, processing
from importlib import reload

def get_idx(filename):
    """Return the '{x}_{y}' identifying string at the end of a numpy file"""
    path, ext = os.path.splitext(filename)
    if ext != '.json':
        base = os.path.basename(path)
        pieces = base.split('_')
        idx = '_'.join(pieces[-2:])
        return idx
    else:
        pass

def get_existing_files(path):
    generator = Path(path).glob('*.tiff')
    tiffs = [url for url in generator]
    ids = set([get_idx(tiff) for tiff in tiffs])
    return ids

def get_existing_blobs(container_client, path = 'test/train/label/'):
    """Return a list of '{x}_{y}' identifying strings from list of blobs"""
    generator = container_client.list_blobs(name_starts_with = path)
    blobs = [f'{blob.name}' for blob in generator]
    ids = set([get_idx(blob) for blob in blobs])
    return ids

def run(dates, aoi:gpd.GeoSeries, name:str, imgsz:int, buff:int, nclasses:int, nchannels:int, ssurgo_table, dem_file, container_client, weights = None, sas_token = None, existing = [], multi = False):
    imgsz = imgsz
    print('imgsz', imgsz)

    buff = buff
    print('buff', buff)

    side = imgsz + (buff*2)
    epoch_id = Path(weights).stem[-3:]

    # retrieve Sentinel-2 imagery from MPC 
    s2Img = pc_tools.get_s2_stac(aoi.geometry.iloc[0], dates) # xarray dataarray
        
    s2HWC = s2Img.transpose('y', 'x', 'band')
    s2Transform = naipHWC.rio.transform(recalc = True)
    s2Res = naipTransform[0]
    s2CRS = naipHWC.rio.crs
    s2EPSG = naipCRS.to_epsg()

    print('Sentinel-2 epsg', s2EPSG)
    print('Sentinel-2 res', s2Res)
    print('Sentinel-2 shape', s2HWC.shape)

    aoi_reproj = aoi.to_crs(naipCRS)
    bounds = aoi_reproj.total_bounds
    print('aoi bounds', bounds)

    window = from_bounds(*bounds, transform = s2Transform)
    H, W = shape(window)
    print(H,W)
  
    trimmed_transform = transform(window, s2Transform)

    # chip indices are the pixel coordinates within the aoi window
    chip_indices = raster_tools.generate_chip_indices(round(H), round(W), buff = buff, kernel = imgsz)
    print(f'{len(chip_indices)} chip indices')
    # chip indices shifted are pixel coordinates of the original s2 img
    chip_indices_shifted = [(round(i[0]+window.row_off), round(i[1]+window.col_off)) for i in chip_indices]
    # let's make a list of ssurgo indices all at once up front
    
    coords = gpd.GeoDataFrame({
      'indices':chip_indices,
      'geometry': [raster_tools.convert_poly_coords(geom = Point(x,y), affine_obj = trimmed_transform) for y,x in chip_indices],
      's2_coords':chip_indices_shifted,
      'idx':[f"{int(chip_indices[i][1])}_{int(chip_indices[i][0])}" for i in range(len(chip_indices))]
      },
      geometry = 'geometry',
      crs=s2CRS)
    print(f'{len(coords)} Sentinel-2 pts')

    # create a mixer dictionary so we can reconstruct the outputs
    mixer = dict({
        "rows": H,
        "cols": W,
        "crs": s2EPSG,
        "size": imgsz,
        "transform": trimmed_transform
    })
    mixer_client = container_client.get_blob_client(f'data/predict/{name}/mixer.json')

    with BytesIO() as buffer:
        # json.dump(mixer, buffer).encode()
        buffer.write(json.dumps(mixer).encode())
        buffer.seek(0)
        mixer_client.upload_blob(buffer, overwrite = True) 

    m = model_tools.get_unet_model(
        nclasses = nclasses,
        nchannels = nchannels,
        filters = [32, 64, 128, 256],
        factors = [3,2,2,2],
        bias = None)

    if weights:
        if 'https' in weights: # if our weights are in blob storage
            weights = f'{weights}?{sas_token}'
            m = model_tools.get_blob_weights(m = m, hdf5_url = weights)
        else: # otherwise if weights on local file system
            m.load_weights(weights,by_name=True)
    m.summary()

    def predict_chips(row):
        index = row['indices']
        y, x = index
        print('S2 point', row['S2_coords'])
        Y, X = row['S2_coords']
        try:
            s2Chip = s2HWC[Y - buff:Y+imgsz+buff, X - buff:X + imgsz + buff, :].values
            assert s2Chip.shape == (side, side, 4), f'naip chip not expected shape ({naipChip.shape})'
            # get model predictions for current chip
            preds = m.predict(np.array([s2Chip]), verbose = 0) # return the probability of solar
            # trim the buffer from prediction trip
            prediction = preds[0][0,buff:(imgsz + buff),buff:(imgsz + buff),:] 
            print(prediction.shape)
            return prediction, y, x

        except (AssertionError, RuntimeError) as msg:
            print(msg)
            return None, None, None
    
    def predict_chips_rio(row):
        prediction, y, x = predict_chips(row)
        if prediction is None:
            print(f'skipping {row}')
            pass
        else:
            # stack = np.concatenate([prediction, naip[600:1800,600:1800,:], dem[600:1800,600:1800,:]], axis = -1)
            arr = np.moveaxis(prediction, -1, 0) #CHW
            C,H,W = arr.shape
            band_list = list(range(1,C+1))
            _, lat = xy(Affine(*mixer['transform']), np.arange(y,y+imgsz), np.repeat(0, imgsz))
            lon, _ = xy(Affine(*mixer['transform']), np.repeat(0, imgsz), np.arange(x, x + imgsz))

            da = xr.DataArray(
                arr,
                coords = {
                    'band': list(range(C)),
                    'y':lat,
                    'x':lon
                }
            )

            da.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace = True)
            # add spatial reference info
            da.rio.write_crs(f"epsg:{mixer['crs']}", inplace = True)
            # write to cog
            # da.rio.to_raster(f'//chesconse-fs/K/GIS/CBT_NonTidalWetlands/Analysis/Intersection_over_Union/{name}/unet{epoch_id}/tiff/{x}_{y}.tif', driver = 'GTiff', windowed = True)
            with BytesIO() as buffer:
                da.rio.to_raster(buffer, driver = "GTiff", windowed = True)
                buffer.seek(0)
                blob_client = container_client.get_blob_client(f'data/predict/{name}/unet{epoch_id}/tiff/{x}_{y}.tif')
                blob_client.upload_blob(buffer, overwrite = True)

    # identify sampling points that need dem data exported
    # existing = s1_blobs.intersection(naip_blobs, dem_blobs, ssurgo_blobs)
    print(existing)

    # subset coordinates to those falling within the aoi
    coords_within = [coords.geometry.within(aoi_reproj.geometry.iloc[0])]

    to_process = coords_within[~coords_within['idx'].isin(existing)] 
    print(len(coords_within), len(coords), len(existing)) 
    
    # Create a MirroredStrategy.
    gpus = tf.config.list_physical_devices('GPU')
    print(f'Number of devices: {len(gpus)}')

    # if len(gpus) > 0:
    #     for gpu in gpus:
    #         # tf.config.experimental.set_memory_growth(gpu, True)
            
    #         device_name = tf.test.gpu_device_name()
    #         print(f'Device name: {device_name}')
    #     strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{x}' for x in range(len(gpus))])
    #     # Open a strategy scope.
    #     with strategy.scope():
    #         to_process_ddf.apply(predict_chips_rio, axis = 1, meta = {'':None})
    #         # to_process.apply(predict_chips_rio, axis = 1)
    # else:
    if multi:
        to_process_ddf = dd.from_pandas(to_process, npartitions = multiprocessing.cpu_count())
        to_process_ddf.apply(predict_chips_rio, axis = 1, meta = {'':None}).compute()
    else:
        to_process.apply(predict_chips_rio, axis = 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', type = str, required = True)
    parser.add_argument('--multi', dest = 'multi', action = 'store_true', default = False)
    args = parser.parse_args()

    # Add PC dask cluster jupyterhub token to environemnt
    env_config = dotenv_values(".env")
    os.environ['JUPYTERHUB_API_TOKEN'] = env_config['JUPYTERHUB_API_TOKEN']
    os.environ['PC_SDK_SUBSCRIPTION_KEY'] = env_config['JUPYTERHUB_API_TOKEN']
    sas_token = env_config['SAS_TOKEN']

    # create special folders './outputs' and './logs' which automatically get saved
    os.makedirs('outputs', exist_ok = True)
    os.makedirs('logs', exist_ok = True)
    out_dir = './outputs'
    log_dir = './logs'

    # this contianer is connected to ArcPro
    container_client = ContainerClient.from_container_url(
        container_url = f'https://aiprojects.blob.core.windows.net/solar?{sas_token}')
    
    with open(args.run_config) as f:
        run_config = json.load(f)
    
    name = run_config["name"]
    # optionally add sas token if aoi is on azure
    aoi_url = run_config["aoi"]
    if 'http' in aoi_url:
        aoi_url = f'{aoi_url}?{sas_token}'
    else:
        pass
    print(aoi_url)

    aoi = gpd.read_file(aoi_url).to_crs(4326)#.geometry.iloc[0]
    local_dir = run_config["data_dir"]
    unet_vars = run_config['unet_vars']
    nchannels = sum([unet_vars[var]['nchannels'] for var in unet_vars.keys() if unet_vars[var]['files'] is not None])
    weights = run_config["weights"]
    epoch_id = Path(weights).stem[-3:]

    # get a list of our existing unique ids for which we have already made predictions
    existing = get_existing_blobs(container_client, f'data/predict/{name}/unet{epoch_id}/tiff/')   
    # existing = get_existing_files(f'{local_dir}/unet{epoch_id}/tiff/')
    print(f'already completed {len(existing)} chips')
    run(
        dates = run_config["dates"],
        imgsz = run_config["imgsz"],
        buff = run_config["buff"],
        aoi = aoi,
        name = name,
        nclasses = run_config['nclasses'],
        nchannels = nchannels,
        weights = weights,
        container_client = container_client,
        sas_token = sas_token,
        existing = existing,
        multi = args.multi
    )

    # # set azure credentials as environment variables - this lets gdal interface with blob storage
    os.environ["AZURE_STORAGE_CONNECTION_STRING"]=env_config["AZURE_STORAGE_CONNECTION_STRING"]
    # # get a list of the tif files we wrote
    blob_generator = container_client.list_blobs(name_starts_with = f'{local_dir}/unet{epoch_id}/tiff/')
    blobs = [blob.name for blob in blob_generator]
    tif_list = [f'/vsiaz/wetlands/{blob}' for blob in blobs if '.tif' in blob if '.tif' in blob]
    print('found', len(tif_list), 'tifs. writing vrt')
    # first write a vrt file aggregating all the tifs
    if len(tif_list) > 2000:
        vrt_file1 = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt1.vrt'
        print('writing VRT', vrt_file1)
        vrt = gdal.BuildVRT(vrt_file1, tif_list[0:500])
        vrt = None
        vrt_file2 = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt2.vrt'
        print('writing VRT', vrt_file2)
        vrt = gdal.BuildVRT(vrt_file2, tif_list[500:1000])
        vrt = None
        vrt_file3 = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt3.vrt'
        print('writing VRT', vrt_file3)
        vrt = gdal.BuildVRT(vrt_file3, tif_list[1000:1500])
        vrt = None
        vrt_file4 = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt4.vrt'
        print('writing VRT', vrt_file4)
        vrt = gdal.BuildVRT(vrt_file4, tif_list[1500:2000])
        vrt = None
        vrt_file5 = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt5.vrt'
        print('writing VRT', vrt_file5)
        vrt = gdal.BuildVRT(vrt_file5, tif_list[2000:2500])
        vrt = None
        vrt_file6 = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt6.vrt'
        print('writing VRT', vrt_file6)
        vrt = gdal.BuildVRT(vrt_file6, tif_list[2500:])
        vrt = None        
    else:
        vrt_file = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/tiff/vrt.vrt'
        print('writing VRT', vrt_file)
        vrt = gdal.BuildVRT(vrt_file, tif_list)
        vrt = None
    # # now build a COG from the VRT
    # cog_file = f'/vsiaz/solar/{local_dir}/unet{epoch_id}/cog.tif'
    # gdal.Translate(cog_file, vrt_file) # THIS IS BLOWING MEMORY
