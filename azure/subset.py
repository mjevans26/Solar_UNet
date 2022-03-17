import rasterio as rio
import json
from rasterio.windows import Window
import numpy as np

with rio.open('./outputs/raw_unet256_Virginia_solar_Jun21.tif') as src:
    H,W = src.shape
    crs = src.crs
    windows = [Window(0,0, W//2, H//2), Window(0, H//2, W//2, H-(H//2)), Window(W//2, 0, W-(W//2), H//2), Window(W//2, H//2, W-(W//2), H-(H//2))]
    for i, window in enumerate(windows):
        subset = src.read(window = window)
        print(subset.shape)
        transform = src.window_transform(window)
        with rio.open(
            f'./outputs/VA2021_Jun21preds{i}.tif',
            'w',
            driver = 'GTiff',
            width = W,
            height = H,
            count = 1,
            dtype = subset.dtype,
            crs = crs,
            transform = transform) as dst:
            dst.write(subset)
            
