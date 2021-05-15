import os
import sys
import zipfile
from rasterio.merge import merge
import rasterio

def extract_files(path):
    '''Extracts and create list of extracted tiff files'''
    if 'Extract' not in os.listdir(path):
        os.mkdir(os.path.join(path, 'Extract'))
    files = [n for n in os.listdir(path) if (n.endswith('.zip'))]
    for n in files:
        with zipfile.ZipFile(os.path.join(path,n), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(path, 'Extract'))
    tiffs = [n for n in os.listdir(os.path.join(path, 'Extract')) if (n.endswith('.tif')) | (n.endswith('.asc'))]
    return tiffs


def to_mosaic(path, tif_list):
    '''mosaic file based on list of tiff files'''
    src_files_to_mosaic = []
    for fp in tif_list:
        src = rasterio.open(os.path.join(path, 'Extract', fp))
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "dtype": 'float32',
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans})
    with rasterio.open(os.path.join(path, 'Extract', 'mosaic'), "w", **out_meta) as dest:
        dest.write(mosaic)

path = str(sys.argv[1]).strip()
files = extract_files(path)
to_mosaic(path, files)
