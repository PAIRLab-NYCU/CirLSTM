# CT package
from pydicom import dcmread
import pydicom
import astra

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import glob
import os
import cv2
from tqdm import tqdm

# training_patient = ['C004', 'L004', 'L006', 'L014', 'N005', 'N012', 'N024', 'N030', 'N047', 'N051', 'N053', 'N056', 'N072']
training_patient = ['Toshiba']
image_folder = "../Toshiba_images/"
folder = glob.glob(image_folder)

def img2sinogram(img, circle_num=1152, h=64, w=512):
    y_max = circle_num * h # Horizontal axis length
    theta = np.linspace(0., 2*np.pi, y_max, endpoint=False)
    
    vol_geom = astra.create_vol_geom(ds.Rows, ds.Columns) # create a volumn format
    proj_geom = astra.create_proj_geom('parallel', 1., ds.Rows, theta) # create a projection format
    projector_id = astra.create_projector('cuda', proj_geom, vol_geom) # create a projector format

    # create sinogram
    sinogram_id, sinogram = astra.create_sino(img, projector_id)
    proj = np.reshape(sinogram, (circle_num, h, w)) # [number, 64, 512]
    
    # delete id
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(projector_id)
    return proj

def rmmin(img, ds):
    windowCenter = ds.WindowCenter
    windowWidth = ds.WindowWidth
    if type(windowCenter) == pydicom.multival.MultiValue and type(windowWidth) == pydicom.multival.MultiValue:
        windowCenter = windowCenter[0]
        windowWidth = windowWidth[0]
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    img[img < minWindow] = 0
    return img

for p in training_patient:
    folder_name = os.path.join(image_folder, "{}".format(p))
    print(folder_name)

    npy_folder = folder_name.replace('images', 'npy')
    print(npy_folder)

    data_list = glob.glob(os.path.join(folder_name, '*'))

    for path in tqdm(data_list):
        ds = dcmread(path, force=True)
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        try:
            img = ds.pixel_array
        except:
            print("!!!")
            raise
            continue
#         img = rmmin(img, ds)
        proj = img2sinogram(img)
        dicomname = path.split('/')[-1].replace('.dcm', '')
        dicomfolder = os.path.join(npy_folder, dicomname)
        if not os.path.isdir(dicomfolder):
            os.makedirs(dicomfolder)
        for i, p in enumerate(proj):
            filename = '{}.npy'.format(i)
            npy_name = os.path.join(dicomfolder, filename)
            np.save(npy_name, p)