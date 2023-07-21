import numpy as np
import math
import astra
import pydicom

def to_HU(img, ds):
    slope = ds.RescaleSlope
    intercept = ds.RescaleIntercept
    img_hu = slope * img.astype(np.float64)
    img_hu = img_hu.astype(np.int16)
    img_hu += np.int16(intercept)
    return img_hu

def windowing(img_hu, ds):
    windowCenter = ds.WindowCenter
    windowWidth = ds.WindowWidth
    if type(windowCenter) == pydicom.multival.MultiValue and type(windowWidth) == pydicom.multival.MultiValue:
        windowCenter = windowCenter[1]
        windowWidth = windowWidth[1]
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    newimg = (img_hu - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return newimg

def reconstruction(sino_id, algo='FBP_CUDA'):
    vol_geom = astra.create_vol_geom(512, 512)
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict(algo)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    # Run the algorithm
    astra.algorithm.run(alg_id)

    # Get the result
    rec = astra.data2d.get(rec_id)
    # delete id
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)
    
    return rec

def create_sino_id(img, x_max=1152, mode='gt'):
    theta = np.linspace(0., 2*np.pi, 64*x_max, endpoint=False)
    proj_geom = astra.create_proj_geom('parallel', 1., 512, theta)
    sino_id = astra.data2d.create('-sino', proj_geom, img);
#     print("{} sinogram id: {}".format(mode, sino_id))
    d = astra.data2d.get(sino_id)
#     print("{} singram shape: {}".format(mode, np.shape(d)))
    
    return sino_id, d

def RMSE(img1, img2):
    diff = img1 - img2
    square = np.power(diff, 2)
    add = square.sum()
    mean = add.mean()
    root = np.sqrt(mean)
    return root

def calc_psnr(img1, img2, PIXEL_MAX):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def img2sinogram(img, circle_num=1152, h=64, w=512):
    y_max = circle_num * h # Horizontal axis length
    theta = np.linspace(0., 2*np.pi, y_max, endpoint=False)
    
    vol_geom = astra.create_vol_geom(ds.Rows, ds.Columns) # create a volumn format
    proj_geom = astra.create_proj_geom('parallel', 1, ds.Rows, theta) # create a projection format
    projector_id = astra.create_projector('cuda', proj_geom, vol_geom) # create a projector format

    # create sinogram
    sinogram_id, sinogram = astra.create_sino(img, projector_id)
    proj = np.reshape(sinogram, (circle_num, h, w)) # [number, 64, 512]
    
    # delete id
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(projector_id)
    return sinogram