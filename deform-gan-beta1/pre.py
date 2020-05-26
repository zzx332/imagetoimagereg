import numpy as np
import scipy
from scipy import misc
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import nibabel as nib
import glob
import skimage.exposure

def wl_normalization(img):
    img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))
    return img.astype(np.uint8)


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    # indices = np.reshape(x*0.8, (-1, 1)), np.reshape(y*0.8, (-1, 1))
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    # print(indices[0].shape)
    return indices
    # return map_coordinates(image, indices, order=1).reshape(shape)


q = 0
img_path = []
HGG_path = sorted(glob.glob("/media/zzx/data/2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG/*"))
LGG_path = sorted(glob.glob("/media/zzx/data/2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training/LGG/*"))
img_path=HGG_path+LGG_path
idx = np.random.choice(285, 285, replace=False)
for i in img_path:
    t1path = i+"/"+i.split('/')[-1]+"_t1.nii.gz"
    t2path = i+"/"+i.split('/')[-1]+"_t2.nii.gz"
    segpath = i+"/"+i.split('/')[-1]+"_seg.nii.gz"
    imgt1=nib.load(t1path).get_data()
    imgt2=nib.load(t2path).get_data()
    seg=nib.load(segpath).get_data()
    imgt1 = imgt1[20:220,20:220,:]
    imgt2 = imgt2[20:220,20:220,:]
    seg = seg[20:220,20:220,:]
    seg[seg>0]=1
    orit1 = scipy.ndimage.zoom(imgt1, np.array((128,128,96)) / np.array(imgt1.shape), order = 1)
    oriseg = scipy.ndimage.zoom(seg, np.array((128,128,96)) / np.array(imgt2.shape), order = 0)
    orit1 = wl_normalization(orit1) /255.0
    orit1 = orit1[8:120,:,:]
    oriseg = oriseg[8:120,:,:]
    # print(img)
    # x = np.zeros((112,128))
    # y = np.zeros((112,128))
    # x = np.zeros((220,220))
    # y = np.zeros((220,220))
    # z = np.zeros((220,220))
    x = np.zeros((200,200))
    y = np.zeros((200,200))
    z = np.zeros((200,200))

    indice = elastic_transform(x,x.shape[1]*2,x.shape[1]*0.08)

    for i in range(imgt1.shape[2]):
        x[:,:] = imgt1[:,:,i]
        y[:,:] = imgt2[:,:,i]
        z[:,:] = seg[:,:,i]
        elastic_imgt1 = map_coordinates(x, indice, order=1).reshape(x.shape)
        elastic_imgt2 = map_coordinates(y, indice, order=1).reshape(y.shape)
        elastic_seg = map_coordinates(z, indice, order=0).reshape(z.shape)
        imgt1[:,:,i] = elastic_imgt1
        imgt2[:,:,i] = elastic_imgt2
        seg[:,:,i] = elastic_seg

    imgt1 = scipy.ndimage.zoom(imgt1, np.array((128,128,96)) / np.array(imgt1.shape), order = 1)
    imgt2 = scipy.ndimage.zoom(imgt2, np.array((128,128,96)) / np.array(imgt2.shape), order = 1)
    seg = scipy.ndimage.zoom(seg, np.array((128,128,96)) / np.array(seg.shape), order = 0)
    imgt1 = wl_normalization(imgt1) / 255.0
    imgt2 = wl_normalization(imgt2) / 255.0
    imgt1 = imgt1[8:120,:,:]
    imgt2 = imgt2[8:120,:,:]
    seg = seg[8:120,:,:]

    # imgt1 = np.transpose(imgt1, (2, 1, 0))
    # imgt2 = np.transpose(imgt2, (2, 1, 0))
    # q+=1

    if q<236:
        path = "train"
    else:
        path = "test"
        # q = 1

    movvol = nib.Nifti1Image(imgt1,affine = None)
    nib.save(movvol, '/media/zzx/data/rebrats18/%s/t1/%03dt1.nii.gz' % (path,idx[q]))
    fixvol = nib.Nifti1Image(imgt2,affine = None)
    nib.save(fixvol, '/media/zzx/data/rebrats18/%s/t2/%03dt2.nii.gz' % (path,idx[q]))
    ori = nib.Nifti1Image(orit1,affine = None)
    nib.save(ori, '/media/zzx/data/rebrats18/%s/orit1/%03dorit1.nii.gz' % (path,idx[q]))        
    segori = nib.Nifti1Image(oriseg,affine = None)
    nib.save(segori, '/media/zzx/data/rebrats18/%s/oriseg/%03doriseg.nii.gz' % (path,idx[q])) 
    segvol = nib.Nifti1Image(seg,affine = None)
    nib.save(segvol, '/media/zzx/data/rebrats18/%s/seg/%03dseg.nii.gz' % (path,idx[q]))  
    q+=1 
    print(q)
# scipy.misc.imsave('/media/zzx/elastic.jpg',elastic_img)