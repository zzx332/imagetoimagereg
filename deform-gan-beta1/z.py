import glob
import nibabel as nib
import numpy as np

data_shape = (96, 128, 112)
t1_path = "/media/zzx/data/brats18/train/t1/"
batch_size = 2

def get_data(data_path):
    img_path = sorted(glob.glob(data_path+"*"))
    data_num = len(img_path)
    all_data = np.ndarray((data_num, data_shape[0], data_shape[1], data_shape[2], 1), "float32")
    for n in range(data_num):
        img=nib.load(img_path[n]).get_data()
        img = np.transpose(img, (2, 1, 0))
        all_data[n, :, :, :, 0] = img
    return all_data
mr_data = get_data(t1_path)
idx_mr = np.random.choice(mr_data.shape[0], batch_size, replace=False)

mr_batch = mr_data[idx_mr]
print(idx_mr,mr_batch.shape)