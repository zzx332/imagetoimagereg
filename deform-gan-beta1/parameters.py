# t1_path = "/media/zzx/data/rebrats18/train/orit1/"
# t2_path = "/media/zzx/data/rebrats18/train/t2/"
t1_path = "/media/zzx/data/deform_chaos/train/mr/"
t2_path = "/media/zzx/data/deform_chaos/train/ct/"
gtt1_path = "/media/zzx/data/rebrats18/train/t1/"
seg_path = "/media/zzx/data/rebrats18/train/oriseg/"
segwarped_path = "/media/zzx/data/rebrats18/train/seg/"

val_gt = "/media/zzx/data/rebrats18/test/t1/"
# val_t1path = "/media/zzx/data/rebrats18/test/orit1/"
# val_t2path = "/media/zzx/data/rebrats18/test/t2/"
# segval_path = "/media/zzx/data/rebrats18/test/oriseg/"
# segwarpedval_path = "/media/zzx/data/rebrats18/test/seg/"
val_t1path = "/media/zzx/data/deform_chaos/test/mr/vol/"
val_t2path = "/media/zzx/data/deform_chaos/test/ct/vol/"
segval_path = "/media/zzx/data/deform_chaos/test/mr/seg/"
segwarpedval_path = "/media/zzx/data/deform_chaos/test/ct/seg/"

data_shape = (96, 128, 112)
# data_shape = (48, 128, 112)

epochs = 20
batch_size = 1
train_num = 10000

