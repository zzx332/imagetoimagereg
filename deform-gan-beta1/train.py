from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.models import *
from network.discriminator import *
from network.generator import *
from network.gan import *
from network.registraion import *
from keras.utils import generic_utils as keras_generic_utils
from network.losses import *
import random
import cv2
import math
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import glob
import nibabel as nib
from keras.callbacks import TensorBoard
from network.transformer import Transformer_3D
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import argparse

config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument('--loadmodel', action='store_true')
parser.add_argument('--gan_path', type=str, default='/media/zzx/deform_GAN/models/Gen/',
                    help='Specifies a gan model path')
parser.add_argument('--disc_path', type=str, default='/media/zzx/deform_GAN/models/Disc/',
                    help='Specifies a disc model path')                    
parser.add_argument('--reg_path', type=str, default='/media/zzx/deform_GAN/models/Reg/',
                    help='Specifies a reg model path')
args = parser.parse_args()
# def get_data(data_path):
#     data_list = os.listdir(data_path)
#     data_num = len(data_list)
#     all_data = np.ndarray((data_num, data_shape[0], data_shape[1], data_shape[2], 1), "float32")
#     for n in range(data_num):
#         if n % 500 == 0:
#             print('Done: {0}/{1} images'.format(n, data_num))
#         vol = np.fromfile(data_path + "%05d.raw" % n, dtype="float")
#         vol.shape = data_shape
#         all_data[n, :, :, :, 0] = vol
#     return all_data
def dice(seg1, seg2, labels=None, nargout=1,data_shape = data_shape):
    vol1 = np.zeros(data_shape)
    vol2 = np.zeros(data_shape)
    vol1 = seg1[0,:,:,:,0]
    vol2 = seg2[0,:,:,:,0]


    '''
    Dice [1] volume overlap metric
    The default is to *not* return a measure for the background layer (label = 0)
    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.
    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)
    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom
    d = np.max(dicem)

    if nargout == 1:
        return  d

    else:
        return (dicem, labels)

def get_data(data_path):
    img_path = sorted(glob.glob(data_path+"*"))
    data_num = len(img_path)
    all_data = np.ndarray((data_num, data_shape[0], data_shape[1], data_shape[2], 1), "float32")
    for n in range(data_num):
        img=nib.load(img_path[n]).get_data()
        img = np.transpose(img, (2, 1, 0))
        all_data[n, :, :, :, 0] = img
    return all_data


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))

    if not args.loadmodel:
        reg_model = reg_unet()
        gen_model = gen_unet()
        disc_model = disc_net()

        gan_model = gan(generator_model=gen_model, discriminator_model=disc_model)
        reg_gen_model = reg_gen(reg_model=reg_model, generator_model=gen_model)

        # ------------------------
        # Define Optimizers
        opt_discriminator = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        opt_dcgan = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        opt_reg = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # ---------------------
        # Compile DCGAN
        gen_model.trainable = True
        reg_model.trainable = False
        disc_model.trainable = False
        # loss = [cc3D(), gradientSimilarity(win=[5, 5, 5]), 'binary_crossentropy']
        loss = ['mae', gradientSimilarity(win=[7, 7, 7]), 'binary_crossentropy']
        # loss_weights = [100, 0, 1]
        loss_weights = [100, 5, 1]
        # loss_weights = [100, 3, 1]
        gan_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        # ---------------------
        # COMPILE DISCRIMINATOR
        disc_model.trainable = True
        disc_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

        # ---------------------
        # COMPILE REG_MODEL
        disc_model.trainable = False
        reg_model.trainable = True
        gen_model.trainable = False
        loss_reg = [gradientSimilarity(win=[7, 7, 7]), gradientLoss('l2'), cc3D(win=[7, 7, 7]),"mse","mae"]
        # loss_weights_reg = [1, 0.5, 0.5]
        # loss_weights_reg = [1, 0.7, 1.0]
        loss_weights_reg = [2, 1.0, 1.0,0,0]
        reg_gen_model.compile(loss=loss_reg, loss_weights=loss_weights_reg, optimizer=opt_reg)
    else:
        print("-------------------------loadmodel--------------------------------")
        gan_path = sorted(glob.glob(args.gan_path+"*"))
        disc_path = sorted(glob.glob(args.disc_path+"*"))
        reg_path = sorted(glob.glob(args.reg_path+"*"))
        gan_model = load_model(gan_path[-1])
        disc_model = load_model(disc_path[-1])
        reg_gen_model = load_model(reg_path[-1])

    # reg_model.load_weights("models/model_at_epoch_1.h5")
    # gen_model.load_weights("models/Gen/model_at_epoch_0.h5")
    # disc_model.load_weights("models/Disc/model_at_epoch_0.h5")

    ## mr 是moving image，ct是fixed image
    
    ##导入数据batch
    mr_data = get_data(t1_path)
    ct_data = get_data(t2_path)
    gt_t1train = get_data(gtt1_path)
    segtrain = get_data(seg_path)
    seggttrain = get_data(segwarped_path)


    gt_t1 = get_data(val_gt)
    val_t1 = get_data(val_t1path)
    val_t2 = get_data(val_t2path)
    segval = get_data(segval_path)
    seggtval = get_data(segwarpedval_path)


    Y_true_batch = np.ones((batch_size, 1), dtype="float32")
    Y_fake_batch = np.zeros((batch_size, 1), dtype="float32")
    y_gen = np.ones((batch_size, 1), dtype="float32")
    zero_flow = np.zeros((batch_size, data_shape[0], data_shape[1], data_shape[2], 3), dtype="float32")
#---------------------------------------------------train ------------------------------------------------------
    gan_loss = tf.placeholder(tf.float32, [],name='gan_loss') 
    discr_loss = tf.placeholder(tf.float32, [],name='discr_loss')
    regis_loss = tf.placeholder(tf.float32, [],name='regis_loss')
    train_mse = tf.placeholder(tf.float32, [],name='train_mse')
    train_dice = tf.placeholder(tf.float32, [],name='train_dice')

    val_mse = tf.placeholder(tf.float32, [],name='val_mse') 
    val_dice = tf.placeholder(tf.float32, [],name='val_dice') 
    
    #可视化训练集、验证集的loss、acc、四个指标，均是标量scalers
    tf.summary.scalar("gan_loss", gan_loss) 
    tf.summary.scalar("discr_loss", discr_loss) 
    tf.summary.scalar("regis_loss", regis_loss) 
    tf.summary.scalar("train_mse", train_mse) 
    tf.summary.scalar("train_dice", train_dice)

    tf.summary.scalar("val_mse", val_mse)
    tf.summary.scalar("val_dice", val_dice)
  
    merge=tf.summary.merge_all()
    train_dice_list = []
    train_mse_list = []
    step = 0
    with tf.Session() as sess:
        logdir = './logs'        
        writer = tf.summary.FileWriter(logdir, sess.graph) 

        for ep in range(epochs):
            print("epochs:" + str(ep))
            progbar = keras_generic_utils.Progbar(train_num)
            for mini_batch in range(0, train_num, batch_size):
                # -----------------------------------train discriminator-------------------------------------------
                disc_model.trainable = True
                #### brats load
                # idx_mr = np.random.choice(mr_data.shape[0], batch_size, replace=False)
                # mr_batch = mr_data[idx_mr]
                # ct_batch = ct_data[idx_mr]
                #### chaos load
                idx_mr = np.random.choice(mr_data.shape[0], batch_size, replace=False)
                mr_batch = mr_data[idx_mr]
                idx_ct = np.random.choice(ct_data.shape[0], batch_size, replace=False)
                ct_batch = ct_data[idx_ct]
                #### 
                seg_batch = segtrain[idx_mr]
                seggt_batch = seggttrain[idx_mr]
                src_t, flow, ct_gen, src_j, seg_warped= reg_gen_model.predict([mr_batch, ct_batch, seg_batch])
                ########delet
                # dice_test = dice(seggt_batch,seg_batch,data_shape=data_shape)
                if random.randint(0, 1) == 0:
                    X_disc_batch = np.concatenate((ct_batch, ct_gen), axis=0)
                    Y_disc_batch = np.concatenate((Y_true_batch, Y_fake_batch), axis=0)
                else:
                    X_disc_batch = np.concatenate((ct_gen, ct_batch), axis=0)
                    Y_disc_batch = np.concatenate((Y_fake_batch, Y_true_batch), axis=0)

                disc_loss = disc_model.train_on_batch(X_disc_batch, Y_disc_batch)

                # --------------------------------------train generator-------------------------------------------
                disc_model.trainable = False
                reg_model.trainable = False
                gen_model.trainable = True
                ### brats
                # idx_mr = np.random.choice(mr_data.shape[0], batch_size, replace=False)

                # mr_batch = mr_data[idx_mr]
                # ct_batch = ct_data[idx_mr]
                # seg_batch = segtrain[idx_mr]
                # seggt_batch = seggttrain[idx_mr]
                ### chaos
                mr_batch = mr_data[idx_mr]
                ct_batch = ct_data[idx_ct]
                seg_batch = segtrain[idx_mr]
                seggt_batch = seggttrain[idx_ct]
                #####
                mr_t, flow, sim, mr_j, seg_warped = reg_gen_model.predict([mr_batch, ct_batch, seg_batch])
                seg_warped[seg_warped>0]=1
                gen_loss = gan_model.train_on_batch(mr_t, [ct_batch, mr_t, y_gen])
                # --------------------------------------train dice-------------------------------------------------
                train_dice_1 = dice(seg_warped,seggt_batch,data_shape = data_shape)
                train_dice_list.append(train_dice_1)
                ##### brats
                # if (mini_batch % 3000 == 0):
                #     warped_save = np.zeros(data_shape)
                #     warped_save = mr_t[0,:,:,:,0]
                #     warped_save = np.transpose(warped_save, (2, 1, 0))
                #     warped_vol = nib.Nifti1Image(warped_save,affine = None)
                #     nib.save(warped_vol, './results/train/%04d_%02dt1.nii.gz' % (mini_batch,idx_mr)) 
                #     gen_save = np.zeros(data_shape)
                #     gen_save = sim[0,:,:,:,0]
                #     gen_save = np.transpose(gen_save, (2, 1, 0))
                #     gen_vol = nib.Nifti1Image(gen_save,affine = None)
                #     nib.save(gen_vol, './results/train/%04d_%02dt1_gen.nii.gz' % (mini_batch,idx_mr))                       
                # --------------------------------------train reg-------------------------------------------------
                disc_model.trainable = False
                reg_model.trainable = True
                gen_model.trainable = False
                ### brats
                # idx_mr = np.random.choice(mr_data.shape[0], batch_size, replace=False)
                # mr_batch = mr_data[idx_mr]
                # ct_batch = ct_data[idx_mr]
                ### chaos
                mr_batch = mr_data[idx_mr]
                ct_batch = ct_data[idx_ct]
                ####

                gtt1_bach = gt_t1train[idx_mr]
                seg_batch = segtrain[idx_mr]
                seggt_batch = seggttrain[idx_mr]
                reg_loss = reg_gen_model.train_on_batch([mr_batch, ct_batch, seg_batch], [ct_batch, zero_flow, ct_batch, gtt1_bach,gtt1_bach])
                # --------------------------------------train mse----------------------------------------------------------
                train_mse_1 = reg_loss[4].tolist()
                train_mse_list.append(train_mse_1)

                if (mini_batch % 235 == 0):
                    train_dice_ = np.mean(train_dice_list)
                    train_mse_ = np.mean(train_mse_list)
                    train_dice_list = []
                    train_mse_list = []
                # --------------------------------------val---------------------------------------------
                if (mini_batch % 50 == 0):
                    val_dicelist = []
                    val_mselist = []
                    ###### brats
                    # for idx_mr in range(50):
                    #     # idx_mr = np.random.choice(val_t1.shape[0], batch_size, replace=False)
                    #     valt1_batch = val_t1[[idx_mr]]
                    #     valt2_batch = val_t2[[idx_mr]]
                    #     valgt_batch = gt_t1[[idx_mr]]
                    #     segval_batch = segval[[idx_mr]]
                    #     seggtval_batch = seggtval[[idx_mr]]
                    #     mr_t, flow, sim, mr_j, segval_warped = reg_gen_model.predict([valt1_batch, valt2_batch, segval_batch])
                    #     regval_loss = reg_gen_model.test_on_batch([valt1_batch, valt2_batch, segval_batch], [valt2_batch, zero_flow, valt2_batch, valgt_batch,valgt_batch])
                    #     segval_warped[segval_warped>0] = 1
                    #     val_dice_1 = dice(segval_warped,seggtval_batch,data_shape = data_shape)
                    #     val_dicelist.append(val_dice_1)
                    #     # segval_warped[[0,1],:,:,:,:]=segval_warped[[1,0],:,:,:,:]
                    #     # seggtval_batch[[0,1],:,:,:,:]=seggtval_batch[[1,0],:,:,:,:]
                    #     # val_dice_2 = dice(segval_warped,seggtval_batch,data_shape = data_shape)
                    #     # dicelist.append(val_dice_2)
                    #     val_mse_1 = regval_loss[4].tolist()
                    #     val_mselist.append(val_mse_1)
                    ###### chaos
                    for val_mr in range(val_t1.shape[0]):
                        for val_ct in range(val_t2.shape[0]):
                            valt1_batch = val_t1[[val_mr]]
                            valt2_batch = val_t2[[val_ct]]
                            valgt_batch = gt_t1[[val_mr]]
                            segval_batch = segval[[val_mr]]
                            seggtval_batch = seggtval[[val_ct]]
                            mr_t, flow, sim, mr_j, segval_warped = reg_gen_model.predict([valt1_batch, valt2_batch, segval_batch])
                            regval_loss = reg_gen_model.test_on_batch([valt1_batch, valt2_batch, segval_batch], [valt2_batch, zero_flow, valt2_batch, valgt_batch,valgt_batch])
                            segval_warped[segval_warped>0] = 1
                            val_dice_1 = dice(segval_warped,seggtval_batch,data_shape = data_shape)
                            val_dicelist.append(val_dice_1)
                            # segval_warped[[0,1],:,:,:,:]=segval_warped[[1,0],:,:,:,:]
                            # seggtval_batch[[0,1],:,:,:,:]=seggtval_batch[[1,0],:,:,:,:]
                            # val_dice_2 = dice(segval_warped,seggtval_batch,data_shape = data_shape)
                            # dicelist.append(val_dice_2)
                            val_mse_1 = regval_loss[4].tolist()
                            val_mselist.append(val_mse_1)
                    ########
                    val_dice_ = np.mean(val_dicelist)
                    val_mse_ = np.mean(val_mselist)

                #############   保存测试图像     
                if (mini_batch % 1000 == 0):
                    ### brats
                    # idx_mr = np.random.choice(val_t1.shape[0], batch_size, replace=False)
                    # valt1_batch = val_t1[idx_mr]
                    # valt2_batch = val_t2[idx_mr]
                    # valgt_batch = gt_t1[idx_mr]
                    # segval_batch = segval[idx_mr]
                    # seggtval_batch = seggtval[idx_mr]
                    ###### chaos
                    idx_mr = np.random.choice(val_t1.shape[0], batch_size, replace=False)
                    idx_ct = np.random.choice(val_t2.shape[0], batch_size, replace=False)
                    valt1_batch = val_t1[idx_mr]
                    valt2_batch = val_t2[idx_ct]
                    valgt_batch = gt_t1[idx_mr]
                    segval_batch = segval[idx_mr]
                    seggtval_batch = seggtval[idx_ct]

                
                    mr_t, flow, sim, mr_j, segval_warped = reg_gen_model.predict([valt1_batch, valt2_batch, segval_batch])
                    regval_loss = reg_gen_model.test_on_batch([valt1_batch, valt2_batch, segval_batch], [valt2_batch, zero_flow, valt2_batch, valgt_batch,valgt_batch])                    
                    
                    warped_save = np.zeros(data_shape)
                    warped_save = mr_t[0,:,:,:,0]
                    warped_save = np.transpose(warped_save, (2, 1, 0))
                    warped_vol = nib.Nifti1Image(warped_save,affine = None)
                    ###################### dir change
                    nib.save(warped_vol, './results_chaos/test/%04d_%02dto%02dt1.nii.gz' % (mini_batch,idx_mr,idx_ct))  
                    gen_save = np.zeros(data_shape)
                    gen_save = sim[0,:,:,:,0]
                    gen_save = np.transpose(gen_save, (2, 1, 0))
                    gen_vol = nib.Nifti1Image(gen_save,affine = None)
                    nib.save(gen_vol, './results_chaos/test/%04d_%02dto%02dt1_gen.nii.gz' % (mini_batch,idx_mr,idx_ct))  
            # print losses
                D_log_loss = disc_loss
                mae_loss = gen_loss[1].tolist()
                ngf_gen_loss = gen_loss[2].tolist()
                gen_log_loss = gen_loss[3].tolist()
                ngf_reg_loss = reg_loss[1].tolist()
                flow_loss = reg_loss[2].tolist()
                cc_loss = reg_loss[3].tolist()
                mse_loss = reg_loss[4].tolist()

                gan_loss_ = mae_loss*100+ngf_gen_loss*5+ngf_reg_loss
                discr_loss_ = D_log_loss
                regis_loss_ = ngf_reg_loss*2+flow_loss+cc_loss

                summary=sess.run(merge,feed_dict={train_mse:train_mse_,train_dice:train_dice_,val_dice:val_dice_,val_mse:val_mse_, gan_loss:gan_loss_,discr_loss:discr_loss_, regis_loss:regis_loss_ })
                writer.add_summary(summary,step)
                step+=batch_size

                if (mini_batch % 1 == 0):
                    progbar.add(batch_size, values=[("Dis", D_log_loss),
                                                    ("MAE", mae_loss),
                                                    ("NGF_gen", ngf_gen_loss),
                                                    ("FAKE", gen_log_loss),
                                                    ("NGF_reg", ngf_reg_loss),
                                                    ("FLOW", flow_loss),
                                                    ("CC", cc_loss),
                                                    ("MSE", mse_loss),
                                                    ("DICE", train_dice_),
                                                    ("DICE_test",val_dice_)
                                                    ])

            # save models
                # if (mini_batch % 1000 == 0):
                #     reg_model.save('models/Reg/model_epoch_%02d_step_%04d.h5' % (ep, mini_batch))
                #     disc_model.save('models/Disc/model_epoch_%02d_step_%04d.h5' % (ep, mini_batch))
                #     gen_model.save('models/Gen/model_epoch_%02d_step_%04d.h5' % (ep, mini_batch))
