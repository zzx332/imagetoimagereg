from keras.layers import Input, Lambda
from keras.models import Model
from parameters import *


def gan(generator_model, discriminator_model):
    src = Input(shape=data_shape + (1,))
    sim = generator_model(src)
    dcgan_output = discriminator_model(sim)
    dc_gan = Model(inputs=src, outputs=[sim, sim, dcgan_output])
    return dc_gan


def reg_gen(reg_model, generator_model):
    src = Input(shape=data_shape + (1,))
    tgt = Input(shape=data_shape + (1,))
    seg = Input(shape=data_shape + (1,))

    src_t, flow, seg_warped = reg_model([src, tgt, seg])
    sim = generator_model(src_t)

    dc_gan = Model(inputs=[src, tgt, seg], outputs=[src_t, flow, sim, src_t, seg_warped])
    return dc_gan
