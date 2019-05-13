import locale
import os
import sys
from os import path

locale.setlocale(locale.LC_ALL, '')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_DIR = path.dirname(path.dirname(path.abspath(__file__))) + '/'

import functools

from common.score import *
from common.data_loader import *
from common.logger import Logger

import numpy as np
import tensorflow as tf

cfg = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("bLoadCheckpoint", True, "bLoadCheckpoint")
tf.app.flags.DEFINE_string("sDataSet", "cifar10", "cifar10, mnist, flowers,tiny")
tf.app.flags.DEFINE_string("sResultTag", "test", "your tag for each test case")

##################################################### Objectives ##############################################################################################

tf.app.flags.DEFINE_string("sGAN_type", 'exp', "")
tf.app.flags.DEFINE_boolean("bPsiMax", False, "")
tf.app.flags.DEFINE_float("alpha", 1.0, "")

tf.app.flags.DEFINE_boolean("bLip", True, "")
tf.app.flags.DEFINE_boolean("bMaxGP", True, "")
tf.app.flags.DEFINE_string("sGP_type", 'cp', 'cp, gp, lp')
tf.app.flags.DEFINE_float("fWeightLip", 0.1, "")
tf.app.flags.DEFINE_float("fBufferBatch", 0.00, "")
tf.app.flags.DEFINE_float("fLipTarget", 1.0, "")  # for gp and lp

################################################# Learning Process ###########################################################################################

tf.app.flags.DEFINE_integer("iRun", 0, "")
tf.app.flags.DEFINE_integer("iMaxIter", 100000, "")
tf.app.flags.DEFINE_integer("iBatchSize", 64, "")
tf.app.flags.DEFINE_integer("GEN_BS_MULTIPLE", 2, "")

tf.app.flags.DEFINE_integer("iTrainG", 1, "")
tf.app.flags.DEFINE_integer("iTrainD", 5, "")

tf.app.flags.DEFINE_float("fLrIniG", 0.0002, "")
tf.app.flags.DEFINE_float("fLrIniD", 0.0002, "")
tf.app.flags.DEFINE_string("oDecayG", 'linear', "linear, exp, none")
tf.app.flags.DEFINE_string("oDecayD", 'linear', "linear, exp, none")

tf.app.flags.DEFINE_float("fBeta1", 0.0, "")
tf.app.flags.DEFINE_float("fBeta2", 0.9, "")
tf.app.flags.DEFINE_float("fEpsilon", 1e-8, "")

##################################################### Network Structure #######################################################################################

tf.app.flags.DEFINE_boolean("bTanh", True, "")
tf.app.flags.DEFINE_boolean("bBnG", True, "")

tf.app.flags.DEFINE_integer("iFSizeG", 3, "")
tf.app.flags.DEFINE_integer("iFSizeD", 3, "")

tf.app.flags.DEFINE_integer("iDimsG", 128, "")
tf.app.flags.DEFINE_integer("iDimsD", 128, "")

tf.app.flags.DEFINE_integer("iDimsC", 3, "")
tf.app.flags.DEFINE_integer("iDimsZ", 128, "")

tf.app.flags.DEFINE_integer("GPU", -1, "")
tf.app.flags.DEFINE_string("sResultDir", SOURCE_DIR + "result/sobolev/", "where to save the checkpoint and sample")

cfg(sys.argv)

############################################################################################################################################

GPU_ID = allocate_gpu(cfg.GPU)

assert cfg.fLrIniD == cfg.fLrIniG
lr_dict = {0.0002: '2e-4', 0.0001: '1e-4'}
weight_dict = {0.1: '0.1', 0.01: '0.01', 1.0: '1.0', 10.0: '10.0'}
cfg.sResultTag = 'tanh%d_gbn%d' % (cfg.bTanh, cfg.bBnG) + ('_max' if cfg.bMaxGP else '_') + cfg.sGP_type + weight_dict.get(cfg.fWeightLip) + '_buffer%.2f_linear%s_%d' %(cfg.fBufferBatch, lr_dict.get(cfg.fLrIniD), cfg.iMaxIter) + '' if cfg.iRun == 0 else '_run%d' % cfg.iRun

if cfg.bLip:
    cfg.sResultTag += '_lip'

cfg.sResultTag += '_' + cfg.sGAN_type

if cfg.sGAN_type in ['lsgan', 'hinge']:
    cfg.sResultTag += '_%.2f' % cfg.alpha

sTestName = cfg.sDataSet + ('_' + cfg.sResultTag if len(cfg.sResultTag) else "")
sTestCaseDir = cfg.sResultDir + sTestName + '/'
sSampleDir = sTestCaseDir + '/sample/'
sCheckpointDir = sTestCaseDir + 'checkpoint/'

makedirs(sSampleDir)
makedirs(sCheckpointDir)
makedirs(sTestCaseDir + 'source/code/')
makedirs(sTestCaseDir + 'source/common/')

logger = Logger()
logger.set_dir(sTestCaseDir)
logger.set_casename(sTestName)

logger.linebreak()
logger.log(sTestCaseDir)

commandline = ''
for arg in ['python3'] + sys.argv:
    commandline += arg + ' '
logger.log('\n' + commandline + '\n')

logger.log(str_flags(cfg.__flags))
logger.log('Using GPU%d\n' % GPU_ID)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

copydir(SOURCE_DIR + "code/", sTestCaseDir + 'source/code/')
copydir(SOURCE_DIR + "common/", sTestCaseDir + 'source/common/')
tf.logging.set_verbosity(tf.logging.ERROR)

############################################################################################################################################

is_training = tf.placeholder(bool, name='is_training')


def apply_conv(x, filters, kernel_size, he_init=True):
    initializer = tf.contrib.layers.variance_scaling_initializer(uniform=True) if he_init else tf.contrib.layers.xavier_initializer(uniform=True)
    return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, padding='SAME', kernel_initializer=initializer, data_format='channels_first')


def activation(x):
    return tf.nn.relu(x)


def identity(x):
    return x


def bn(x):
    return tf.layers.batch_normalization(x, axis=1, training=is_training)


def downsample(x):
    return tf.layers.average_pooling2d(x, 2, 2, data_format='channels_first')


def upsample(x):
    return tf.depth_to_space(tf.concat([x, x, x, x], axis=1), 2, data_format="NCHW")


def conv_meanpool(x, **kwargs):
    return downsample(apply_conv(x, **kwargs))


def meanpool_conv(x, **kwargs):
    return apply_conv(downsample(x), **kwargs)


def upsample_conv(x, **kwargs):
    return apply_conv(upsample(x), **kwargs)


def resblock(x, filters, kernel_size, resample=None, norm_fn=identity):
    if resample == 'down':
        conv_1 = functools.partial(apply_conv, filters=filters, kernel_size=kernel_size)
        conv_2 = functools.partial(conv_meanpool, filters=filters, kernel_size=kernel_size)
        conv_shortcut = functools.partial(conv_meanpool, filters=filters, kernel_size=1, he_init=False)

    elif resample == 'up':
        conv_1 = functools.partial(upsample_conv, filters=filters, kernel_size=kernel_size)
        conv_2 = functools.partial(apply_conv, filters=filters, kernel_size=kernel_size)
        conv_shortcut = functools.partial(upsample_conv, filters=filters, kernel_size=1, he_init=False)

    else:
        assert resample == None
        conv_1 = functools.partial(apply_conv, filters=filters, kernel_size=kernel_size)
        conv_2 = functools.partial(apply_conv, filters=filters, kernel_size=kernel_size)
        conv_shortcut = identity

    update = conv_1(activation(norm_fn(x)))
    update = conv_2(activation(norm_fn(update)))
    skip = conv_shortcut(x)
    return skip + update


def resblock_optimized(x, filters, kernel_size):
    update = apply_conv(x, filters=filters, kernel_size=kernel_size)
    update = conv_meanpool(activation(update), filters=filters, kernel_size=kernel_size)
    skip = meanpool_conv(x, filters=filters, kernel_size=1, he_init=False)
    return skip + update


def sample_z(n):
    noise = np.random.randn(n, cfg.iDimsZ)
    return noise


def generator(z, reuse):
    with tf.variable_scope('generator', reuse=reuse):
        norm_fn = bn if cfg.bBnG else identity
        with tf.variable_scope('pre_process'):
            z = tf.layers.dense(z, 4 * 4 * cfg.iDimsG)
            x = tf.reshape(z, [-1, cfg.iDimsG, 4, 4])

        with tf.variable_scope('x1'):
            x = resblock(x, cfg.iDimsG, cfg.iFSizeG, resample='up', norm_fn=norm_fn)  # 8
        with tf.variable_scope('x2'):
            x = resblock(x, cfg.iDimsG, cfg.iFSizeG, resample='up', norm_fn=norm_fn)  # 16
        with tf.variable_scope('x3'):
            x = resblock(x, cfg.iDimsG, cfg.iFSizeG, resample='up', norm_fn=norm_fn)  # 32

        with tf.variable_scope('post_process'):
            x = activation(norm_fn(x))
            x = apply_conv(x, cfg.iDimsC, cfg.iFSizeG, he_init=False)
            return tf.tanh(x) if cfg.bTanh else x


def discriminator(x, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
        with tf.variable_scope('pre_process'):
            x = resblock_optimized(x, cfg.iDimsD, cfg.iFSizeD)  # 16

        with tf.variable_scope('x1'):
            x = resblock(x, cfg.iDimsD, cfg.iFSizeD, resample='down')  # 8
        with tf.variable_scope('x2'):
            x = resblock(x, cfg.iDimsD, cfg.iFSizeD)  # 8
        with tf.variable_scope('x3'):
            x = resblock(x, cfg.iDimsD, cfg.iFSizeD)  # 8

        with tf.variable_scope('post_process'):
            x = activation(x)
            x = tf.reduce_mean(x, axis=[2, 3])
            x = tf.layers.dense(x, 1)
            return x


class EMAHelper(object):
    def __init__(self, decay=0.99, session=None, var_list=None):
        self.session = tf.get_default_session() if session is None else session
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if var_list is None else var_list
        self.ema = tf.train.ExponentialMovingAverage(decay=decay)
        self.apply = self.ema.apply(self.var_list)
        self.averages = [self.ema.average(var) for var in self.var_list]

    def average_dict(self):
        ema_averages_results = self.session.run(self.averages)
        return {var: value for var, value in zip(self.var_list, ema_averages_results)}


############################################################################################################################################

def load_dataset(dataset_name):
    if dataset_name == 'cifar10':
        return load_cifar10()
    if dataset_name == 'mnist':
        return load_mnist()
    if dataset_name == 'flowers':
        return load_flower()
    if dataset_name == 'tiny':
        return load_tiny_imagenet()


dataX, dataY, testX, testY = load_dataset(cfg.sDataSet)
real_gen = data_gen_epoch(dataX, cfg.iBatchSize)
cfg.iDimsC = np.shape(dataX)[1]

real_datas = tf.placeholder(tf.float32, (None,) + np.shape(dataX)[1:], name='real_data')
iter_datas = tf.placeholder(tf.float32, (None,) + np.shape(dataX)[1:], name='iter_data')

z = tf.placeholder(tf.float32, [None, cfg.iDimsZ], name='z')
fake_datas = generator(z, reuse=False)

fake_logits = discriminator(fake_datas, reuse=False)
real_logits = discriminator(real_datas, reuse=True)

if cfg.sGAN_type == 'lsgan':
    logger.log('using lsgan loss: %.2f' % cfg.alpha)
    dis_real_loss = tf.square(real_logits - cfg.alpha)
    dis_fake_loss = tf.square(fake_logits + cfg.alpha)
    gen_fake_loss = tf.square(fake_logits - cfg.alpha)

elif cfg.sGAN_type == 'x':
    logger.log('using wgan loss')
    dis_real_loss = -real_logits
    dis_fake_loss = fake_logits
    gen_fake_loss = -fake_logits

elif cfg.sGAN_type == 'sqrt':
    logger.log('using sqrt loss')
    dis_real_loss = tf.sqrt(tf.square(real_logits) + 1) - real_logits
    dis_fake_loss = tf.sqrt(tf.square(fake_logits) + 1) + fake_logits
    if cfg.bPsiMax:
        gen_fake_loss = tf.sqrt(tf.square(fake_logits) + 1) - fake_logits
    else:
        gen_fake_loss = -fake_logits

elif cfg.sGAN_type == 'log_sigmoid':
    logger.log('using log_sigmoid loss')
    dis_real_loss = -tf.log_sigmoid(real_logits)
    dis_fake_loss = -tf.log_sigmoid(-fake_logits)
    if cfg.bPsiMax:
        gen_fake_loss = -tf.log_sigmoid(fake_logits)
    else:
        gen_fake_loss = -fake_logits

elif cfg.sGAN_type == 'exp':
    logger.log('using exp loss')
    dis_real_loss = tf.exp(-real_logits)
    dis_fake_loss = tf.exp(fake_logits)
    if cfg.bPsiMax:
        gen_fake_loss = tf.exp(-fake_logits)
    else:
        gen_fake_loss = -fake_logits

elif cfg.sGAN_type == 'hinge':
    logger.log('using hinge loss: %.2f' % cfg.alpha)
    dis_real_loss = -tf.minimum(tf.constant(0.), real_logits - tf.constant(cfg.alpha))
    dis_fake_loss = -tf.minimum(tf.constant(0.), -fake_logits - tf.constant(cfg.alpha))
    if cfg.bPsiMax:
        gen_fake_loss = -tf.minimum(tf.constant(0.), fake_logits - tf.constant(cfg.alpha))
    else:
        gen_fake_loss = -fake_logits

else:
    logger.log('using default gan loss')
    dis_real_loss = -tf.log_sigmoid(real_logits)
    dis_fake_loss = -tf.log_sigmoid(-fake_logits)
    gen_fake_loss = -tf.log_sigmoid(fake_logits)

dis_tot_loss = dis_gan_loss = tf.reduce_mean(dis_fake_loss) + tf.reduce_mean(dis_real_loss)
gen_tot_loss = gen_gan_loss = tf.reduce_mean(gen_fake_loss)

dis_lip_loss = interpolates = slopes = tf.constant(0)

if cfg.bLip:
    alpha = tf.random_uniform(shape=[tf.shape(real_datas)[0]] + [1] * (real_datas.get_shape().ndims - 1), minval=0., maxval=1.)
    interpolates = real_datas * alpha + fake_datas * (1. - alpha)

    if cfg.bMaxGP:
        if cfg.fBufferBatch >= 0:
            interpolates = tf.concat([iter_datas, interpolates], 0)
        else:
            interpolates = tf.concat([iter_datas, interpolates[tf.shape(iter_datas)[0]:]], 0)

    gradients = tf.gradients(discriminator(interpolates, reuse=True), interpolates)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=list(range(1, real_datas.get_shape().ndims))))

    slopes_t = tf.reduce_max(slopes) if cfg.bMaxGP else slopes

    if cfg.sGP_type == 'cp':
        dis_lip_loss = cfg.fWeightLip * tf.reduce_mean(tf.square(slopes_t))
    elif cfg.sGP_type == 'gp':
        dis_lip_loss = cfg.fWeightLip * tf.reduce_mean(tf.square(slopes_t - cfg.fLipTarget))
    elif cfg.sGP_type == 'lp':
        dis_lip_loss = cfg.fWeightLip * tf.reduce_mean(tf.square(tf.maximum(0.0, slopes_t - cfg.fLipTarget)))

    dis_tot_loss += dis_lip_loss

tot_vars = tf.trainable_variables()
dis_vars = [var for var in tot_vars if 'discriminator' in var.name]
gen_vars = [var for var in tot_vars if 'generator' in var.name]

gen_ema = EMAHelper(decay=0.99, session=sess, var_list=gen_vars)

global_step = tf.Variable(0, trainable=False, name='global_step')
step_op = tf.assign_add(global_step, 1)

gen_lr = tf.constant(cfg.fLrIniG)
if 'linear' == cfg.oDecayG:
    gen_lr = gen_lr * tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter))
gen_optimizer = tf.train.AdamOptimizer(learning_rate=gen_lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)
gen_gradient_values = gen_optimizer.compute_gradients(gen_tot_loss, var_list=gen_vars)
gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
with tf.control_dependencies(gen_update_ops):
    gen_optimize_ops = gen_optimizer.apply_gradients(gen_gradient_values)

dis_lr = tf.constant(cfg.fLrIniD)
if 'linear' == cfg.oDecayD:
    dis_lr = dis_lr * tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter))
dis_optimizer = tf.train.AdamOptimizer(learning_rate=dis_lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)
dis_gradient_values = dis_optimizer.compute_gradients(dis_tot_loss, var_list=dis_vars)
dis_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
with tf.control_dependencies(dis_update_ops):
    dis_optimize_ops = dis_optimizer.apply_gradients(dis_gradient_values)

real_gradients = tf.gradients(real_logits, real_datas)[0]
fake_gradients = tf.gradients(fake_logits, fake_datas)[0]

varphi_gradients = tf.gradients(dis_real_loss, real_logits)[0]
phi_gradients = tf.gradients(dis_fake_loss, fake_logits)[0]

disvar_lip_gradients = tf.gradients(dis_lip_loss, dis_vars)
disvar_gan_gradients = tf.gradients(dis_gan_loss, dis_vars)
disvar_tot_gradients = tf.gradients(dis_tot_loss, dis_vars)
genvar_tot_gradients = tf.gradients(gen_tot_loss, gen_vars)

disvar_lip_gradients = [tf.constant(0.0) if grad is None else grad for grad in disvar_lip_gradients]
disvar_gan_gradients = [tf.constant(0.0) if grad is None else grad for grad in disvar_gan_gradients]
disvar_tot_gradients = [tf.constant(0.0) if grad is None else grad for grad in disvar_tot_gradients]
genvar_tot_gradients = [tf.constant(0.0) if grad is None else grad for grad in genvar_tot_gradients]

saver = tf.train.Saver(max_to_keep=1000)


############################################################################################################################################

def gen_images(n):
    images = []
    for i in range(n // cfg.iBatchSize + 1):
        images.append(sess.run(fake_datas, feed_dict={z: sample_z(cfg.iBatchSize), is_training: False, **gen_ema.average_dict()}))
    images = np.concatenate(images, 0)
    return images[:n]


def gen_images_with_noise(noise):
    images = []
    n = len(noise)
    ibs = n // cfg.iBatchSize + 1
    noiseA = sample_z(cfg.iBatchSize * ibs)
    noiseA[:n] = noise
    for i in range(ibs):
        images.append(sess.run(fake_datas, feed_dict={z: noiseA[cfg.iBatchSize * i:cfg.iBatchSize * (i + 1)], is_training: False, **gen_ema.average_dict()}))
    images = np.concatenate(images, 0)
    return images[:n]


def param_count(gradient_value):
    total_param_count = 0
    for g, v in gradient_value:
        shape = v.get_shape()
        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count
    return total_param_count


def log_netstate():
    _real_datas = real_gen.__next__()
    logger.linebreak()
    _gen_vars, _genvar_tot_gradients = sess.run([gen_vars, genvar_tot_gradients], feed_dict={real_datas: _real_datas, iter_datas: _iter_datas, z: sample_z(cfg.iBatchSize), is_training: True})
    for i in range(len(_gen_vars)):
        logger.log('weight values: %8.5f %8.5f, tot gradient: %8.5f %8.5f    ' % (
            np.mean(_gen_vars[i]), np.std(_gen_vars[i]), np.mean(_genvar_tot_gradients[i]), np.std(_genvar_tot_gradients[i])) + gen_vars[i].name + ' shape: ' + str(gen_vars[i].shape))
    logger.linebreak()
    _dis_vars, _disvar_lip_gradients, _disvar_gan_gradients, _disvar_tot_gradients = \
        sess.run([dis_vars, disvar_lip_gradients, disvar_gan_gradients, disvar_tot_gradients],
                 feed_dict={real_datas: _real_datas, iter_datas: _iter_datas,  z: sample_z(cfg.iBatchSize), is_training: True})
    for i in range(len(_dis_vars)):
        logger.log('weight values: %8.5f %8.5f, lip gradient: %8.5f %8.5f, gan gradient: %8.5f %8.5f, tot gradient: %8.5f %8.5f    ' % (
            np.mean(_dis_vars[i]), np.std(_dis_vars[i]), np.mean(_disvar_lip_gradients[i]), np.std(_disvar_lip_gradients[i]), np.mean(_disvar_gan_gradients[i]),
            np.std(_disvar_gan_gradients[i]), np.mean(_disvar_tot_gradients[i]), np.std(_disvar_tot_gradients[i])) + dis_vars[i].name + ' shape: ' + str(dis_vars[i].shape))
    logger.linebreak()


ref_icp_preds, ref_icp_activations, ref_mu, ref_sigma, icp_model = None, None, None, None, None


def get_score(n=50000, noise=None, split_std=False):
    logger.linebreak()

    global ref_icp_preds, ref_icp_activations, ref_mu, ref_sigma, icp_model

    if icp_model is None:
        icp_model = PreTrainedDense_Flower() if cfg.sDataSet == "flowers" else PreTrainedInception()

    if ref_icp_activations is None:
        logger.log('Evaluating Reference Statistic: icp_model')
        ref_icp_preds, ref_icp_activations = icp_model.get_preds(dataX.transpose(0, 2, 3, 1))
        logger.log('\nref_icp_score: %.3f\n' % InceptionScore.inception_score_H(ref_icp_preds)[0])
        logger.log('Evaluating ref_mu, ref_sigma')
        ref_mu, ref_sigma = FID.get_stat(ref_icp_activations)

    stime = time.time()
    logger.log('Generating samples')
    samples = gen_images(n) if noise is None else gen_images_with_noise(noise)
    samples = samples.transpose(0, 2, 3, 1)
    logger.log('Time: %.2f' % (time.time() - stime))

    logger.log('Evaluating icp_preds_activcations')
    icp_preds, icp_activcations = icp_model.get_preds(samples)
    logger.log('Time: %.2f' % (time.time() - stime))

    logger.log('Evaluating icp_score')
    icp_score = InceptionScore.inception_score_KL(icp_preds)
    logger.log('Time: %.2f' % (time.time() - stime))

    logger.log('Evaluating mu, sigma')
    mu, sigma = FID.get_stat(icp_activcations)
    logger.log('Time: %.2f' % (time.time() - stime))

    logger.log('Evaluating fid')
    fid = FID.get_FID_with_stat(mu, sigma, ref_mu, ref_sigma)
    logger.log('Time: %.2f' % (time.time() - stime))

    if split_std:
        logger.log('Evaluating icp split std')
        icp_mean, icp_std = InceptionScore.inception_score_split_std(icp_preds)[:2]
        logger.log('Time: %.2f' % (time.time() - stime))
        logger.log('Evaluating fid split std')
        fid_mean, fid_std = FID.get_FID_with_activations_split_std(icp_activcations, ref_mu, ref_sigma)[:2]
        logger.log('Time: %.2f' % (time.time() - stime))
        logger.log('icp_score:%.2f (icp_mean:%.2f ± icp_std:%.2f), fid:%.2f (fid_mean:%.2f ± fid_std:%.2f)' % (icp_score, icp_mean, icp_std, fid, fid_mean, fid_std))
    else:
        icp_mean, icp_std, fid_mean, fid_std = None, None, None, None
        logger.log('icp_score:%.2f, fid:%.2f' % (icp_score, fid))

    return icp_score, icp_mean, icp_std, fid, fid_mean, fid_std


############################################################################################################################################

iter = 0
last_save_time = last_icp_time = last_log_time = last_plot_time = last_imshave_time = time.time()

if cfg.bLoadCheckpoint:
    try:
        if load_model(saver, sess, sCheckpointDir):
            logger.log(" [*] Load SUCCESS")
            iter = sess.run(global_step)
            logger.load()
            logger.tick(iter)
            logger.log('\n\n')
            logger.flush()
            logger.log('\n\n')
        else:
            assert False
    except:
        logger.clear()
        logger.log(" [*] Load FAILED")
        ini_model(sess)
else:
    ini_model(sess)

_real_datas = real_gen.__next__()
_fake_datas = gen_images(cfg.iBatchSize)
_iter_datas = ((_real_datas + _fake_datas) / 2)[:int(cfg.iBatchSize * cfg.fBufferBatch)]

log_netstate()

logger.log("Generator Total Parameter Count: {}".format(locale.format("%d", param_count(gen_gradient_values), grouping=True)))
logger.log("Discriminator Total Parameter Count: {}\n\n".format(locale.format("%d", param_count(dis_gradient_values), grouping=True)))

save_images(dataX[:256].reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [16, 16], sTestCaseDir + 'real_image16x16.png')

start_iter = iter
start_time = time.time()

fixed_noise = sample_z(50000)

while iter < cfg.iMaxIter:

    iter += 1
    train_start_time = time.time()

    for i in range(cfg.iTrainD):
        _, _dis_tot_loss, _dis_gan_loss, _dis_lip_loss, _interpolates, _dphi, _dvarphi, _slopes, _dis_lr, _real_logits, _fake_logits = sess.run(
            [dis_optimize_ops, dis_tot_loss, dis_gan_loss, dis_lip_loss, interpolates, phi_gradients, varphi_gradients, slopes, dis_lr, real_logits, fake_logits],
                feed_dict={real_datas: real_gen.__next__(), iter_datas: _iter_datas, z: sample_z(cfg.iBatchSize), is_training: True})

    for i in range(cfg.iTrainG):
        _, _, _gen_tot_loss, _gen_gan_loss, _gen_lr = sess.run([gen_optimize_ops, gen_ema.apply, gen_tot_loss, gen_gan_loss, gen_lr], feed_dict={z: sample_z(cfg.iBatchSize * cfg.GEN_BS_MULTIPLE), is_training: True})

    sess.run(step_op)
    logger.info('time_train', time.time() - train_start_time)

    log_start_time = time.time()

    logger.tick(iter)
    logger.info('klrG', _gen_lr * 1000)
    logger.info('klrD', _dis_lr * 1000)

    logger.info('logit_real', np.mean(_real_logits))
    logger.info('logit_fake', np.mean(_fake_logits))

    logger.info('loss_dis_gp', _dis_lip_loss)
    logger.info('loss_dis_gan', _dis_gan_loss)
    logger.info('loss_dis_tot', _dis_tot_loss)

    logger.info('loss_gen_gan', _gen_gan_loss)
    logger.info('loss_gen_tot', _gen_tot_loss)

    logger.info('d_phi', np.mean(_dphi))
    logger.info('d_varphi', np.mean(_dvarphi))

    logger.info('slopes_mean', np.mean(_slopes))
    logger.info('slopes_max', np.max(_slopes))

    if cfg.bLip and cfg.bMaxGP:
        _iter_datas = _interpolates[np.argsort(-np.asarray(_slopes))[:len(_iter_datas)]]

    if np.any(np.isnan(_dis_tot_loss)) or np.any(np.isnan(_gen_tot_loss)):
        log_netstate()
        logger.flush()
        exit(0)

    if time.time() - last_icp_time > 60 * 30 and (cfg.sDataSet == 'cifar10' or cfg.sDataSet == 'flowers' or cfg.sDataSet == 'tiny') or iter == cfg.iMaxIter:
        icp_score, icp_mean, icp_std, fid, fid_mean, fid_std = get_score(noise=fixed_noise)
        logger.info('score_icp', icp_score, icp_mean, icp_std)
        logger.info('score_fid', fid, fid_mean, fid_std)
        last_icp_time = time.time()

    if time.time() - last_save_time > 60 * 60 or iter == cfg.iMaxIter:
        stime = time.time()
        log_netstate()
        logger.save()
        save_model(saver, sess, sCheckpointDir, step=iter)
        last_save_time = time.time()
        logger.log('Model saved')
        logger.log('Time: %.2f' % (time.time() - stime))

    if iter == 1000 or iter == 10000 or iter % (cfg.iMaxIter // 10) == 0 or iter == cfg.iMaxIter:
        logger.plot_together(['logit_real', 'logit_fake'], [r'$\mathbb{E}_{x \sim P_r} f(x)$', r'$\mathbb{E}_{x \sim P_g} f(x)$'], ['olive', 'skyblue'], 'logits_' + str(iter) + '.pdf')
        f0 = gen_images(256)
        save_images(f0.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [16, 16], sTestCaseDir + 'gen_image16x16_%d.png' % iter)
        f0 = f0[:128]
        g0 = sess.run(real_gradients, feed_dict={real_datas: f0})
        g0 = g0 / (np.max(np.abs(g0), axis=(1, 2, 3), keepdims=True) + 1e-8)
        grad_image = np.stack([f0, g0], 1)
        save_images(grad_image.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [16, 16], sTestCaseDir + 'grad_image16x16_%d.png' % iter)

    if time.time() - last_plot_time > 60 * 30 or iter == cfg.iMaxIter:
        stime = time.time()
        logger.plot()
        logger.plot_together(['logit_real', 'logit_fake'], [r'$\mathbb{E}_{x \sim P_r} f(x)$', r'$\mathbb{E}_{x \sim P_g} f(x)$'], ['olive', 'skyblue'], 'logits.pdf')
        last_plot_time = time.time()
        logger.log('Plotted')
        logger.log('Time: %.2f' % (time.time() - stime))

    if time.time() - last_imshave_time > 60 * 10 or iter == cfg.iMaxIter:
        stime = time.time()
        f0 = gen_images_with_noise(fixed_noise[:256])
        save_images(f0.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [16, 16], sSampleDir + 'gen_image16x16_%d.png' % iter)
        last_imshave_time = time.time()
        logger.log('Image snapshot saved')
        logger.log('Time: %.2f' % (time.time() - stime))

    logger.info('time_log', time.time() - log_start_time)

    if time.time() - last_log_time > 60 * 1 or iter == 1:
        logger.info('time_used', (time.time() - start_time) / 3600)
        logger.info('time_remain', (time.time() - start_time) / (iter - start_iter) * (cfg.iMaxIter - iter) / 3600)
        logger.flush()
        last_log_time = time.time()

logger.flush()
logger.plot()

logger.log('final scoring')
fixed_noise = sample_z(500000)
get_score(noise=fixed_noise, split_std=True)