import sys, locale, time, os
from os import path

locale.setlocale(locale.LC_ALL, '')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_DIR = path.dirname(path.dirname(path.abspath(__file__))) + '/'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from common.data_loader import *
from common.logger import Logger
from common.optimizer import *

import sklearn.datasets
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable

cfg = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("sDataSet", "toy", "cifar10, mnist, toy")
tf.app.flags.DEFINE_boolean("bLoadCheckpoint", False, "bLoadCheckpoint")
tf.app.flags.DEFINE_string("sResultTag", "test_case3", "your tag for each test case")

############################################################################################################################################

tf.app.flags.DEFINE_integer("n", 2, "")

tf.app.flags.DEFINE_string("sGAN_Type", 'log_sigmoid', "")

tf.app.flags.DEFINE_boolean("bLip", True, "")
tf.app.flags.DEFINE_boolean("bMaxGP", True, "")
tf.app.flags.DEFINE_boolean("bGP", False, "")
tf.app.flags.DEFINE_boolean("bLP", False, "")
tf.app.flags.DEFINE_boolean("bCP", False, "")

tf.app.flags.DEFINE_float("fWeightLip", 0.1, "")
tf.app.flags.DEFINE_float("fWeightZero", 0.0, "")

tf.app.flags.DEFINE_boolean("bUseSN", False, "")
tf.app.flags.DEFINE_boolean("bUseSNK", False, "")

############################################################################################################################################

tf.app.flags.DEFINE_integer("iMaxIter", 1000000, "")
tf.app.flags.DEFINE_integer("iBatchSize", 256, "")

tf.app.flags.DEFINE_float("fLrIniD", 0.0001, "")
tf.app.flags.DEFINE_string("oDecay", 'none', "linear, exp, none, ramp")

tf.app.flags.DEFINE_float("fBeta1", 0.0, "")
tf.app.flags.DEFINE_float("fBeta2", 0.9, "")
tf.app.flags.DEFINE_float("fEpsilon", 1e-8, "")
tf.app.flags.DEFINE_string("oOptD", 'adam', "adams, adam, sgd, mom")

tf.app.flags.DEFINE_integer("iDimsC", 3, "")
tf.app.flags.DEFINE_integer("iMinSizeD", 4, "")

tf.app.flags.DEFINE_string("discriminator", 'discriminator_mlp', "")
tf.app.flags.DEFINE_integer("iBaseNumFilterD", 1024, "")
tf.app.flags.DEFINE_integer("iBlockPerLayerD", 4, "")

tf.app.flags.DEFINE_string("oActD", 'relu', "elulike, relu")
tf.app.flags.DEFINE_string("oBnD", 'none', "bn, ln, none")

tf.app.flags.DEFINE_boolean("bUseWN", False, "")
tf.app.flags.DEFINE_float("fScaleActD", 1.00, "") #np.sqrt(2/(1+alpha**2))
tf.app.flags.DEFINE_float("fDefaultGain", 1.00, "") #np.sqrt(2/(1+alpha**2))
tf.app.flags.DEFINE_float("fInitWeightStddev", 1.00, "")
tf.app.flags.DEFINE_string("oInitType", 'normal', "truncated_normal, normal, uniform, orthogonal")

tf.app.flags.DEFINE_integer("GPU", -1, "")
tf.app.flags.DEFINE_string("sResultDir", SOURCE_DIR + "result/Sythetic/", "where to save the checkpoint and sample")

cfg(sys.argv)

GPU_ID = allocate_gpu(cfg.GPU)

############################################################################################################################################

np.random.seed(1000)
tf.set_random_seed(1000)

from common.ops import *
set_enable_bias(True)
set_data_format('NCHW')
set_enable_wn(cfg.bUseWN)
set_enable_sn(cfg.bUseSN)
set_enable_snk(cfg.bUseSNK)
set_default_gain(cfg.fDefaultGain)
set_init_type(cfg.oInitType)
set_init_weight_stddev(cfg.fInitWeightStddev)

def transform(data, mean=(0, 0), size=1.0, rot=0.0, hflip=False, vflip=False):
    data *= size
    rotMatrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    data = np.matmul(data, rotMatrix)
    if hflip: data[:, 0] *= -1
    if vflip: data[:, 1] *= -1
    data += mean
    return data

def sqaure_generator(num_sample, noise, transform):
    while True:
        x = np.random.rand(num_sample) - 0.5
        y = np.random.rand(num_sample) - 0.5
        data = np.asarray([x, y]).transpose().astype('float32') * 2.0
        data += noise * np.random.randn(num_sample, 2)
        data = transform(data)
        yield data

def discrete_generator(num_sample, noise, transform, num_meta=2):
    meta_data = np.asarray([[i, j] for i in range(-num_meta, num_meta+1,1) for j in range(-num_meta, num_meta+1, 1)]) / float(num_meta)
    while True:
        idx = np.random.random_integers(0, len(meta_data)-1, num_sample)
        data = meta_data[idx].astype('float32')
        data += noise * np.random.randn(num_sample, 2)
        data = transform(data)
        yield data

def boundary_generator(num_sample, noise, transform, num_meta=2):
    meta_data = []
    for i in range(-num_meta, num_meta+1, 1):
        meta_data.append([i, -num_meta])
        meta_data.append([i, +num_meta])
    for i in range(-num_meta+1, num_meta, 1):
        meta_data.append([-num_meta, i])
        meta_data.append([+num_meta, i])
    meta_data = np.asarray(meta_data) / float(num_meta)

    while True:
        idx = np.random.random_integers(0, len(meta_data)-1, num_sample)
        data = meta_data[idx].astype('float32')
        data += noise * np.random.randn(num_sample, 2)
        data = transform(data)
        yield data

def circle_generator(num_sample, noise, transform):
    while True:
        linspace = np.random.rand(num_sample)
        x = np.cos(linspace * 2 * np.pi)
        y = np.sin(linspace * 2 * np.pi)
        data = np.asarray([x, y]).transpose().astype('float32')
        data += noise * np.random.randn(num_sample, 2)
        data = transform(data)
        yield data

def scurve_generator(num_sample, noise, transform):
    while True:
        data = sklearn.datasets.make_s_curve(
            n_samples=num_sample,
            noise=noise
        )[0]
        data = data.astype('float32')[:, [0, 2]]
        data /= 2.0
        data = transform(data)
        yield data

def swiss_generator(num_sample, noise, transform):
    while True:
        data = sklearn.datasets.make_swiss_roll(
            n_samples=num_sample,
            noise=noise
        )[0]
        data = data.astype('float32')[:, [0, 2]]
        data /= 14.13717
        data = transform(data)
        yield data

def gaussian_generator(num_sample, noise, transform):
    while True:
        data = np.random.multivariate_normal([0.0, 0.0], noise * np.eye(2), num_sample)
        data = transform(data)
        yield data

def mix_generator(num_sample, generators, weights):
    while True:
        data = np.concatenate([generators[i].__next__() for i in range(len(generators))], 0)
        data_index = np.random.choice(len(weights), num_sample, replace=True, p=weights)
        data2 = np.concatenate([data[num_sample * i:num_sample * i + np.sum(data_index == i)] for i in range(len(weights))], 0)
        np.random.shuffle(data2)
        yield data2

cfg.iDimsC = 2

# fake_gen = circle_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5))
# real_gen = circle_generator(cfg.iBatchSize, 0.0, partial(transform, size=1.5))

# fake_gen = scurve_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(-1.0, 0), rot=np.pi / 2, hflip=True))
# real_gen = scurve_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, 0), rot=np.pi / 2))

fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.0, mean=(-1.0, -0.0)))
real_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.0, mean=(+1.0, +0.0)))

# fake_gen = scurve_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.8, mean=(0, 0), hflip=False, rot=np.pi / 4))
# real_gen = scurve_generator(cfg.iBatchSize, 0.0, partial(transform, size=1.5, mean=(0, 0)))

# fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.0, 0.0)))
# real_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(+1.0, 0.0)))

# fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.0, 0.0)))
# real_gen = mix_generator(cfg.iBatchSize, [gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(+1.0, 0.0))),
#                           gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.5, 0.0)))], [0.9, 0.1])

# fake_gen = sqaure_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.0, 0.0)))
# real_gen = mix_generator(cfg.iBatchSize, [gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(+1.0, 0.0))),
#                           gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.0, 0.0)))], [0.9, 0.1])

# fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=1.0, mean=(+0.0, 0)))
# real_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=1.0, mean=(+0.0, 0)))

if 'case1.1' in cfg.sResultTag:

    fake_gen = sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(-1.0, -0.0)))
    real_gen = sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, +0.0)))

elif 'case1' in cfg.sResultTag:

    fake_gen = mix_generator(cfg.iBatchSize, [boundary_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.50, mean=(-1.0, -0.0))),
                                              sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.25, mean=(-1.0, -0.0)))], [0.20, 0.80])
    real_gen = sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, +0.0)))

elif 'case2' in cfg.sResultTag:

    fake_gen = mix_generator(cfg.iBatchSize, [sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(-1.0, -0.0))),
                                              sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, +0.0)))], [0.80, 0.20])
    real_gen = mix_generator(cfg.iBatchSize, [sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(-1.0, -0.0))),
                                              sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, +0.0)))], [0.20, 0.80])

elif 'case3.3' in cfg.sResultTag:

    fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(-1.0, -0.0)))
    real_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(+1.0, +0.0)))

elif 'case3' in cfg.sResultTag:

    fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(-0.75, 0.0)))
    real_gen = mix_generator(cfg.iBatchSize, [gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(+1.0, 0.0))),
                                              gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(-1.25, 0.0)))], [0.5, 0.5])

elif 'case4' in cfg.sResultTag:

    n = cfg.n
    np.random.seed(123456789)

    if n == 0:
        n = 2
        r0 = [[+0.8, +0.5], [-0.8, -0.5]]
        f0 = [[-0.8, +0.5], [+0.8, -0.5]]
    else:
        r0 = (np.random.rand(n, 2) - 0.5) * 2
        f0 = (np.random.rand(n, 2) - 0.5) * 2

    def get_mix_gen(centers):
        mix_gen = []
        for i in range(n):
            mix_gen.append(gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.0, mean=(centers[i][0], centers[i][1]))))
        return mix_gen

    fake_gen = mix_generator(cfg.iBatchSize, get_mix_gen(f0), [1 / n] * n)
    real_gen = mix_generator(cfg.iBatchSize, get_mix_gen(r0), [1 / n] * n)

elif 'case5' in cfg.sResultTag:

    fake_gen = swiss_generator(cfg.iBatchSize, 1.0, partial(transform, size=1.0, mean=(0, 0)))
    real_gen = swiss_generator(cfg.iBatchSize, 0.0, partial(transform, size=1.0, mean=(0, 0)))

elif 'case6' in cfg.sResultTag:

    n = cfg.n
    std = 0

    if n < 0:
        n= -n
        std = 1e-2

    def get_mix_gen(centers, std):
        mix_gen = []
        for i in range(len(centers)):
            mix_gen.append(gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=std, mean=(centers[i][0], centers[i][1]))))
        return mix_gen

    if n == 2:

        f0 = [[-0.5, +0.5], [+0.5, -0.5]]
        r0 = [[+0.5, +0.5], [-0.5, -0.5]]

        fake_gen = mix_generator(cfg.iBatchSize, get_mix_gen(f0, std), [1 / n] * n)
        real_gen = mix_generator(cfg.iBatchSize, get_mix_gen(r0, 0), [1 / n] * n)

    elif n == 4:

        np.random.seed(123456789)
        f0 = (np.random.rand(2, 2) - 0.5) * 2
        r0 = (np.random.rand(4, 2) - 0.5) * 2

        fake_gen = mix_generator(cfg.iBatchSize, get_mix_gen(f0, std), [1 / 2] * 2)
        real_gen = mix_generator(cfg.iBatchSize, get_mix_gen(r0, 0), [1 / 4] * 4)


def discriminator_mlp(input, num_logits, name=None):

    layers = []
    iBaseNumFilterD = cfg.iBaseNumFilterD

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

            h0 = input
            layers.append(h0)

            for i in range(cfg.iBlockPerLayerD):

                with tf.variable_scope('layer' + str(i)):

                    h0 = linear(h0, iBaseNumFilterD)
                    layers.append(h0)

                    h0 = normalize(h0, cfg.oBnD)
                    layers.append(h0)

                    h0 = activate(h0, cfg.oActD, cfg.fScaleActD)
                    layers.append(h0)

            h0 = linear(h0, num_logits, name='final_linear')
            layers.append(h0)

        return h0, layers


def discriminator_mlp_dense(input, num_logits, name=None):

    layers = []
    iBaseNumFilterD = cfg.iBaseNumFilterD

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

            h0 = input
            layers.append(h0)

            for i in range(cfg.iBlockPerLayerD):

                with tf.variable_scope('layer' + str(i)):

                    h1 = h0

                    with tf.variable_scope('composite'):

                        h1 = linear(h1, iBaseNumFilterD)
                        layers.append(h1)

                        h1 = normalize(h1, cfg.oBnD)
                        layers.append(h1)

                        h1 = activate(h1, cfg.oActD, cfg.fScaleActD)
                        layers.append(h1)

                    h0 = tf.concat(values=[h0, h1], axis=1)

            h0 = linear(h0, num_logits, name='final_linear')
            layers.append(h0)

        return h0, layers

############################################################################################################################################

sTestName = cfg.sDataSet + ('_' + cfg.sResultTag if len(cfg.sResultTag) else "")

sTestCaseDir = cfg.sResultDir + sTestName + '/'
sSampleDir = sTestCaseDir + 'samples/'
sCheckpointDir = sTestCaseDir + 'checkpoint/'

makedirs(sCheckpointDir)
makedirs(sSampleDir)
makedirs(sTestCaseDir + 'source/code/')
makedirs(sTestCaseDir + 'source/common/')

logger = Logger()
logger.set_dir(sTestCaseDir)
logger.set_casename(sTestName)
logger.log(sTestCaseDir)

commandline = ''
for arg in ['CUDA_VISIBLE_DEVICES="0" python3'] + sys.argv:
    commandline += arg + ' '
logger.log(commandline)

logger.log(str_flags(cfg.__flags))
logger.log('Using GPU%d\n' % GPU_ID)

copydir(SOURCE_DIR + "code/", sTestCaseDir + 'source/code/')
copydir(SOURCE_DIR + "common/", sTestCaseDir + 'source/common/')

############################################################################################################################################

tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

discriminator = globals()[cfg.discriminator]

real_datas = tf.placeholder(tf.float32, [None, cfg.iDimsC], name='real_datas')
fake_datas = tf.placeholder(tf.float32, [None, cfg.iDimsC], name='fake_datas')
iter_datas = tf.placeholder(tf.float32, [None, cfg.iDimsC], name='iter_datas')

real_logits, dis_real_layers = discriminator(real_datas, 1, 'real')
fake_logits, dis_fake_layers = discriminator(fake_datas, 1, 'fake')

real_logits = tf.reshape(real_logits, [-1])
fake_logits = tf.reshape(fake_logits, [-1])

if cfg.sGAN_Type == 'lsgan':
    dis_real_loss = tf.square(real_logits - 1.0)
    dis_fake_loss = tf.square(fake_logits + 1.0)
    logger.log('using lsgan loss')

elif cfg.sGAN_Type == 'x':
    dis_real_loss = -real_logits
    dis_fake_loss = fake_logits
    logger.log('using wgan loss')

elif cfg.sGAN_Type == 'sqrt':
    dis_real_loss = tf.sqrt(tf.square(real_logits) + 1) - real_logits
    dis_fake_loss = tf.sqrt(tf.square(fake_logits) + 1) + fake_logits
    logger.log('using sqrt loss')

elif cfg.sGAN_Type == 'log_sigmoid':
    dis_real_loss = -tf.log_sigmoid(real_logits)
    dis_fake_loss = -tf.log_sigmoid(-fake_logits)
    logger.log('using log_sigmoid loss')

elif cfg.sGAN_Type == 'exp':
    dis_real_loss = tf.exp(-real_logits)
    dis_fake_loss = tf.exp(fake_logits)
    logger.log('using exp loss')

else:

    dis_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits))
    dis_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits))
    logger.log('using default gan loss (log_sigmoid)')

dis_gan_loss = tf.reduce_mean(dis_fake_loss) + tf.reduce_mean(dis_real_loss)
dis_zero_loss = cfg.fWeightZero * tf.square(tf.reduce_mean(fake_logits) + tf.reduce_mean(real_logits))
dis_tot_loss = dis_gan_loss + dis_zero_loss

dis_lip_loss = interpolates = slopes = dotk = gradients = tf.constant(0)

if cfg.bLip:

    alpha = tf.random_uniform(shape=[tf.shape(fake_datas)[0], 1], minval=0., maxval=1.)

    differences = fake_datas - real_datas
    interpolates = real_datas + alpha * differences

    if cfg.bMaxGP:
        interpolates = tf.concat([iter_datas, interpolates[tf.shape(iter_datas)[0]:]], 0)

    interpolates_logits, interpolates_layers = discriminator(interpolates, 1, 'inter')
    gradients = tf.gradients(interpolates_logits, interpolates)[0]

    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))  #tf.norm()

    if cfg.bUseSN:
        dotk = tf.reduce_prod(SPECTRAL_NORM_K_LIST)
        dis_lip_loss = cfg.fWeightLip * (dotk ** 2)
    else:
        if cfg.bMaxGP:
            dis_lip_loss = cfg.fWeightLip * tf.reduce_max(tf.square(slopes))
        elif cfg.bGP:
            dis_lip_loss = cfg.fWeightLip * tf.reduce_mean(tf.square(slopes - 1.0))
        elif cfg.bLP:
            dis_lip_loss = cfg.fWeightLip * tf.reduce_mean(tf.square(tf.maximum(0.0, slopes - 1.0)))
        elif cfg.bCP:
            dis_lip_loss = cfg.fWeightLip * tf.reduce_mean(tf.square(slopes))

    dis_tot_loss += dis_lip_loss

############################################################################################################################################

tot_vars = tf.trainable_variables()
dis_vars = [var for var in tot_vars if 'discriminator' in var.name]

global_step = tf.Variable(0, trainable=False, name='global_step')

dis_lr = tf.constant(cfg.fLrIniD)
if cfg.oDecay == 'linear':
    dis_lr = tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter)) * tf.constant(cfg.fLrIniD)
elif cfg.oDecay == 'exp':
    dis_lr = tf.train.exponential_decay(cfg.fLrIniD, global_step, cfg.iMaxIter // 10, 0.5, True)
elif cfg.oDecay == 'ramp':
    dis_lr = rampup(global_step, cfg.iMaxIter / 100.0 / cfg.iBatchSize) * tf.constant(cfg.fLrIniD)

dis_optimizer = None
if cfg.oOptD == 'sgd':
    dis_optimizer = Grad(learning_rate=dis_lr)
elif cfg.oOptD == 'mom':
    dis_optimizer = Mom(learning_rate=dis_lr, beta1=0.9)
elif cfg.oOptD == 'adam':
    dis_optimizer = Adam(learning_rate=dis_lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)

dis_gradient_values = dis_optimizer.compute_gradients(dis_tot_loss, var_list=dis_vars)
dis_optimize_ops = dis_optimizer.apply_gradients(dis_gradient_values, global_step=global_step)

############################################################################################################################################

real_gradients = tf.gradients(real_logits, real_datas)[0]

varphi_gradients = tf.gradients(dis_real_loss, real_logits)[0]
phi_gradients = tf.gradients(dis_fake_loss, fake_logits)[0]

disvar_lip_gradients = tf.gradients(dis_lip_loss, dis_vars)
disvar_gan_gradients = tf.gradients(dis_gan_loss, dis_vars)
disvar_tot_gradients = tf.gradients(dis_tot_loss, dis_vars)

disvar_lip_gradients = [tf.constant(0.0) if grad is None else grad for grad in disvar_lip_gradients]

saver = tf.train.Saver(max_to_keep=1000)
writer = tf.summary.FileWriter(sTestCaseDir, sess.graph)

############################################################################################################################################

def plot(names, x_map_size, y_map_size, x_value_range, y_value_range, mode='fake', contour=False):

    def get_current_logits_map():

        logits_map = np.zeros([y_map_size, x_map_size])
        gradients_map = np.zeros([y_map_size, x_map_size, 2])

        for i in range(y_map_size):  # the i-th row and j-th column
            locations = []
            for j in range(x_map_size):
                y = y_value_range[1] - (y_value_range[1] - y_value_range[0]) / y_map_size * (i + 0.5)
                x = x_value_range[0] + (x_value_range[1] - x_value_range[0]) / x_map_size * (j + 0.5)
                locations.append([x, y])
            locations = np.asarray(locations).reshape([x_map_size, 2])
            logits_map[i], gradients_map[i] = sess.run([real_logits, real_gradients], feed_dict={real_datas: locations})

        return logits_map, gradients_map

    def boundary_data(num_meta, mean, size):

        meta_data = []
        for i in range(-num_meta, num_meta + 1, 1):
            meta_data.append([i, -num_meta])
            meta_data.append([i, +num_meta])
        for i in range(-num_meta + 1, num_meta, 1):
            meta_data.append([-num_meta, i])
            meta_data.append([+num_meta, i])
        meta_data = np.asarray(meta_data) / float(num_meta)

        meta_data *= size
        meta_data += mean

        return meta_data

    def get_data_and_gradient(gen, num, pre_sample=None):
        data = []
        logit = []
        gradient = []
        if pre_sample is not None:
            _logit, _gradient = sess.run([real_logits, real_gradients], feed_dict={real_datas: pre_sample})
            data.append(pre_sample)
            logit.append(_logit)
            gradient.append(_gradient)
        for i in range(num // cfg.iBatchSize + 1):
            _data = gen.__next__()
            _logit, _gradient = sess.run([real_logits, real_gradients], feed_dict={real_datas: _data})
            data.append(_data)
            logit.append(_logit)
            gradient.append(_gradient)
        data = np.concatenate(data, axis=0)
        logit = np.concatenate(logit, axis=0)
        gradient = np.concatenate(gradient, axis=0)
        return data[:num], logit[:num], gradient[:num]

    pre_sample = None
    if 'Case_1' in cfg.sResultTag:
        pre_sample = boundary_data(5, [-1.0, 0.0], 0.25)
    elif 'Case_2' in cfg.sResultTag:
        pre_sample = np.concatenate([boundary_data(5, [-1.0, 0.0], 0.5), boundary_data(5, [1.0, 0.0], 0.5)], 0)

    _real_datas, _real_logits, _real_gradients = get_data_and_gradient(real_gen, 1024)
    _fake_datas, _fake_logits, _fake_gradients = get_data_and_gradient(fake_gen, 1024, pre_sample)

    if cfg.sGAN_Type == 'lsgan' and not cfg.bLip:
        cmin = -2.0
        cmax = +2.0
    else:
        cmin = np.min(np.concatenate([_fake_logits, _real_logits], 0))
        cmax = np.max(np.concatenate([_fake_logits, _real_logits], 0))

    rcParams['font.family'] = 'monospace'
    fig, ax = plt.subplots(dpi=300)

    logits_map, gradients_map = get_current_logits_map()
    # pickle.dump([_real_datas, _real_logits, _real_gradients, _fake_datas, _fake_logits, _fake_gradients, logits_map, gradients_map], open(sSampleDir + names[0] + '.pck', 'wb'))

    if contour:
        im = ax.contourf(logits_map, 50, extent=[x_value_range[0], x_value_range[1], y_value_range[0], y_value_range[1]])
    else:
        im = ax.imshow(logits_map, extent=[x_value_range[0], x_value_range[1], y_value_range[0], y_value_range[1]], vmin=cmin, vmax=cmax, cmap='viridis')

    plt.scatter(_real_datas[:, 0], _real_datas[:, 1], marker='+', s=1.5, label='real samples', color='navy') #purple')#indigo')#navy')balck
    plt.scatter(_fake_datas[:, 0], _fake_datas[:, 1], marker='*', s=1.5, label='fake samples', color='ivory') #ivory') #'silver')white

    plt.xlim(x_value_range[0], x_value_range[1])
    plt.ylim(y_value_range[0], y_value_range[1])

    if mode != 'none':

        if mode == 'fake':
            xx, yy, uu, vv = _fake_datas[:, 0], _fake_datas[:, 1], _fake_gradients[:, 0], _fake_gradients[:, 1]
        else:
            num_arrow = 20
            skip = (slice(y_map_size // num_arrow // 2, None, y_map_size // num_arrow), slice(x_map_size // num_arrow // 2, None, x_map_size // num_arrow))
            y, x = np.mgrid[y_value_range[1]:y_value_range[0]:y_map_size * 1j, x_value_range[0]:x_value_range[1]:x_map_size * 1j]
            xx, yy, uu, vv = x[skip], y[skip], gradients_map[skip][:, :, 0], gradients_map[skip][:, :, 1]

        ref_scale = np.max(np.linalg.norm(_fake_gradients, axis=1)) / 2 if np.max(np.linalg.norm(_fake_gradients, axis=1)) / np.mean(np.linalg.norm(_fake_gradients, axis=1)) < 2 else np.mean(np.linalg.norm(_fake_gradients, axis=1))

        if 'Case_5' in cfg.sResultTag:
            ref_scale = np.mean(np.linalg.norm(_fake_gradients, axis=1)) / 5

        len = np.hypot(uu, vv)
        uu = uu / (len+1e-8) * np.minimum(len, 2 * ref_scale)
        vv = vv / (len+1e-8) * np.minimum(len, 2 * ref_scale)

        q = ax.quiver(xx, yy, uu, vv, color='red', angles='xy', width=0.001, minlength=0.8, minshaft=3, scale=ref_scale * 30) # violet, fuchsia
        plt.quiverkey(q, 0.65, 0.920, ref_scale, r'$\Vert\nabla_{\!x}f(x)\Vert$=%.2E' % (float(ref_scale)), labelpos='E', coordinates='figure')

    ax.set(aspect=1, title='')
    plt.legend(loc='upper left', prop={'size': 10})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, extend='both', format='%.1f')

    plt.tight_layout()

    for name in names:
        plt.savefig(sSampleDir + name)

    plt.close()

def param_count(gradient_value):
    total_param_count = 0
    for g, v in gradient_value:
        shape = v.get_shape()
        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count
    return total_param_count


def log_netstate(log_layers=False, log_weight=True):

    if log_layers:
        logger.linebreak()
        dis_layers = dis_real_layers + dis_fake_layers
        _dis_layers = sess.run(dis_layers, feed_dict={real_datas: _real_datas, iter_datas: _iter_datas, fake_datas: _fake_datas})
        for i in range(len(_dis_layers)):
            logger.log('layer values: %8.5f %8.5f    ' % (np.mean(_dis_layers[i]), np.std(_dis_layers[i])) + dis_layers[i].name + ' shape: ' + str(_dis_layers[i].shape))

    if log_weight:
        logger.linebreak()
        _dis_vars, _disvar_lip_gradients, _disvar_gan_gradients, _disvar_tot_gradients = sess.run([dis_vars, disvar_lip_gradients, disvar_gan_gradients, disvar_tot_gradients], feed_dict={real_datas: _real_datas, iter_datas: _iter_datas, fake_datas: _fake_datas})
        for i in range(len(_dis_vars)):
            logger.log('weight values: %8.5f %8.5f, lip gradient: %8.5f %8.5f, gan gradient: %8.5f %8.5f, tot gradient: %8.5f %8.5f    ' % (np.mean(_dis_vars[i]), np.std(_dis_vars[i]), np.mean(_disvar_lip_gradients[i]), np.std(_disvar_lip_gradients[i]), np.mean(_disvar_gan_gradients[i]), np.std(_disvar_gan_gradients[i]), np.mean(_disvar_tot_gradients[i]), np.std(_disvar_tot_gradients[i])) + dis_vars[i].name + ' shape: ' + str(dis_vars[i].shape))

    logger.linebreak()

############################################################################################################################################

iter = 0
last_save_time = last_icp_time = last_log_time = last_plot_time = time.time()

if cfg.bLoadCheckpoint:
    try:
        if load_model(saver, sess, sCheckpointDir):
            logger.log(" [*] Load SUCCESS")
            iter = sess.run(global_step)
            logger.tick(iter)
            logger.load()
            logger.linebreak()
            logger.flush()
            logger.linebreak()
        else:
            assert False
    except:
        logger.clear()
        logger.log(" [*] Load FAILED")
        ini_model(sess)
else:
    ini_model(sess)

alphat = np.random.uniform(size=[cfg.iBatchSize, 1])
_real_datas = real_gen.__next__()
_fake_datas = fake_gen.__next__()
_iter_datas = (_real_datas * alphat + _fake_datas * (1-alphat))[:cfg.iBatchSize // 8]

if cfg.bUseSN:
    sess.run(SPECTRAL_NORM_UV_UPDATE_OPS_LIST)

if cfg.bUseSNK:
    sess.run(SPECTRAL_NORM_K_INIT_OPS_LIST)

log_netstate(log_layers=True)
logger.log("Discriminator Total Parameter Count: {}\n".format(locale.format("%d", param_count(dis_gradient_values), grouping=True)))

while iter < cfg.iMaxIter:

    iter += 1
    train_start_time = time.time()

    _real_datas = real_gen.__next__()
    _fake_datas = fake_gen.__next__()

    if cfg.bUseSN:
        sess.run(SPECTRAL_NORM_UV_UPDATE_OPS_LIST)

    _, _dis_tot_loss, _dis_gan_loss, _dis_lip_loss, _interpolates, _dphi, _dvarphi, _slopes, _dotk, _dis_zero_loss, _dis_lr, _real_logits, _fake_logits = sess.run(
        [dis_optimize_ops, dis_tot_loss, dis_gan_loss, dis_lip_loss, interpolates, phi_gradients, varphi_gradients, slopes, dotk, dis_zero_loss, dis_lr, real_logits, fake_logits],
        feed_dict={real_datas: _real_datas, fake_datas: _fake_datas, iter_datas: _iter_datas})

    log_start_time = time.time()

    logger.tick(iter)
    logger.info('klr', _dis_lr * 1000)
    logger.info('time_train', time.time() - train_start_time)

    logger.info('logit_real', np.mean(_real_logits))
    logger.info('logit_fake', np.mean(_fake_logits))

    logger.info('loss_gp', _dis_lip_loss)
    logger.info('loss_gan', _dis_gan_loss)
    logger.info('loss_tot', _dis_tot_loss)
    logger.info('loss_zero', _dis_zero_loss)

    logger.info('d_phi', np.mean(_dphi))
    logger.info('d_varphi', np.mean(_dvarphi))

    logger.info('dotk', _dotk)
    logger.info('slopes_max', np.max(_slopes))
    logger.info('slopes_mean', np.mean(_slopes))

    if cfg.bLip and cfg.bMaxGP:
        _iter_datas = _interpolates[np.argsort(-np.asarray(_slopes))[:len(_iter_datas)]]

    if np.any(np.isnan(_real_logits)) or np.any(np.isnan(_fake_logits)):
        log_netstate()
        logger.flush()
        open(sTestCaseDir + "_NAN_", 'w')
        exit(0)

    if time.time() - last_save_time > 60 * 10:
        log_netstate(log_layers=True)
        logger.save()
        save_model(saver, sess, sCheckpointDir, step=iter)
        last_save_time = time.time()

    if time.time() - last_plot_time > 60 * 10:
        log_netstate()
        logger.plot()
        last_plot_time = time.time()

    if time.time() - last_log_time > 60 * 1:

        if 'case5' in cfg.sResultTag:
            plot(['map_fake_%d.pdf' % iter], 600, 600, [-1.5, 1.5], [-1.5, 1.5], 'none')
        else:
            plot(['map_fake_%d.pdf' % iter], 900, 600, [-2.0, 2.0], [-1.5, 1.5], 'fake')

        logger.info('time_log', time.time() - log_start_time)
        logger.flush()
        last_log_time = time.time()