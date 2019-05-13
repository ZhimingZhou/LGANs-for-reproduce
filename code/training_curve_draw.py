import pickle
import collections
import numpy as np
from scipy.signal import savgol_filter
from os import path
SOURCE_DIR = path.dirname(path.dirname(path.abspath(__file__))) + '/'

import matplotlib

matplotlib.use('Agg')
title_font = {'size': 16, 'weight': 'bold'}
axis_font = {'size': 16, 'weight': 'bold'}
font = {'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

dataset = 'cifar10'
runs = ['', '_run2', '_run3', '_run4']
weights = ['0.01', '0.1', '1.0', '10.0']
config = 'tanh1_gbn1_maxcp%s_buffer0.00_linear2e-4_200000%s_lip'
config_gp = 'tanh1_gbn1_gp%s_buffer0.00_linear2e-4_200000%s_lip'
names = ['score_fid', 'score_icp', 'slopes_max', 'slopes_mean', 'loss_dis_tot', 'logit_real']
rdir = SOURCE_DIR + 'result/sobolev/'

losses = [
    'x_gp',
    'x',
    'exp',
    'log_sigmoid',
    'sqrt',
    # 'lsgan_0.01',
    # 'lsgan_0.10',
    'lsgan_1.00',
    # 'hinge_0.01',
    # 'hinge_0.10',
    'hinge_1.00'
]

name_dict = {
    'score_fid': 'FID',
    'score_icp': 'Inception Score',
    'x_gp': r'$x$ (with GP)',
    'x': r'$x$',
    'exp': r'$\exp(x)$',
    'log_sigmoid': r'$-\log(\sigma(-x))$',
    'sqrt': r'$x+\sqrt{x^2+1}$',
    'lsgan_0.01': r'$(x+0.01)^2$',
    'lsgan_0.10': r'$(x+0.1)^2$',
    'lsgan_1.00': r'$(x+1)^2$',
    'hinge_0.01': r'$\max(0, x+0.01)$',
    'hinge_0.10': r'$\max(0, x+0.1)$',
    'hinge_1.00': r'$\max(0, x+1)$',
}


def check_param(strlog1, weight):
    while True:
        k1 = strlog1.find('fWeightLip:')
        if k1 != -1:
            if len(strlog1) > k1 + len('fWeightLip:') + len(weight):
                weight_t = strlog1[k1 + len('fWeightLip:'):k1 + len('fWeightLip:') + len(weight)]
                if weight != weight_t:
                    return False
            strlog1 = strlog1[k1 + len('fWeightLip:'):]
        else:
            break
    return True


def get_best_fid(strlog1):
    best_fid_score = 1000
    while True:
        k1 = strlog1.rfind('score_fid:')
        if k1 != -1 and len(strlog1) > k1 + len('score_fid:') + 5:
            k1 += len('score_fid:')
            str = strlog1[k1:k1 + 5]
            fid_score = float(str)
            if fid_score < best_fid_score:
                best_fid_score = fid_score
            strlog1 = strlog1[:k1]
        else:
            break
    return best_fid_score


def get_best_icp(strlog1):
    best_icp_score = 0
    while True:
        k1 = strlog1.rfind('score_icp:')
        if k1 != -1 and len(strlog1) > k1 + len('score_fid:') + 5:
            k1 += len('score_icp:')
            str = strlog1[k1:k1 + 5]
            fid_score = float(str)
            if fid_score > best_icp_score:
                best_icp_score = fid_score
            strlog1 = strlog1[:k1]
        else:
            break
    return best_icp_score


def get_final_score(strlog1):
    while True:
        k1 = strlog1.rfind('final scoring')
        if k1 != -1:
            strlog2 = strlog1[k1:]
            strlog1 = strlog1[:k1]
            k2 = strlog2.find('Evaluating fid split std')
            if k2 != -1:
                strlog3 = strlog2[k2:].splitlines()
                if len(strlog3) > 2 and 'icp_mean' in strlog3[2]:
                    k3 = strlog3[2].rfind('fid_mean:') + len('fid_mean:')
                    final_fid_score = float(strlog3[2][k3:k3 + 5])
                    final_scores = strlog3[2]
                    return final_fid_score, final_scores
        else:
            break
    return 1000, ''


def get_current_iter(strlog1):
    while True:
        k4 = strlog1.rfind('ITER:')
        if k4 != -1:
            k4 += len('ITER:')
            k5 = strlog1[k4:].find(',')
            if k5 != -1:
                str = strlog1[k4:k4 + k5]
                iter = int(str)
                return iter
            strlog1 = strlog1[:k4]
        else:
            break
    return 0


def get_case_result(strlog1):
    cur_iter = get_current_iter(strlog1)
    best_fid = get_best_fid(strlog1)
    best_icp = get_best_icp(strlog1)
    final_fid_score, final_scores = get_final_score(strlog1)
    return cur_iter, best_fid, best_icp, final_fid_score, final_scores


cases = []
score_strs = []
for loss in losses:
    best_case = ''
    best_iter = 0
    best_score = 1000
    best_score_str = ''
    for weight in weights:
        for run in runs:
            config0 = config_gp if 'gp' in loss else config
            case = rdir + dataset + '_' + config0 % (weight, run) + '_' + loss.replace('_gp', '')
            try:
                strlog1 = open(case + '/log.txt', 'r').read()
            except:
                # print('Error: open ' + case)
                continue

            if not check_param(strlog1, weight):
                print('Error: check ' + case)

            cur_iter, best_fid, best_icp, final_fid_score, final_scores = get_case_result(strlog1)
            k1 = len('icp_score:8.57 (icp_mean:8.57 ± icp_std:0.03), ')
            score_str = 'best_icp:%.2f, ' % best_icp + final_scores[:k1] + 'best_fid:%.2f, ' % best_fid + (
                final_scores[k1:] if final_scores != '' else 'cur_iter:%d' % cur_iter) + '\t:\t' + dataset + '_' + loss + '_w' + weight + run
            print(score_str)

            if best_score > final_fid_score:
                best_score = final_fid_score
                best_case = case
                best_score_str = score_str

            if best_score == 1000:
                if cur_iter > best_iter:
                    best_iter = cur_iter
                    best_case = case
                    best_score_str = score_str

    cases.append(best_case)
    score_strs.append(best_score_str)

print('\n\n\nBest Case:')
for score_str in score_strs:
    print(score_str)

save_dir = SOURCE_DIR + 'result/draw/'

logs = []
for case in cases:
    since_beginning = collections.defaultdict(lambda: {})
    try:
        with open(case + '/log.pkl', 'rb') as f:
            since_beginning.update(pickle.load(f))
    except:
        try:
            with open(case + '/log_back.pkl', 'rb') as f:
                since_beginning.update(pickle.load(f))
        except:
            print("ERROR: " + case + '/log.pkl')
    logs.append(since_beginning)

num = 0
import os

while os.path.exists(save_dir + str(num)):
    num += 1
save_dir = save_dir + str(num) + '/'
os.makedirs(save_dir)


def get_dict(name):
    return name_dict.get(name) if name in name_dict.keys() else name


from common.utils import imread, imsave
for loss, case in zip(losses, cases):
    image = imread(case + '/gen_image16x16_200000.png')
    image = image[image.shape[0]//2:, image.shape[1]//2:]
    imsave(image, 2, save_dir + dataset + '_' + loss.replace('.', '_') + '.png')

os.system('cp ' + SOURCE_DIR + 'code/training_curve_draw.py ' + save_dir)

for name in names:

    plt.clf()


    def smooth(y, winsize):
        if winsize % 2 == 0:
            winsize -= 1
        if winsize < 3:
            return y
        return savgol_filter(y, winsize, 1, mode='mirror')


    for log, loss in zip(logs, losses):
        try:
            x_vals = np.sort(list(log[name].keys()))
            y_vals = [log[name][x] for x in x_vals]
            x_vals = x_vals
            if 'score' in name:
                plt.plot(x_vals, smooth(y_vals, 0), label=get_dict(loss))
            else:
                plt.plot(x_vals[10000:], smooth(y_vals, 1000)[10000:], label=get_dict(loss))
        except:
            print('Error: ' + name + loss)

    plt.xlabel('Iterations')
    plt.ylabel(get_dict(name))
    plt.legend(loc=0)
    plt.tight_layout()

    plt.savefig(save_dir + dataset + '_training_curve_' + name + '.pdf')
    print(save_dir + dataset + '_training_curve_' + name + '.pdf')
    os.system('pdfcrop ' + save_dir + dataset + '_training_curve_' + name + '.pdf')