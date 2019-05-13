import numpy as np
import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shutil import *

import time
import collections
import pickle
from datetime import datetime
from scipy.signal import savgol_filter

class Logger:

    def __init__(self):

        self.since_beginning = collections.defaultdict(lambda: {})
        self.since_last_flush = collections.defaultdict(lambda: {})
        self.mean_std = collections.defaultdict(lambda: {})

        self.iter = 0
        self.logfile = None
        self.logtime = True
        self.save_folder = ""
        self.casename = ""

    def clear(self):
        self.since_beginning = collections.defaultdict(lambda: {})
        self.since_last_flush = collections.defaultdict(lambda: {})

    def tick(self, iter):
        self.iter = iter

    def enable_logscale_plot(self):
        self.logscale = True

    def set_dir(self, folder):
        self.save_folder = folder
        self.logfile = open(self.save_folder + "log.txt", 'a')

    def set_casename(self, name):
        self.casename = name

    def info(self, name, value, mean=None, std=None):
        self.since_last_flush[name][self.iter] = value
        if mean is not None:
            self.mean_std[name]['mean'] = mean
        if std is not None:
                self.mean_std[name]['std'] = std

    def linebreak(self):
        self.log('')

    def log(self, str):
        if self.logfile is not None:
            self.logfile.write(str + '\n')
        print(str)

    def plot_together(self, names, legends, colors, save_name):

        def smooth(y, winsize):
            if winsize % 2 == 0:
                winsize -= 1
            if winsize <= 5:
                return y
            return savgol_filter(y, winsize, 1, mode='mirror')

        plt.clf()
        plt.rcParams['agg.path.chunksize'] = 20000

        for i, name in enumerate(names):
            x_vals = np.sort(list(self.since_beginning[name].keys()))
            y_vals = [self.since_beginning[name][x] for x in x_vals]
            y_vals_s = smooth(y_vals, len(y_vals) // 250)
            plt.plot(x_vals, y_vals_s, label=legends[i], color=colors[i])

        plt.xlabel('iteration')
        plt.legend(loc=2)
        plt.tight_layout()

        plt.savefig(self.save_folder + save_name)

    def plot(self, bSmooth=True, bLogScale=False, his=1.0):

        for name in np.sort(list(self.since_beginning.keys())):

            x_vals = np.sort(list(self.since_beginning[name].keys()))
            y_vals = [self.since_beginning[name][x] for x in x_vals]

            if np.mean(y_vals) == 0.0 and np.std(y_vals) == 0.0:
                continue

            def smooth(y, winsize):
                if winsize % 2 == 0:
                    winsize -= 1
                if winsize <= 5:
                    return y
                return savgol_filter(y, winsize, 1, mode='mirror')

            plt.clf()
            plt.rcParams['agg.path.chunksize'] = 20000

            y_vals_s = smooth(y_vals, len(y_vals) // 250) if bSmooth else y_vals
            plt.plot(x_vals, y_vals_s)

            if np.std(y_vals[-1000:]) > 0:
                plt.ylim(ymin=np.min(y_vals_s[-int(len(y_vals)*his):])-1e-5, ymax=np.max(y_vals_s[-int(len(y_vals)*his):]))

            plt.xlabel('iteration')
            plt.ylabel(name)
            plt.tight_layout()
            plt.savefig(self.save_folder + name.replace(' ', '_') + '.pdf')

            if bLogScale and np.std(y_vals[-1000:]) > 0 and (np.max(y_vals) - np.min(y_vals)) / np.std(y_vals[-1000:]) > 1e3:
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(self.save_folder + name.replace(' ', '_') + '_log.pdf')

    def flush(self, stdlen=10):

        prints = []

        for name in np.sort(list(set(self.since_last_flush.keys()).union(set(self.since_beginning.keys())))):

            if self.since_last_flush.get(name) is not None:
                vals = self.since_last_flush.get(name)
                self.since_beginning[name].update(vals)
                if abs(np.std(list(vals.values()))) < 1e-5:
                    if abs(np.mean(list(vals.values()))) < 1e-5:
                        pass
                    else:
                        prints.append("{}:{:.4f}".format(name, np.mean(list(vals.values()))))
                else:
                    prints.append("{}:{:.4f}~{:.4f}({:.4f}±{:.4f})".format(name, np.min(list(vals.values())), np.max(list(vals.values())), np.mean(list(vals.values())), np.std(list(vals.values()))))
            else:
                x_vals = np.sort(list(self.since_beginning[name].keys()))
                y_vals = [self.since_beginning[name][x] for x in x_vals]
                prints.append("{}:{:.4f}({:.4f}±{:.4f})".format(name, y_vals[-1], self.mean_std[name].get('mean') if self.mean_std[name].get('mean') is not None else np.mean(y_vals[-stdlen:]), self.mean_std[name].get('std') if self.mean_std[name].get('std') is not None else np.std(y_vals[-stdlen:])))

        loginfo = "\n{} ITER:{}, {}".format((datetime.now().strftime('%Y-%m-%d %H:%M:%S ') if self.logtime else '') + self.casename, self.iter, ", ".join(prints))
        self.log(loginfo)

        self.since_last_flush.clear()
        self.logfile.flush()

    def save(self):
        if os.path.exists(self.save_folder + 'log.pkl'):
            copy(self.save_folder + 'log.pkl', self.save_folder + 'log_back.pkl')
        with open(self.save_folder + 'log.pkl', 'wb') as f:
            pickle.dump(dict(self.since_beginning), f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.save_folder + 'log.pkl', 'rb') as f:
            self.since_beginning.update(pickle.load(f))