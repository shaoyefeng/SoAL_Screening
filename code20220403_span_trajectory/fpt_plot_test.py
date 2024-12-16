# -*- coding: utf-8 -*-
import os
import pickle
from fpt_analysis import load_dfs_bouts, load_folder_dfs_bouts
from fpt_plot import plot_by_name, plot_summary_by_name

g_cache = {}
def load_dfs_bouts_cache(pair_folder, postfix=None): # _we_stat0.pickle
    if not postfix:
        postfix = "_cir_center_stat0.pickle"
    if isinstance(pair_folder, list):
        prefix = "list"
    else:
        prefix = pair_folder
    if g_cache.get(prefix + postfix):
        return g_cache[prefix + postfix]
    if isinstance(pair_folder, list):
        ret = load_folder_dfs_bouts(pair_folder, postfix=postfix)
    elif not os.path.basename(pair_folder).startswith("20"):  # geno_folder
        ret = load_folder_dfs_bouts(pair_folder, postfix=postfix)
    else:
        ret = load_dfs_bouts(pair_folder, postfix=postfix)
    g_cache[prefix + postfix] = ret
    return ret
def save_cache(geno):
    pickle.dump(g_cache, open("img/cache_%s.pickle" % geno, "wb"))
def restore_cache(geno):
    f = "img/cache_%s.pickle" % geno
    if not os.path.exists(f):
        return
    global g_cache
    g_cache = pickle.load(open(f, "rb"))

def plot_test(folder, name, names=None, need_save_cache=False, **args):
    geno = os.path.basename(folder)
    dfs, bouts, n = load_dfs_bouts_cache(folder)
    filename = "img/%s-%s" % (geno, name.replace(":", ","))
    if need_save_cache:
        save_cache(geno)
    if names:
        plot_summary_by_name(names, dfs, filename, bouts=bouts, n=n, **args)
    else:
        plot_by_name(dfs, filename, bouts=bouts, **args)

plot_test(r"D:\exp\data4\geno_data\FW31\20191224_150730_A_8", "d_theta", ["d_theta", "d_speed"], col=2, save_svg=False)
