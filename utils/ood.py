# Adapted from https://github.com/hendrycks/outlier-exposure/blob/master/utils/display_results.py
import numpy as np
import sklearn.metrics as sk

recall_level_default = 0.95


def print_measures(auroc, aupr, tnr, recall_level=recall_level_default):
    return [
        'TNR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * tnr),
        'AUROC: \t\t\t{:.2f}'.format(100 * auroc),
        'AUPR:  \t\t\t{:.2f}'.format(100 * aupr),
    ]


def print_measures_with_std(aurocs, auprs, tnrs, recall_level=recall_level_default):
    return [
        'TNR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 * recall_level), 100 * np.mean(tnrs), 100 * np.std(tnrs)),
        'AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)),
        'AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)),
    ]