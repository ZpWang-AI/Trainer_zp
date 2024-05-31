import numpy as np

from typing import *
from matplotlib import pyplot as plt


def plot_curve(
    *xys, title:str=None, savefig_path:str=None,
    x_name:str=None, y_name:str=None,
    x_range:Tuple[float, float]=None, y_range:Tuple[float, float]=None,
    close:bool=False,
):
    plt.plot(*xys,)
    plt.title(title)
    if x_name is not None:
        plt.xlabel(x_name)
    if y_name is not None:
        plt.ylabel(y_name)
    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)
    if savefig_path is not None:
        plt.savefig(savefig_path)
    if close:
        plt.close()
        
        
def mark_extremum(
    x, y, 
    mark_max=False, mark_min=False, 
    format_y_func=None,
):
    if format_y_func is None:
        format_y_func = lambda y:y
    if mark_max:
        mid = np.argmax(y)
        plt.text(x[mid], y[mid], str(format_y_func(y[mid])))
    if mark_min:
        mid = np.argmin(y)
        plt.text(x[mid], y[mid], str(format_y_func(y[mid])))
        
    