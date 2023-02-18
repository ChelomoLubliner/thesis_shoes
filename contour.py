import numpy as np
import os
from numpy import asarray
import glob
import matplotlib.pyplot as plt
import imageio.v3 as iio
import skimage.color
import skimage.filters
import matplotlib.pyplot  as plt
from PIL import Image, ImageEnhance
import cv2


path = 'C:/Users/lubli/Documents/Thesis_Shoes/python_files/'
w, h = 307, 395





def main_contour():
    list_lines = np.load(path + 'Saved/list_matrices.npy')
    freq_min_18 = np.load(path + 'Saved/freq_min_18.npy')
    Image.fromarray(freq_min_18).show()