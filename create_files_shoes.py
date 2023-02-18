import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

path = 'C:/Users/lubli/Documents/Thesis_Shoes/python_files/'

my_dict = {1: True, 0: False}
w, h = 307, 395


# Opened file, save list_lines, and images
def contacts_data():
    list_lines = []
    with open(path + "Data/contacts_data.txt", 'r') as f:
        lines = f.readlines()
        print('Total lines:', len(lines))
        for i in tqdm(range(len(lines))):
            line = list(lines[i])
            line = np.array(list(map(int, line[:-1]))).reshape(h, w)
            list_lines.append(np.matrix(line))
            line = np.vectorize(my_dict.get)(line)
            img = Image.fromarray(line)
            img.save(path + 'Images/Shoes/im' + str(i) + '.png')

        # flip the image8 because it's left shoes
        list_lines[8] = np.fliplr(list_lines[8])
        line = np.vectorize(my_dict.get)(list_lines[8])
        im8 = Image.fromarray(line)
        im8.save(path + 'Images/Shoes/im8.png')
        # save list_lines
        np.save(path + 'Saved/list_matrices.npy', list_lines)
        print('Saved/list_matrices.npy OK')


def superposition_all_shoes(list_lines, len_lines):
    print('superposition_all_shoes')
    total = list_lines.sum(axis=0)
    Image.fromarray(total.astype(bool)).show()


# Sum every image matrice
def superposed_shoes(list_lines, len_lines):
    print('superposed_shoes')
    total_temp = list_lines[0]
    for i in tqdm(range(1, len_lines)):
        total_temp = total_temp + list_lines[i]
        if (i % 20 == 0) & (i != 0):
            line = np.vectorize(my_dict.get)(total_temp)
            img = Image.fromarray(line)
            img.save(path + 'Images/Superposed_Shoes/im' + str(i) + '.png')
            total_temp = list_lines[i]


def superposed_pixels(list_lines):
    print('/n superposed_pixels')
    # Dict to convert 0->False and non 0->True
    new_dict = {0: False}
    # max_pixel occurrence
    # total = superposed pixels
    total = list_lines.sum(axis=0)
    max_pixel = total.max()
    for max_pix in tqdm(range(max_pixel, 0, -1)):  # -1 -1
        new_dict[max_pix] = True
        line = np.vectorize(new_dict.get)(total)
        if max_pix == 18:
            np.save(path + 'Saved/freq_min_18.npy', line)
        if (max_pix % 10 == 0) | (max_pix < 25):
            img = Image.fromarray(line)
            img.save(path + 'Images/Superposed_Pixels/freq_min_' + str(max_pix) + '.png')


def superposed_pixels_reversed(list_lines):
    # Dict to convert 0->False and non 0->True
    print('/n superposed_pixels_reversed')
    new_dict = {0: False}
    total = list_lines.sum(axis=0)
    max_pixel = total.max()
    for max_pix in tqdm(range(1, max_pixel)):  # -1 -1
        new_dict[max_pix] = True
        line = np.vectorize(new_dict.get)(total)
        if (max_pix % 10 == 0) | (max_pix < 25):
            img = Image.fromarray(line)
            img.save(path + 'Images/Superposed_Pixels_Reversed/freq_max_' + str(max_pix) + '.png')


def heatmap_superposed(list_lines):
    print('/n heatmap_superposed')
    plt.imshow(list_lines.sum(axis=0), cmap='jet', interpolation='sinc')
    plt.show()


def main_load_superpositions():
    contacts_data()
    list_lines = np.load(path + 'Saved/list_matrices.npy')
    len_lines = len(list_lines)
    print('Array of ', len_lines, 'lines')
    superposition_all_shoes(list_lines, len_lines)
    superposed_shoes(list_lines, len_lines)
    superposed_pixels(list_lines)
    superposed_pixels_reversed(list_lines)
    heatmap_superposed(list_lines)

