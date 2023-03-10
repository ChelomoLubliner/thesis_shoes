import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

import os
from numpy import asarray
from PIL import  ImageEnhance
import glob
import matplotlib.pyplot as plt
import imageio.v3 as iio
import skimage.color
import skimage.filters
from scipy.interpolate import UnivariateSpline
import sys
from scipy.interpolate import interp2d
#np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot  as plt


path = 'C:/Users/Chelomo/Desktop/These/Files_thesis/'
w, h = 307, 395





# superpose the prototype shoe on every shoe to see which pixels would be deleted
def superpose_image_contour(im_num):
    shoes = cv2.imread(path + 'Images/Shoes/im' + str(im_num) + '.png')
    contour = cv2.imread(path + 'Saved/im_18.png')
    dst = cv2.addWeighted(contour, 0.5, shoes, 0.7, 0)
    img_arr = np.hstack((shoes, dst))
    # Convert NumPy array to Pillow image
    Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)).show()

# the prototype shoe, freq_min_18 will be a limit for every shoe.
# Delete every pixel that isn't in the prototype shoe (bitwise_and)
def remove_noise(list_matrices, im_num):
    contour = np.load(path + 'Saved/freq_min_18.npy')
    # contour = np.array(Image.open(path+'Images/Superposed_Pixels/freq_min_160.png'), )
    combined_arr = np.bitwise_and(list_matrices[im_num], contour)
    combined_image = Image.fromarray(combined_arr)
    original_image = Image.fromarray(list_matrices[im_num])
    new_image = Image.new('1', (2 * w, h), )
    new_image.paste(original_image, (0, 0))
    new_image.paste(combined_image, (w, 0))
    new_image.show()
    print(f"original image of {list_matrices[im_num].sum()} pixels, New image of {combined_arr.sum()} "
          f"pixels, difference of {list_matrices[im_num].sum() - combined_arr.sum()} pixels")
    return combined_arr

# this algorithm receive a list of points, select_min_max_x/y
# union them (most points would be duplicated)
# return a list of point
def get_contour(im_array):
    all_points = np.argwhere(im_array==True)
    new_points_x = select_min_max_x(all_points)
    new_points_y = select_min_max_y(all_points)
    final_points = list(set().union(new_points_x, new_points_y))
    return final_points

def dict_points_x(all_points):
    points = all_points
    # create an empty dictionary to store the x and y values
    points_dict = {}
    # iterate over the list of points and add the x and y values to the dictionary
    for x_i, y_i in points:
        if x_i not in points_dict:
            points_dict[x_i] = [y_i]
        else:
            points_dict[x_i].append(y_i)
    return points_dict

# iterate over every X and take the first and last y pixel for each x
def select_min_max_x(all_points):
    points_dict =  dict_points_x(all_points)
    new_points_x = []
    for x_i in points_dict.keys():
        points_dict[x_i] = min(points_dict[x_i]), max(points_dict[x_i])
        for new_y_i in points_dict[x_i]:
            new_points_x.append((x_i, new_y_i))
    return new_points_x


# iterate over every Y and take the first and last X pixel for each Y
# this is essential to run both on x-axis and y-axis because there are some lines
# (horizontally or vertically that could be deleted)
def select_min_max_y(all_points):
    points = all_points
    # create an empty dictionary to store the x and y values
    points_dict = {}

    # iterate over the list of points and add the x and y values to the dictionary
    for x_i, y_i in points:
        if y_i not in points_dict:
            points_dict[y_i] = [x_i]
        else:
            points_dict[y_i].append(x_i)
    new_points_y = []
    for y_i in points_dict.keys():
        points_dict[y_i] = min(points_dict[y_i]), max(points_dict[y_i])
        for new_x_i in points_dict[y_i]:
            new_points_y.append((new_x_i, y_i))
    return new_points_y


# get an array of coordinates (such as contour pixels)
# return a scatter plot of these points
def scatter_plot_contour(coordinates,im_num):
    x = np.array([coord[0] for coord in coordinates])
    y = np.array([coord[1] for coord in coordinates])
    trace = go.Scatter(x=x, y=y, mode='markers')
    data = [trace]
    # Define the layout of the plot
    layout = go.Layout(title='Scatter plot of contour Image '+str(im_num))
    # Create the plot and save it to an HTML file
    fig = go.Figure(data=data, layout=layout)
    pyo.plot(fig, filename=path +'Images/Scatter_Plot/plot_cont'+str(im_num)+'.html')







def save_new_contour_shoe(new_points, im_num):
    new_arr = np.zeros((h, w), dtype=bool)
    new_arr[tuple(zip(*new_points))] = True
    new_image = Image.fromarray(new_arr)
    new_image.save(path + 'Images/Contour_Shoes/cont_im' + str(im_num) + '.png')
    # Image.fromarray


# this is a complete flow for an image
def remove_noise_get_contour(list_matrices, im_num):
    superpose_image_contour(im_num)
    new_im = remove_noise(list_matrices, im_num)
    final_points = get_contour(new_im)
    print('Scatter Plot')
    scatter_plot_contour(final_points, im_num)
    save_new_contour_shoe(final_points, im_num)

def set_continuous_100_300(coordinates):
    points = coordinates
    # create an empty dictionary to store the x and y values
    points_dict = {}
    # iterate over the list of points and add the x and y values to the dictionary
    for x_i, y_i in points:
        if x_i not in points_dict:
            points_dict[x_i] = [y_i]
        else:
            points_dict[x_i].append(y_i)
    for x_i in points_dict.keys():
        if x_i in range(100, 300):
            if x_i+1 not in points_dict:
                points_dict[x_i+1] = points_dict[x_i]
    for x_i in points_dict.keys():
        if x_i in range(100, 300):
            print(x_i, points_dict[x_i])

def alg_repair(points_dict):
    # iterate over the list of points and add the x and y values to the dictionary
    for x_i in range(100,300):
        if x_i not in points_dict:
            points_dict[x_i] = points_dict[x_i-1]
    for x_i in range(100,300):
        print('before', points_dict[x_i])
        for j in [0, 1]:
            if abs(points_dict[x_i][j] - points_dict[x_i - 1][j]) > 1:
                points_dict[x_i][j] = points_dict[x_i - 1][j]
        print('after', points_dict[x_i])

    new_points_x = []
    for x_i in points_dict.keys():
        if x_i in range(100,300):
            for new_y_i in points_dict[x_i]:
                new_points_x.append((x_i, new_y_i))
    return new_points_x

def only_100_300(points_dict):
    for x_i in range(100,300):
        if x_i not in points_dict:
            points_dict[x_i] = points_dict[x_i-1]
    new_points_x = []
    for x_i in points_dict.keys():
        if x_i in range(100, 300):
            for new_y_i in points_dict[x_i]:
                new_points_x.append((x_i, new_y_i))
    return new_points_x



    """new_points_x = []
    for x_i in points_dict.keys():
        points_dict[x_i] = min(points_dict[x_i]), max(points_dict[x_i])
        for new_y_i in points_dict[x_i]:
            new_points_x.append((x_i, new_y_i))
    return new_points_x"""
def main_contour():
    arr_64 = np.asarray(Image.open(path + 'Images/Contour_Shoes/cont_im64.png'))
    all_points = np.argwhere(arr_64 == True)
    new_points = select_min_max_x(all_points)
    dict_x = dict_points_x(new_points)
    points = only_100_300(dict_x)
    x = np.array([coord[0] for coord in points])
    y = np.array([coord[1] for coord in points])
    spl = UnivariateSpline(x, y)
    xs = np.linspace(-3, 3, 1000)
    plt.plot(xs, spl(xs), 'g', lw=3)
    #set_continuous_100_300(new_points)
    #new_points_x = alg_repair(dict_x)
    #scatter_plot_contour(points, 64)
    #list_matrices = np.load(path + 'Saved/list_matrices.npy').astype(bool)
    #remove_noise_get_contour(list_matrices, 122)



#main_contour()