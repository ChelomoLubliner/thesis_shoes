import numpy as np
import os
from numpy import asarray
import glob
import matplotlib.pyplot as plt
import imageio.v3 as iio
import skimage.color
import skimage.filters
from scipy.interpolate import UnivariateSpline
import sys
from scipy.interpolate import interp2d
import plotly.graph_objs as go
import plotly.offline as pyo
#np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot  as plt
from PIL import Image, ImageEnhance
import cv2


path = 'C:/Users/lubli/Documents/Thesis_Shoes/python_files/'
w, h = 307, 395

def contour_openCV():
    # read the image
    image = cv2.imread(path +'Saved/im_18.png')
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply thresholding to convert the image to black and white
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # find the contours in the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create a black canvas with the same size as the original image
    canvas = np.zeros_like(image)
    # draw the contours on the canvas
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
    # convert the canvas to a binary image
    binary = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY)[1]
    bool_array = binary.astype(bool)
    np.save(path + 'Saved/contour_array.npy', bool_array)
    im_contour = Image.fromarray(bool_array)
    im_contour.save(path + 'Saved/im_contour.png')
    im_contour.show()

def plot_contour(coordinates):
    x = np.array([coord[0] for coord in coordinates])
    y = np.array([coord[1] for coord in coordinates])
    trace = go.Scatter(x=x, y=y, mode='markers')
    data = [trace]

    # Define the layout of the plot
    layout = go.Layout(title='Scatter plot of coordinates')

    # Create the plot and save it to an HTML file
    fig = go.Figure(data=data, layout=layout)
    pyo.plot(fig)

def superpose_image_contour(im_num):
    shoes = cv2.imread(path +'Images/Shoes/im'+str(im_num)+'.png')
    contour = cv2.imread(path+'Saved/im_18.png')
    dst = cv2.addWeighted(contour, 0.5, shoes, 0.7, 0)
    img_arr = np.hstack((shoes, dst))
    cv2.imshow('Blended Image',img_arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def remove_noise(im_num):
    list_matrices = np.load(path +'Saved/list_matrices.npy').astype(bool)
    contour = np.load(path+'Saved/freq_min_18.npy')
    #contour = np.array(Image.open(path+'Images/Superposed_Pixels/freq_min_160.png'), )
    combined_arr = np.bitwise_and(list_matrices[im_num], contour)
    combined_image = Image.fromarray(combined_arr)
    orignial_image = Image.fromarray(list_matrices[im_num])
    new_image = Image.new('1',(2*w, h), )
    new_image.paste(orignial_image,(0,0))
    new_image.paste(combined_image,(w,0))
    new_image.show()
    print(f"original image of {list_matrices[im_num].sum()} pixels, New image of {combined_arr.sum()} pixels, difference of {list_matrices[im_num].sum() - combined_arr.sum()} piels")
    return combined_arr


def main_contour():
    contour = np.load(path +'Saved/contour_array.npy')
    #only_contour = np.where(contour == True)
    coco = np.argwhere(contour == True)
    x = np.array([coord[0] for coord in coco])
    y = np.array([coord[1] for coord in coco])



    # Define some points:
    theta = np.linspace(-3, 2, 40)
    points = np.vstack((x, y)).T

    # add some noise:
    #points = points + 0.05 * np.random.randn(*points.shape)

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    # Build a list of the spline function, one for each dimension:
    splines = [UnivariateSpline(distance, coords, k=3, s=.2) for coords in points.T]

    # Computed the spline for the asked distances:
    alpha = np.linspace(0, 1, 75)
    points_fitted = np.vstack(spl(alpha) for spl in splines).T

    # Graph:
    plt.plot(*points.T, 'ok', label='original points');
    plt.plot(*points_fitted.T, '-r', label='fitted spline k=3, s=.2');
    plt.axis('equal');
    plt.legend();
    plt.xlabel('x');
    plt.ylabel('y');
    plt.show()

#main_contour()


def temp():
    # print(coco[0])
    # print(coco[1])
    # rng = np.random.default_rng()
    x = coco[0]  # np.linspace(-3, 3, 50)
    y = coco[1]  # np.exp(-x ** 2) + 0.1 * rng.standard_normal(50)
    plt.plot(x, y)
    # plt.show()





    # Create a scatter plot of the coordinates
    trace = go.Scatter(x=x, y=y, mode='markers')
    data = [trace]

    # Define the layout of the plot
    layout = go.Layout(title='Scatter plot of coordinates')

    # Create the plot and save it to an HTML file
    fig = go.Figure(data=data, layout=layout)
    pyo.plot(fig, filename='scatter-plot.html')
    """"
    for p, q in zip(x_values, y_values):
        x_cord = p  # try this change (p and q are already the coordinates)
        y_cord = q
        plt.scatter([x_cord], [y_cord])
    plt.show()
    """


def calculate_contour():
    arr_18 = np.load(path + 'saved/freq_min_18.npy')
    #Image.fromarray(arr_18).show()
    print(arr_18)
    all_points = np.argwhere(arr_18 == True)
    print(len(all_points))
    coco = all_points  # [500:600]
    x = np.array([coord[0] for coord in coco])
    y = np.array([coord[1] for coord in coco])

calculate_contour()