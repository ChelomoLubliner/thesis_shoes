import create_files_shoes
import contour
import numpy as np
path = 'C:/Users/lubli/Documents/Thesis_Shoes/python_files/'


if __name__ == '__main__':
    #376, 167 , 64, 144, 146
    list_matrices = np.load(path + 'Saved/list_matrices.npy').astype(bool)
    contour.remove_noise_get_contour(list_matrices, 40)