from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def array_to_image(array, type=None):
    plt.imshow(array, cmap='gray')
    
    if type == 'grad':
        plt.colorbar(label='gradient')
    
    plt.show()

def find_max_index(array):
    flattened_index = array.argmax()
    row_number = array.shape[0]
    
    row_index = int(flattened_index / row_number)
    col_index = flattened_index - row_number * row_index
    
    return row_index, col_index

def remove_max_pixel(image):
    row, col = find_max_index(image)
    image[row][col] = 0
