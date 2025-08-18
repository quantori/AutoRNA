import cv2
import numpy as np

def create_random_matrix(filename):
    image_size = 256
    squares_per_side = 8
    square_size = image_size // squares_per_side
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    for i in range(squares_per_side):
        for j in range(squares_per_side):
            color = np.random.randint(0, 256)  # Generate a random shade of grey
            start_x = i * square_size
            start_y = j * square_size
            image[start_x:start_x+square_size, start_y:start_y+square_size] = color

    cv2.imwrite(filename, image)

if __name__ == '__main__':
    filename = "for_article/grey_squares_new.png"
    create_random_matrix(filename)
