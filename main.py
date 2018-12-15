import numpy as np
import skimage
import matplotlib.pyplot as plt

def generate_puzzle(img, n_segments=5):
    # grayscaling
    img = skimage.color.rgb2gray(img)

    # resizing to a square
    size = min(img.shape) // n_segments * n_segments
    #print('Size:', size)
    img = skimage.transform.resize(img, (size, size), anti_aliasing=True)

    #skimage.io.imshow(img)
    #plt.show()

    # cutting the image into different pieces
    segment_size = size // n_segments
    pieces = np.array([img[
        segment_size * i : segment_size * (i + 1),
        segment_size * j : segment_size * (j + 1)
    ] for i in range(n_segments) for j in range(n_segments)])

    # shuffling and rotating the pieces
    np.random.shuffle(pieces)
    pieces = np.array([skimage.transform.rotate(piece, 90 * np.random.randint(0, 4))
        for piece in pieces])

    # creating the 2d structure for the pieces
    pieces = pieces.reshape((n_segments, n_segments, segment_size, segment_size))
    #print('Pieces shape:', pieces.shape)

    '''f, ax = plt.subplots(n_segments, n_segments, figsize=(10, 10))

    for i in range(n_segments):
        for j in range(n_segments):
            ax[i, j].imshow(pieces[i, j], cmap='gray')
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])

    plt.tight_layout()
    plt.show()'''

    return pieces

if __name__ == '__main__':
    # read in an image
    img = skimage.io.imread('pictures/michelangelo-creation-of-adam.jpg')

    pieces = generate_puzzle(img)
    print(pieces.shape)
