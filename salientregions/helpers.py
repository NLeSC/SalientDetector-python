# -*- coding: utf-8 -*-
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math


def show_image(img, window_name='image'):
    """Display the image.
    When a key is pressed, the window is closed

    Parameters
    ----------
    img :  numpy array
        image
    window_name : str, optional
        name of the window
    """
    fig = plt.figure()
    plt.axis("off")
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    fig.canvas.set_window_title(window_name)
    plt.gcf().canvas.mpl_connect('key_press_event',
                                 lambda event: plt.close(event.canvas.figure))
    plt.show()


def visualize_elements(img,
                       holes=None, islands=None,
                       indentations=None, protrusions=None,
                       visualize=True,
                       display_name='salient regions'):
    """Display the image with the salient regions provided.

    Parameters
    ----------
    img : numpy array
        image
    holes : numpy array
        Binary mask of the holes, to display in blue
    islands :  numpy array
        Binary mask of the islands, to display in yellow
    indentations : numpy array
        Binary mask of the indentations, to display in green
    protrusions :  numpy array
        Binary mask of the protrusions, to display in red
    visualize:  bool, optional
        visualizations flag
    display_name : str, optional
        name of the window


    Returns
    ----------
    img_to_show : numpy array
        image with the colored regions
    """
    # colormap bgr
    colormap = {'holes': [255, 0, 0],  # BLUE
                'islands': [0, 255, 255],  # YELLOW
                'indentations': [0, 255, 0],  # GREEN
                'protrusions': [0, 0, 255]  # RED
                }

    # if the image is grayscale, make it BGR:
    if len(img.shape) == 2:
        img_to_show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_to_show = img.copy()
    if holes is not None:
        img_to_show[[holes > 0]] = colormap['holes']
    if islands is not None:
        img_to_show[[islands > 0]] = colormap['islands']
    if indentations is not None:
        img_to_show[[indentations > 0]] = colormap['indentations']
    if protrusions is not None:
        img_to_show[[protrusions > 0]] = colormap['protrusions']

    if visualize:
        show_image(img_to_show, window_name=display_name)
    return img_to_show


def read_matfile(filename, visualize=True):
    """Read a matfile with the binary masks for the salient regions.
    Returns the masks with 0/255 values for the 4 salient types

    Parameters
    ----------
    filename: str
        Path to the mat file
    visualize: bool, optional
        option for visualizing the process

    Returns
    ----------
    holes:  numpy array
        Binary image with holes as foreground
    islands: numpy array
        Binary image with islands as foreground
    protrusions: numpy array
        Binary image with protrusions as foreground
    indentations: numpy array
        Binary image with indentations as foreground
    """
    matfile = sio.loadmat(filename)
    regions = matfile['saliency_masks'] * 255
    holes = regions[:, :, 0]
    islands = regions[:, :, 1]
    indentations = regions[:, :, 2]
    protrusions = regions[:, :, 3]
    if visualize:
        show_image(holes, 'holes')
        show_image(islands, 'islands')
        show_image(indentations, 'indentations')
        show_image(protrusions, 'protrusions')
    return holes, islands, indentations, protrusions


def image_diff(img1, img2, visualize=True):
    """Compares two images and shows the difference.
    Useful for testing purposes.

    Parameters
    ----------
    img1: numpy array
        first image to compare
    img2: numpy array
        second image to compare
    visualize: bool, optional
        option for visualizing the process

    Returns
    ----------
    is_same: bool
        True if all pixels of the two images are equal
    """
    if visualize:
        show_image(cv2.bitwise_xor(img1, img2), 'difference')
    return np.all(img1 == img2)


def array_diff(arr1, arr2, rtol=1e-05, atol=1e-08):
    """Compares two arrays. Useful for testing purposes.

    Parameters
    ----------
    arr1: 2-dimensional numpy, first array to compare
    arr2: 2-dimensional numpy, second array to compare

    Returns
    ----------
    is_close: bool
        True if elemetns of the two arrays are close within the defaults tolerance
        (see numpy.allclose documentaiton for tolerance values)
    """
    return np.allclose(arr1, arr2, rtol, atol)


def region2ellipse(half_major_axis, half_minor_axis, theta):
    """ Conversion of elliptic parameters to polynomial coefficients.

    Parameters
    ----------
    half_major_axis: float
        Half of the length of the ellipse's major axis
    half_minor_axis: float
        Half of the length of the ellipse's minor axis
    theta: float
        The ellipse orientation angle (radians) between the major and the x axis

    Returns
    ----------
    A, B, C: floats
        The coefficients of the polynomial equation of an ellipse :math:`Ax^2 + Bxy + Cy^2 = 1`
    """

    # trigonometric functions
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_cos_theta = sin_theta * cos_theta

    # squares
    a_sq = half_major_axis * half_major_axis
    b_sq = half_minor_axis * half_minor_axis
    sin_theta_sq = sin_theta * sin_theta
    cos_theta_sq = cos_theta * cos_theta

    # common denominator
    denom = a_sq * b_sq

    # polynomial coefficients
    A = (b_sq * cos_theta_sq + a_sq * sin_theta_sq) / denom
    B = ((b_sq - a_sq) * sin_cos_theta) / denom
    C = (b_sq * sin_theta_sq + a_sq * cos_theta_sq) / denom

    return A, B, C


def binary_mask2ellipse_features_single(binary_mask, connectivity=4, saliency_type=1):
    """ Conversion of a single type of binary regions to ellipse features.

    Parameters
    ----------
    binary_mask: 2-D numpy array
        Binary mask of the detected salient regions of the given saliency type
    connectivity: int
        Neighborhood connectivity
    saliency_type: int
        Type of salient regions. The code  is:
        1: holes
        2: islands
        3: indentations
        4: protrusions

    Returns
    ----------
    num_regions: int
        The number of saleint regions of saliency_type
    features: numpy array
        array with ellipse features for each of the ellipses

    Note
    ----------
    Every row in the resulting feature array corresponds to a single
    region/ellipse and is of format:
    ``x0 y0 A B C saliency_type`` ,
    where ``(x0,y0)`` are the coordinates of the ellipse centroid and ``A``, ``B`` and ``C``
    are the polynomial coefficients from the ellipse equation :math:`Ax^2 + Bxy + Cy^2 = 1`.
    """

    #num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=connectivity)
    binary_mask2 = binary_mask.copy()
    _, contours, _ = cv2.findContours(
        binary_mask2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    num_regions = len(contours)
    features = np.zeros((num_regions, 6), float)
    index_regions = 0

    for cnt in contours:

        index_regions = index_regions + 1

        # fit an ellipse to the contour
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
       # print "x,y: ", x, y
        # ellipse parameters
        a = np.fix(MA / 2)
        b = np.fix(ma / 2)

        if ((a > 0) and (b > 0)):
            x0 = x
            y0 = y
            if (angle == 0):
                angle = 180
            angle_rad = angle * math.pi / 180

            # compute the elliptic polynomial coefficients, aka features
            [A, B, C] = region2ellipse(a, b, -angle_rad)
            features[index_regions - 1, ] = ([x0, y0, A, B, C, saliency_type])
        else:
            features[index_regions - 1,
                     ] = ([np.nan,
                           np.nan,
                           np.nan,
                           np.nan,
                           np.nan,
                           saliency_type])

    return num_regions, features

def binary_mask2ellipse_features(regions, connectivity=4):
    """ Conversion of multiple types of regions to ellipse features.

    Parameters
    ----------
    regions: dict
        Dict of binary masks of the detected salient regions 
    connectivity: int
        Neighborhood connectivity


    Returns
    ----------
    num_regions: dict
        The number of saleint regions for each saliency_type
    features: dict
        array with ellipse features for each of the ellipses, for each saliency type

    Note
    ----------
    Every row in the resulting feature array corresponds to a single
    region/ellipse and is of format:
    ``x0 y0 A B C saliency_type`` ,
    where ``(x0,y0)`` are the coordinates of the ellipse centroid and ``A``, ``B`` and ``C``
    are the polynomial coefficients from the ellipse equation :math:`Ax^2 + Bxy + Cy^2 = 1`.
    """
    region2int = {"holes": 1,
                  "islands":2,
                  "indentations": 3,
                  "protrusions": 4}
    num_regions = {}
    features = {}
    for saltype in regions.keys():
        num_regions_s, features_s =  binary_mask2ellipse_features_single(regions[saltype], 
                                                              connectivity=connectivity, 
                                                              saliency_type=region2int[saltype])
        num_regions[saltype] = num_regions_s
        features[saltype] = features_s
    return num_regions, features