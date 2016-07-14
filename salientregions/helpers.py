'''
Module for helper functions, e.g. image and regions visualization, region to ellipse conversion, loading MAT files, array/vector difference etc.
'''
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from numpy import linalg as LA
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import six
from six.moves import range


def show_image(img, title=None):
    """Display the image.
    When a key is pressed, the window is closed

    Parameters
    ----------
    img :  numpy array
        image
    title : str, optional
        Title of the image
    """
    fig = plt.figure()
    plt.axis("off")
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    fig.canvas.set_window_title(title)
    if title is not None:
        fig.canvas.set_window_title(title)
        fig.suptitle(title)
    plt.gcf().canvas.mpl_connect('key_press_event',
                                 lambda event: plt.close(event.canvas.figure))
    plt.show()


# colormap bgr
colormap = {'holes': [255, 0, 0],  # BLUE
            'islands': [0, 255, 255],  # YELLOW
            'indentations': [0, 255, 0],  # GREEN
            'protrusions': [0, 0, 255]  # RED
            }

def visualize_elements(img, regions=None,
                       holes=None, islands=None,
                       indentations=None, protrusions=None,
                       visualize=True,
                       title='salient regions'):
    """Display the image with the salient regions provided.

    Parameters
    ----------
    img : numpy array
        image
    regions : dict
        dictionary with the regions to show
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

    # if the image is grayscale, make it BGR:
    if len(img.shape) == 2:
        img_to_show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_to_show = img.copy()

    if regions is not None:
        holes = regions.get("holes", None)
        islands = regions.get("islands", None)
        indentations = regions.get("indentations", None)
        protrusions = regions.get("protrusions", None)
    if holes is not None:
        img_to_show[[holes > 0]] = colormap['holes']
    if islands is not None:
        img_to_show[[islands > 0]] = colormap['islands']
    if indentations is not None:
        img_to_show[[indentations > 0]] = colormap['indentations']
    if protrusions is not None:
        img_to_show[[protrusions > 0]] = colormap['protrusions']

    if visualize:
        show_image(img_to_show, title=title)
    return img_to_show

def visualize_elements_ellipses(img, features,
                       visualize=True,
                       title='salient regions'):
    """Display the image with the salient regions provided.

    Parameters
    ----------
    img : numpy array
        image
    features : dict
        dictionary with the ellipse features of the regions to show
    visualize:  bool, optional
        visualizations flag
    display_name : str, optional
        name of the window


    Returns
    ----------
    img_to_show : numpy array
        image with the colored regions
    """
    if len(img.shape) == 2:
        img_to_show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_to_show = img.copy()
    for region_type in features.keys():
        img_to_show = visualize_ellipses(img_to_show, features[region_type],
                                         colormap[region_type], visualize=False)
    if visualize:
        show_image(img_to_show, title=title)
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
    holes = regions[:,:, 0]
    islands = regions[:,:, 1]
    indentations = regions[:,:, 2]
    protrusions = regions[:,:, 3]
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
        show_image(cv2.bitwise_xor(img1, img2), 'Difference between images')
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


def standard2poly_ellipse(half_major_axis, half_minor_axis, theta):
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

def poly2standard_ellipse(A, B, C):
    """ Conversion of elliptic polynomial coefficients to standard parameters.

    Parameters
    ----------
    A, B, C: floats
        The coefficients of the polynomial equation of an ellipse :math:`Ax^2 + Bxy + Cy^2 = 1`

    Returns
    ----------
    half_major_axis: float
        Half of the length of the ellipse's major axis
    half_minor_axis: float
        Half of the length of the ellipse's minor axis
    theta: float
        The ellipse orientation angle (radians) between the major and the x axis

     NOTE
     ------
     WARNING: The conversion might be correct only if the resulting angle is between 0 and pi/2!
    """
    # construct a matrix from the polynomial coefficients
    M = np.array([[A, B], [B, C]])

    # find the eigenvalues
    evals = LA.eigh(M)[0]
    order = evals.argsort()[::-1]
    evals = evals[order]
    e_min = evals[-1]
    e_max = evals[0]

    # derive the angle directly from the coefficients
    if B == 0:
        if A < C:
            theta = 0
        else:
            theta = np.pi/2
    else:
        if A < C:
            theta =  0.5*np.arctan(2*B/(A-C))
        else:
            theta = np.pi/2 + 0.5*np.arctan(2*B/(A-C))

    # axis lengths
    half_major_axis = 1/np.sqrt(e_min)
    half_minor_axis = 1/np.sqrt(e_max)

    return half_major_axis, half_minor_axis, theta

def binary_mask2ellipse_features_single(binary_mask, connectivity=4, saliency_type=1, min_square=False):
    """ Conversion of a single saliency type of binary regions to ellipse features.

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
    min_square: bool, optional
        whether to use minimum sqrt fitting for ellipses
        (default is bounded rotated rectangle fitting)

    Returns
    ----------
    num_regions: int
        The number of saleint regions of saliency_type
    features_standard: numpy array
        array with standard ellipse features for each of the ellipses for a given saliency type
    features_poly: numpy array
        array with polynomial ellipse features for each of the ellipses  for a given saliency type

    Notes
    ----------
    Every row in the resulting feature_standard array corresponds to a single
    region/ellipse and is of format:
    ``x0 y0 a b angle saliency_type`` ,
    where ``(x0,y0)`` are the coordinates of the ellipse centroid and ``a``, ``b`` and ``angle``(in degrees)
    are the standard parameters from the ellipse equation:
    math:`(x+cos(angle) + y+sin(angle))^2/a^2 + (x*sin(angle) - y*cos(angle))^2/b^2  = 1`

    Every row in the resulting feature_poly array corresponds to a single
    region/ellipse and is of format:
    ``x0 y0 A B C saliency_type`` ,
    where ``(x0,y0)`` are the coordinates of the ellipse centroid and ``A``, ``B`` and ``C``
    are the polynomial coefficients from the ellipse equation :math:`Ax^2 + Bxy + Cy^2 = 1`.
    """

    # num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=connectivity)
    binary_mask2 = binary_mask.copy()
    _, contours, hierarchy = cv2.findContours(
        binary_mask2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    indices_regions = np.where(hierarchy[0,:, 3] == -1)[0]
    num_regions = len(indices_regions)
    features_standard = np.zeros((num_regions, 6), float)
    features_poly = np.zeros((num_regions, 6), float)
    i = 0
    for index_regions in indices_regions:
        cnt = contours[index_regions]
        # fit an ellipse to the contour
        if min_square:
           # (x, y), (ma, MA), angle = cv2.fitEllipse(cnt)
            ellipse = cv2.fitEllipse(cnt)
            # center, axis_length and orientation of ellipse
            (center, axes, angle_deg) = ellipse
            # center of the ellipse
            (x, y) = center
            # length of MAJOR and minor axis
            MA = max(axes)
            ma = min(axes)
        else:
            (x, y), (ma, MA), angle_deg = cv2.minAreaRect(cnt)
            #(center, axes, angle_deg) = cv2.minAreaRect(cnt)

        # ellipse parameters


        a = np.fix(MA / 2)
        b = np.fix(ma / 2)

        if ((a > 0) and (b > 0)):
            x0 = x
            y0 = y
            if (angle_deg == 0):
                angle_deg = 180
            # angle_rad_manual = angle_deg * math.pi / 180
            angle_rad = math.radians(angle_deg)
            # compute the elliptic polynomial coefficients, aka features
            [A, B, C] = standard2poly_ellipse(a, b, -angle_rad)
            #[A, B, C] = standard2poly_ellipse(a, b, angle_rad)
            features_poly[i, ] = ([x0, y0, A, B, C, saliency_type])
            features_standard[i, ] = ([x, y, a, b, angle_rad, saliency_type])
        else:
            # We still output the ellipse as NaN
            features_poly[i,
                     ] = ([np.nan,
                           np.nan,
                           np.nan,
                           np.nan,
                           np.nan,
                           saliency_type])
        # standard parameters
        # features_standard[i, ] = ([x, y, a, b, angle_deg, saliency_type])

        i += 1
    return num_regions, features_standard, features_poly

def visualize_ellipses(img, features, color=(0, 0, 255), visualize=True):
    """ Visualise ellipses in an image

    Parameters
    ----------
    regions: img
        image to show the ellipses on
    features: numpy array
        standard ellipse features for each of the ellipses
    color: tuple of ints, optional
        color to show the ellipses
    visualize:  bool, optional
        visualizations flag

    Returns
    ----------
    img_to_show: numpy array
        image with the colored ellipses

    """
    # if the image is grayscale, make it BGR:
    if len(img.shape) == 2:
        img_to_show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_to_show = img.copy()

    for (x, y, a, b, angle_rad, _) in features:
    # for (x, y, a, b, angle_deg, _) in features:
        angle_deg = math.degrees(angle_rad)
        img_to_show = cv2.ellipse(img_to_show, (int(x), int(y)), (int(b), int(a)), int(angle_deg), 0, 360, color, 2)
       # img_to_show = cv2.ellipse(img_to_show, (int(x), int(y)), (int(a), int(b)), int(angle_deg), 0, 360, color, 2)
    if visualize:
        show_image(img_to_show)
    return img_to_show


def binary_mask2ellipse_features(regions, connectivity=4, min_square=False):
    """ Conversion of all types of regions to ellipse features.

    Parameters
    ----------
    regions: dict
        Dict of binary masks of the detected salient regions
    connectivity: int, optional
        Neighborhood connectivity
    min_square: bool, optional
        whether to use minimum sqrt fitting for ellipses
        (default is bounded rotated rectangle fitting)


    Returns
    ----------
    num_regions: dict
        The number of saleint regions for each saliency_type
    features_standard: dict
        dictionary with standard ellipse features for each of the ellipses
    features_poly: dict
        dictionary with polynomial ellipse features for each of the ellipses

    Note
    ----------
    The keys of the dictionaries are the saliency type.

    Every row in the array per key of  features_standard corresponds to a single
    region/ellipse and is of format:
    ``x0 y0 a b angle saliency_type`` ,
    where ``(x0,y0)`` are the coordinates of the ellipse centroid and ``a``, ``b`` and ``angle``(in degrees)
    are the standard parameters from the ellipse equation:
    math:`(x+cos(angle) + y+sin(angle))^2/a^2 + (x*sin(angle) - y*cos(angle))^2/b^2  = 1`

    Every row in the array per key of  features_poly corresponds to a single
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
    features_standard = {}
    features_poly = {}

    for saltype in regions.keys():
       # print "Saliency type: ", saltype
        num_regions_s, features_standard_s, features_poly_s =  binary_mask2ellipse_features_single(regions[saltype],
                                                connectivity=connectivity,  saliency_type=region2int[saltype], min_square=min_square)
        num_regions[saltype] = num_regions_s
        # print "Number of regions for that saliency type: ", num_regions_s
        features_standard[saltype] = features_standard_s
        features_poly[saltype] = features_poly_s

    return num_regions, features_standard, features_poly

def save_ellipse_features2file(num_regions, features, filename):
    """ Saving the ellipse features (polynomial or standard) to file.

    Parameters
    ----------
    num_regions: dict
        The number of saleint regions for each saliency type
    features: dict
        dictionary with ellipse features for each of the ellipses
    filename: str
        the filename where to save the features

    Returns
    --------
    total_num_regions: int
        the total number of salient regions of saliency types

    NOTES
    -------
    see load_ellipse_features_from_file

    """
    total_num_regions = 0

    # open the file in writing mode
    f = open(filename, 'w')

    for saltype in num_regions.keys():
        total_num_regions += num_regions[saltype]


    f.write('0 \n');
    f.write(str(total_num_regions))
    f.write('\n');

    for saltype in num_regions.keys():
        features_s = features[saltype]
        # print "saliency type: ", saltype
        # write into the file per ellipse
        # for ellipse_entry in features_poly_s: #
        for n in range(num_regions[saltype]):
            ellipse_entry = features_s[n,:]
            # print "n: features", n,":", ellipse_entry
            for e in ellipse_entry:
                f.write(str(e))
                f.write(' ')
            f.write('\n')

    # close the file
    f.close()

    return total_num_regions


def load_ellipse_features_from_file(filename):
    """ Load  elipse features (polynomial or standard) from, file.

    Parameters
    ----------
    filename: str
        the filename where to load the features from

    Returns
    --------
    total_num_regions: int
        the total number of salient regions of saliency types
    num_regions: dict
        The number of saleint regions for each saliency type
    features: dict
        dictionary with ellipse features for each of the ellipses

    NOTES
    -------
    see save_ellipse_features2file
   """

    # initializations
    region2int = {"holes": 1,
                  "islands":2,
                  "indentations": 3,
                  "protrusions": 4}
    int2region = {v: k for (k, v) in six.iteritems(region2int)}
    keys = list(region2int.keys())

    total_num_regions = 0
    num_regions = {k: 0 for k in keys}
    features_lists = {k: [] for k in keys}

    # open the filein mdoe reading
    f = open(filename, 'r')

    # skip the first line  (contains a 0)
    f.readline()
    # next one is the total number of regions
    total_num_regions = int(f.readline())

    # read off the feautres line by line
    for i in range(total_num_regions):
        line = f.readline()
        # get the last element- the type
        line_numbers = line.split()
        sal_type = int2region[int(float(line_numbers[-1]))]
        # make the string list- to a float list
        feature_list = [float(l) for l in line_numbers]
        features_lists[sal_type].append(feature_list)
        num_regions[sal_type] += 1

    # close the file
    f.close()

    # make numpy arrays from the lists
    features = {k: np.array(v) for (k, v) in six.iteritems(features_lists)}

    return total_num_regions, num_regions, features
