'''
Binary detection of salient regions using mathematical moprhology.
'''
from __future__ import absolute_import
import cv2
from . import helpers
import numpy as np
from six.moves import range


class BinaryDetector(object):

    """
    Class for detecting salient regions in binary images.

    Parameters
    ------
    SE : numpy array
            The structuring element to use in processing the image
    lam : float
        lambda, minimumm area of a connected component
    area_factor: float
        factor that describes the minimum area of a significent CC
    connectivity: int
        What connectivity to use to define CCs

    Attributes
    ------
    holes : numpy array
        binary mask of the holes
    islands : numpy array
        binary mask of the islands
    indentations : numpy array
        binary mask of the indentations
    protrusions : numpy array
        binary mask of the protrusions

    Note
    ------
    The methods `detect`, `get_holes`, `get_islands`, `get_indentations`
    and `get_protrusions` invoke the calculation of the regions. After that, the
    regions are also available as attributes `holes`, `islands`, `indentations`
    and `protrusions`.
    """

    def __init__(self, SE, lam, area_factor, connectivity):
        self.SE = SE
        self.lam = lam
        self.area_factor = area_factor
        self.connectivity = connectivity
        self._img = None
        self._invimg = None
        self._filled = None
        self._invfilled = None
        self.holes = None
        self.islands = None
        self.indentations = None
        self.protrusions = None

    def detect(self, img, find_holes=True, find_islands=True,
               find_indentations=True, find_protrusions=True, visualize=True):
        """Find salient regions of the types specified.

        Parameters
        ------
        img: numpy array
            binary image to detect regions
        find_holes: bool, optional
            Whether to detect regions of type hole
        find_islands: bool, optional
            Whether to detect regions of type island
        find_indentations: bool, optional
            Whether to detect regions of type indentation
        find_protrusions: bool, optional
            Whether to detect regions of type protrusion
        visualize: bool, optional
            option for visualizing the process

        Returns
        ------
        regions: dict
            For each type of region, the maks with detected regions.
        """
        regions = {}
        self.reset()
        self._img = img
        # Get holes and islands
        if find_holes:
            regions['holes'] = self.get_holes()

        # Islands are just holes of the inverse image
        if find_islands:
            regions['islands'] = self.get_islands()

        # Get indentations and protrusions
        if find_indentations:
            regions['indentations'] = self.get_indentations()

        if find_protrusions:
            regions['protrusions'] = self.get_protrusions()

        if visualize:
            helpers.visualize_elements(
                img, holes=regions.get(
                    'holes', None), islands=regions.get(
                    'islands', None), indentations=regions.get(
                    'indentations', None), protrusions=regions.get(
                    'protrusions', None),
                title='Salient regions in binary image')
        return regions

    def reset(self):
        """ Reset all attributes.
        """
        self._img = None
        self._invimg = None
        self._filled = None
        self._invfilled = None
        self.holes = None
        self.islands = None
        self.indentations = None
        self.protrusions = None

    def get_holes(self):
        """Get salient regions of type 'hole'
        """
        if self.holes is None:
            # Fill the image
            self._filled = _fill_image(self._img, self.connectivity)

            # Detect the holes
            self.holes = self._detect_holelike(
                img=self._img, filled=self._filled)
        return self.holes

    def get_islands(self):
        """Get salient regions of type 'island'
        """
        if self.islands is None:
            # Get the inverse image
            self._invimg = cv2.bitwise_not(self._img)
            # Fill the inverse image
            self._invfilled = _fill_image(self._invimg, self.connectivity)
            self.islands = self._detect_holelike(
                img=self._invimg, filled=self._invfilled)
        return self.islands

    def get_protrusions(self):
        """Get salient regions of type 'protrusion'
        """
        if self.protrusions is None:
            holes = self.get_holes()
            self.protrusions = self._detect_protrusionlike(
                self._img, self._filled, holes)
        return self.protrusions

    def get_indentations(self):
        """Get salient regions of type 'indentation'
        """
        if self.indentations is None:
            islands = self.get_islands()
            self.indentations = self._detect_protrusionlike(
                self._invimg, self._invfilled, islands)
        return self.indentations

    def _detect_holelike(self, img, filled):
        """Detect hole-like salient regions, using the image and its filled version

        Parameters
        ------
        img: 2-dimensional numpy array with values 0/255
            Image to detect holes
        filled: 2-dimensional numpy array with values 0/255, optional
            Precomputed filled image

        Returns
        ------
        holes: 2-dimensional numpy array with values 0/255
            Mask with all holes as foreground.
        """

        # Get all the holes (including those that are noise)
        all_the_holes = cv2.bitwise_and(filled, cv2.bitwise_not(img))
        # Substract the noise elements
        theholes = self._remove_small_elements(all_the_holes,
                                               remove_border_elements=True)
        return theholes

    def _detect_protrusionlike(self, img, filled, holes):
        """Detect 'protrusion'-like salient regions

        Parameters
        ------
        img: 2-dimensional numpy array with values 0/255
            image to detect holes
        filled: 2-dimensional numpy array with values 0/255, optional
            precomputed filled image
        holes: 2-dimensional numpy array with values 0/255
            The earlier detected holes

        Returns
        ------
        protrusions: 2-dimensional numpy array with values 0/255
            Image with all protrusions as foreground.
        """

        # Calculate minimum area for connected components
        min_area = self.area_factor * img.size

        # Initalize protrusion image
        prots1 = np.zeros(img.shape, dtype='uint8')
        prots2 = np.zeros(img.shape, dtype='uint8')

        # Retrieve all connected components
        nccs, labels, stats, centroids = cv2.connectedComponentsWithStats(
            filled, connectivity=self.connectivity)
        for i in range(1, nccs):
            area = stats[i, cv2.CC_STAT_AREA]
            # For the significant CCs, perform tophat
            if area > min_area:
                ccimage = np.array(255 * (labels == i), dtype='uint8')
                wth = cv2.morphologyEx(ccimage, cv2.MORPH_TOPHAT, self.SE)
                prots1 += wth

        prots1_nonoise = self._remove_small_elements(prots1)

        # Now get indentations of significant holes
        nccs2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(
            holes, connectivity=self.connectivity)
        for i in range(1, nccs2):
            area = stats2[i, cv2.CC_STAT_AREA]
            ccimage = np.array(255 * (labels2 == i), dtype='uint8')
            ccimage_filled = _fill_image(ccimage, self.connectivity)
            # For the significant CCs, perform tophat
            if area > min_area:
                bth = cv2.morphologyEx(
                    ccimage_filled, cv2.MORPH_BLACKHAT, self.SE)
                prots2 += bth

        prots2_nonoise = self._remove_small_elements(prots2)

        prots = cv2.add(prots1_nonoise, prots2_nonoise)
        return prots

    def _remove_small_elements(
            self,
            elements,
            connectivity=None,
            remove_border_elements=True,
            visualize=False):
        """Remove elements (Connected Components) that are smaller
        then a given threshold

        Parameters
        ------
        elements : numpy array
            binary image with elements
        connectivity: int, optional
            What connectivity to use to define CCs
        remove_border_elements: bool, optional
            Also remove elements that are attached to the border
        visualize: bool, optional
            option for visualizing the process

        Returns
        ------
        result : numpy array
            Binary image with all elements larger then lam
        """
        if connectivity is None:
            connectivity = self.connectivity
        result = elements.copy()
        nr_elements, labels, stats, _ = cv2.connectedComponentsWithStats(
            elements, connectivity=connectivity)

        leftborder = 0
        rightborder = elements.shape[1]
        upperborder = 0
        lowerborder = elements.shape[0]
        for i in range(1, nr_elements):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.lam:
                result[[labels == i]] = 0

            if remove_border_elements:
                xmin = stats[i, cv2.CC_STAT_LEFT]
                xmax = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
                ymin = stats[i, cv2.CC_STAT_TOP]
                ymax = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
                if xmin <= leftborder \
                        or xmax >= rightborder \
                        or ymin <= upperborder \
                        or ymax >= lowerborder:
                    result[[labels == i]] = 0
        if visualize:
            helpers.show_image(result, 'Small elements removed')
        return result


def _fill_image(img, connectivity):
    """Fills all holes in connected components in a binary image.

    Parameters
    ------
    img : numpy array
        binary image to fill

    Returns
    ------
    filled : numpy array
        The filled image
    """
    # Copy the image with an extra border
    h, w = img.shape[:2]
    img_border = np.zeros((h + 2, w + 2), np.uint8)
    img_border[1:-1, 1:-1] = img

    floodfilled = img_border.copy()
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(floodfilled, mask, (0, 0), 255, flags=connectivity)
    floodfill_inv = cv2.bitwise_not(floodfilled)
    filled = img_border | floodfill_inv
    filled = filled[1:-1, 1:-1]
    return filled
