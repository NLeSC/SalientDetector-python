"""
Testing the detection functions.
"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .context import salientregions as sr
from .context import salientregions_detectors as srd
import unittest
import cv2
import os
import numpy as np


class _DetectorForTesting(srd.Detector):

    '''
    Mock implementation of Detector that inherits from the superclass.
    '''

    def detect(self):
        pass


class SETester(unittest.TestCase):

    '''
    Test class for the Detector method `getSE`
    '''

    def setUp(self):
        self.SE_true = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]],
                                dtype='uint8')
        self.lam_true = 10
        self.detector = _DetectorForTesting(SE_size_factor=0.05,
                                            lam_factor=5,
                                            area_factor=0.05,
                                            connectivity=4)

    def test_getSE(self):
        '''
        Test the method `getSE` and assert that the Structuring Element is correct.
        '''
        self.detector.get_SE(100 * 110)
        SE = self.detector.SE
        assert np.all(SE == self.SE_true)

    def test_getlam(self):
        '''
        Test the method `getSE` and assert that the value of lamda is correct.
        '''
        self.detector.get_SE(100 * 110)
        lam = self.detector.lam
        assert lam == self.lam_true


class SalientRegionDetectorTester(unittest.TestCase):

    '''
    Test class for the SalientRegionDetector
    '''

    def setUp(self):
        '''
        loads images and create detector objects
        '''
        SE_size_factor = 0.02
        lam_factor = 3
        area_factor = 0.001
        connectivity = 8

        area_factor_large = 0.001
        area_factor_very_large = 0.01
        weight_all = 0.33
        weight_large = 0.33
        weight_very_large = 0.33
        offset = 80
        stepsize = 1
        self.lam_gray = 24
        self.lam_color = 27

        binarizer_gray = sr.DatadrivenBinarizer(lam=self.lam_gray,
                                                area_factor_large=area_factor_large,
                                                area_factor_verylarge=area_factor_very_large,
                                                weights=(
                                                weight_all,
                                                weight_large,
                                                weight_very_large),
                                                offset=offset,
                                                stepsize=stepsize,
                                                connectivity=connectivity)
        self.det_gray = sr.SalientDetector(
            binarizer=binarizer_gray,
            SE_size_factor=SE_size_factor,
            lam_factor=lam_factor,
            connectivity=connectivity)

        self.testdata_features_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'features/'))
        self.holes_true_gray, self.islands_true_gray, _, _ = sr.read_matfile(
            os.path.join(self.testdata_features_path, 'Gray/Gray_scale_dmsrallregions.mat'), visualize=False)

        binarizer_color = sr.DatadrivenBinarizer(lam=self.lam_color,
                                                 area_factor_large=area_factor_large,
                                                 area_factor_verylarge=area_factor_very_large,
                                                 weights=(
                                                 weight_all,
                                                 weight_large,
                                                 weight_very_large),
                                                 offset=offset,
                                                 stepsize=stepsize,
                                                 connectivity=connectivity)
        self.det_color = sr.SalientDetector(
            binarizer=binarizer_color,
            SE_size_factor=SE_size_factor,
            area_factor=area_factor,
            lam_factor=lam_factor,
            connectivity=connectivity)

        self.holes_true_color, self.islands_true_color, _, _ = sr.read_matfile(
            os.path.join(self.testdata_features_path, 'Color/color_dmsrallregions.mat'), visualize=False)

        self.det_default = sr.SalientDetector()

        testdata_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'images/Gray/'))
        self.img_gray = cv2.imread(
            os.path.join(
                testdata_path,
                'Gray_scale.png'), cv2.IMREAD_GRAYSCALE)
        testdata_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'images/Color/'))
        self.img_color = cv2.imread(
            os.path.join(
                testdata_path,
                'color.png'))

    def test_gray_image(self):
        '''
        Tests the salient detector on a gray scale image with specific settings
        '''
        regions = self.det_gray.detect(self.img_gray,
                                       find_holes=True,
                                       find_islands=True,
                                       find_indentations=True,
                                       find_protrusions=True,
                                       visualize=False)
        # TODO: compare with known regions
        assert 'holes' in regions
        assert 'islands' in regions
        assert 'indentations' in regions
        assert 'protrusions' in regions
        assert self.det_gray.lam == self.lam_gray
        assert sr.image_diff(
            regions['holes'],
            self.holes_true_gray,
            visualize=False)
        assert sr.image_diff(
            regions['islands'],
            self.islands_true_gray,
            visualize=False)

    def test_color_image(self):
        '''
        Tests the salient detector on a gray scale image with specific settings
        '''
        regions = self.det_color.detect(self.img_color,
                                        find_holes=True,
                                        find_islands=True,
                                        find_indentations=True,
                                        find_protrusions=True,
                                        visualize=False)
        # TODO: compare with known regions
        assert 'holes' in regions
        assert 'islands' in regions
        assert 'indentations' in regions
        assert 'protrusions' in regions
        assert self.det_color.lam == self.lam_color
        assert sr.image_diff(
            regions['holes'],
            self.holes_true_color,
            visualize=False)
        assert sr.image_diff(
            regions['islands'],
            self.islands_true_color,
            visualize=False)


class MSSRDetectorTester(unittest.TestCase):

    '''
    Test class for the SalientRegionDetector
    '''

    def setUp(self):
        '''
        loads images and create detector objects
        '''
        testdata_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'images/Gray/'))
        self.img_gray = cv2.imread(
            os.path.join(
                testdata_path,
                'Gray_scale.png'), cv2.IMREAD_GRAYSCALE)
        testdata_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'images/Color/'))
        self.img_color = cv2.imread(
            os.path.join(
                testdata_path,
                'color.png'))

        testdata_features_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'features/Color/'))
        self.holes_true, self.islands_true, _, _ = sr.read_matfile(
            os.path.join(testdata_features_path, 'color_allregions.mat'), visualize=False)

        SE_size_factor = 0.02
        lam_factor = 5
        area_factor = 0.03
        connectivity = 4
        min_thres = 1
        max_thres = 255
        stepsize = 10
        perc = 0.6
        self.lam = 45

        self.det = sr.MSSRDetector(
            min_thres=min_thres, max_thres=max_thres, step=stepsize,
            perc=perc, SE_size_factor=SE_size_factor,
            lam_factor=lam_factor,
            area_factor=area_factor,
            connectivity=connectivity)

    # def test_gray(self):
    # 
    #     regions = self.det.detect(self.img_gray,
    #                               visualize=False)
    #     assert self.det.lam == self.lam
    #     assert 'holes' in regions
    #     assert 'islands' in regions
    #     assert 'indentations' in regions
    #     assert 'protrusions' in regions

    def test_color(self):
        '''
        Test the MSSRA detector on a color image
        '''
        regions = self.det.detect(self.img_color,
                                  visualize=False)
        assert 'holes' in regions
        assert 'islands' in regions
        assert 'indentations' in regions
        assert 'protrusions' in regions
        assert sr.image_diff(
            regions['holes'],
            self.holes_true,
            visualize=False)
        assert sr.image_diff(
            regions['islands'],
            self.islands_true,
            visualize=False)
