# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:10:57 2016

@author: elena
"""
from .context import salientregions as sr
import unittest
import cv2
import os
import numpy as np


class HelpersEllipseTester(unittest.TestCase):

    '''
    Tests for the helper functions related to ellipses
    '''

    def setUp(self):
        '''
        Load the binary masks to make ellipses from, and create the ground truths.
        '''
        # testing region to ellipse conversion
        self.major_axis_len = 15
        self.minor_axis_len = 9
        self.theta = 0.52
        self.coeff = [0.006395179230685, -0.003407029045900, 0.010394944226105]

        # testing elliptic features
        testdata_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'images/Binary/'))

        self.ellipse1_mask = np.array(
            cv2.imread(
                os.path.join(
                    testdata_path,
                    'Binary_ellipse1.png'), cv2.IMREAD_GRAYSCALE))

        self.features_standard_ellipse1 = np.array([200, 175, 34, 14, 0, 2])
        self.features_poly_ellipse1 = 100.00 * \
            np.array(
                [2.000000000000000,
                 1.750000000000000,
                 0.000008650519031,
                 -0.000000000000000,
                 0.000051020408163,
                 0.020000000000000])

        self.ellipse2_mask = np.array(
            cv2.imread(
                os.path.join(
                    testdata_path,
                    'Binary_ellipse2.png'), cv2.IMREAD_GRAYSCALE))
        self.features_standard_ellipse2 = np.array([187, 38.5, 10, 5, 90, 2])
        self.features_poly_ellipse2 = 100.00 * np.array([1.870000000000000,
                                                         0.385000000000000,
                                                         0.000400000000000,
                                                         0.000000000000000,
                                                         0.000100000000000,
                                                         0.020000000000000])

        self.ellipse3_mask = np.array(
            cv2.imread(
                os.path.join(
                    testdata_path,
                    'Binary_ellipse3.png'), cv2.IMREAD_GRAYSCALE))
        self.features_standard_ellipse3 = np.array(
            [101.9, 90.4, 24, 21, -9.9, 2])
        self.features_poly_ellipse3 = 100.00 * \
            np.array(
                [1.019717800289436,
                 0.904095513748191,
                 0.000017518811785,
                 -0.000000901804067,
                 0.000022518036288,
                 0.020000000000000])

        self.ellipse4_mask = np.array(
            cv2.imread(
                os.path.join(
                    testdata_path,
                    'Binary_ellipse4.png'), cv2.IMREAD_GRAYSCALE))
        self.features_standard_ellipse4 = np.array(
            [65.3, 186, 28, 13, 50.8, 2])
        self.features_poly_ellipse4 = 100.00 * np.array([0.653333333333333,
                                                         1.860687093779016,
                                                         0.000040675758984,
                                                         0.000022724787475,
                                                         0.000031250940690,
                                                         0.020000000000000])

        self.connectivty = 4
        self.rtol = 2  # default for np.allclose is 1e-05!!
        self.atol = 1e-02  # default for np.allclose is 1e-08

        # tesing the saving and loading
        self.num_regions = 7
        self.num_holes = 1
        self.num_islands = 2
        self.num_indent = 3
        self.num_protr = 1
        self.features = {}

        features_testpath = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'features/'))
        self.features_filename = os.path.join(
            features_testpath,
            'ellipse_features.txt')

    def test_region2ellipse(self):
        '''
        Test the function `region2ellipse`.
        '''
        A, B, C = sr.helpers.region2ellipse(
            self.major_axis_len, self.minor_axis_len, self.theta)
        coeff = [A, B, C]

        assert sr.helpers.array_diff(self.coeff, coeff)

    def test_mask2features_poly_ellipse1(self):
        '''
        Test the function `binary_mask2ellipse_features_single` for test image 1.
        '''
        _, _, features = sr.helpers.binary_mask2ellipse_features_single(
            self.ellipse1_mask, self.connectivty, 2)

        print "MATLAB features:", self.features_poly_ellipse1
        print "Python features:", features
        print 'Difference: ', features - self.features_poly_ellipse1
        print 'Max abs. difference: ', np.max(np.max(np.abs(features - self.features_poly_ellipse1)))

        assert sr.helpers.array_diff(
            self.features_poly_ellipse1,
            features,
            self.rtol,
            self.atol)

    def test_mask2features_poly_ellipse2(self):
        '''
        Test the function `binary_mask2ellipse_features_single` for test image 2.
        '''
        _, _, features = sr.helpers.binary_mask2ellipse_features_single(
            self.ellipse2_mask, self.connectivty, 2)

        print "MATLAB features:", self.features_poly_ellipse2
        print "Python features:", features
        print 'Difference: ', features - self.features_poly_ellipse2
        print 'Max abs.difference: ', np.max(np.max(np.abs(features - self.features_poly_ellipse2)))

        assert sr.helpers.array_diff(
            self.features_poly_ellipse2,
            features,
            self.rtol,
            self.atol)

    def test_mask2features_poly_ellipse3(self):
        '''
        Test the function `binary_mask2ellipse_features_single` for test image 3.
        '''
        _, _, features_poly = sr.helpers.binary_mask2ellipse_features_single(
            self.ellipse3_mask, self.connectivty, 2)

        assert sr.helpers.array_diff(
            self.features_poly_ellipse3,
            features_poly,
            self.rtol,
            self.atol)

    def test_mask2features_poly_ellipse4(self):
        '''
        Test the function `binary_mask2ellipse_features_single` for test image 4.
        '''
        _, _, features = sr.helpers.binary_mask2ellipse_features_single(
            self.ellipse4_mask, self.connectivty, 2)

        assert sr.helpers.array_diff(
            self.features_poly_ellipse4,
            features,
            self.rtol,
            self.atol)
