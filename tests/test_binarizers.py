# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .context import salientregions as sr
import unittest
import cv2
import os


class DataDrivenBinarizerTester(unittest.TestCase):

    '''
    Tests for the DataDriven binarizer
    '''

    def setUp(self):
        '''
        Load the test image and the 'true' binarized image
        '''
        testdata_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'images/Gray/'))
        testdata_features_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'features/Gray/'))
        self.image = cv2.imread(
            os.path.join(
                testdata_path,
                'Gray_scale.png'),
            cv2.IMREAD_GRAYSCALE)
        self.binarized_true = cv2.imread(
            os.path.join(
                testdata_features_path,
                'Binarization_data_driven.png'), cv2.IMREAD_GRAYSCALE)
        self.binarizer = sr.DatadrivenBinarizer(lam=24,
                                                area_factor_large=0.001,
                                                area_factor_verylarge=0.01,
                                                weights=(0.33, 0.33, 0.33),
                                                offset=80,
                                                stepsize=1,
                                                connectivity=8)
        self.threshold_true = 142

    def test_binarize(self):
        '''
        Test the function `binarize`
        Compare the binarized image.
        '''
        binarized = self.binarizer.binarize(self.image, visualize=False)
        assert sr.image_diff(self.binarized_true,
                                binarized,
                                visualize=False)

    def test_binarize_threshold(self):
        '''
        Test the function `binarize_withthreshold`
        Compare the resulting threshold.
        '''
        threshold, _ = self.binarizer.binarize_withthreshold(
            self.image, visualize=False)
        assert threshold == self.threshold_true


class ThresholdBinarizerTester(unittest.TestCase):

    '''
    Tests for the Threshold Binarizer
    '''

    def setUp(self):
        '''
        Load the test image and the 'true' binarized images for several thresholds
        '''
        testdata_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'images/Gray/'))
        self.image = cv2.imread(os.path.join(testdata_path, 'Gray_scale.png'))
        self.binarized_true_175 = cv2.imread(
            os.path.join(
                testdata_path,
                'Binarized_thresh175.png'), cv2.IMREAD_GRAYSCALE)
        self.threshold175 = sr.ThresholdBinarizer(175)
        self.binarized_true_57 = cv2.imread(
            os.path.join(
                testdata_path,
                'Binarized_thresh57.png'), cv2.IMREAD_GRAYSCALE)
        self.threshold57 = sr.ThresholdBinarizer(57)
        self.binarized_true_0 = cv2.imread(
            os.path.join(
                testdata_path,
                'Binarized_thresh0.png'), cv2.IMREAD_GRAYSCALE)
        self.threshold0 = sr.ThresholdBinarizer(0)
        self.binarized_true_255 = cv2.imread(
            os.path.join(
                testdata_path,
                'Binarized_thresh255.png'), cv2.IMREAD_GRAYSCALE)
        self.threshold255 = sr.ThresholdBinarizer(255)
        self.threshold256 = sr.ThresholdBinarizer(256)
        self.thresholdneg1 = sr.ThresholdBinarizer(-1)

    def test_binarize175(self):
        '''
        Test the function `binarize` for threshold 175.
        Compare the binarized image.
        '''
        binarized = self.threshold175.binarize(self.image, visualize=False)
        assert sr.image_diff(
            self.binarized_true_175,
            binarized,
            visualize=False)

    def test_binarize57(self):
        '''
        Test the function `binarize` for threshold 57.
        Compare the binarized image.
        '''
        binarized = self.threshold57.binarize(self.image, visualize=False)
        assert sr.image_diff(
            self.binarized_true_57,
            binarized,
            visualize=False)

    def test_binarize0(self):
        '''
        Test the function `binarize` for threshold 0.
        Compare the binarized image.
        '''
        binarized = self.threshold0.binarize(self.image, visualize=False)
        assert sr.image_diff(self.binarized_true_0, binarized, visualize=False)

    def test_binarize255(self):
        '''
        Test the function `binarize` for threshold 255.
        Compare the binarized image.
        '''
        binarized = self.threshold255.binarize(self.image, visualize=False)
        assert sr.image_diff(
            self.binarized_true_255,
            binarized,
            visualize=False)

    # Edge cases: if threshold outside of allowed range, it should be capped
    def test_binarize256(self):
        '''
        Test the function `binarize` for threshold 256.
        Compare the binarized image.
        '''
        binarized = self.threshold256.binarize(self.image, visualize=False)
        assert sr.image_diff(
            self.binarized_true_255,
            binarized,
            visualize=False)

    def test_binarizeneg1(self):
        '''
        Test the function `binarize` for threshold -1.
        Compare the binarized image.
        '''
        binarized = self.thresholdneg1.binarize(self.image, visualize=False)
        assert sr.image_diff(
            self.binarized_true_0,
            binarized,
            visualize=False)
