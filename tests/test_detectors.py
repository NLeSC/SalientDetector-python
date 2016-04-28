# -*- coding: utf-8 -*-
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
