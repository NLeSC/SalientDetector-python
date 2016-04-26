# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:10:40 2016

@author: dafne
"""
from .context import salientregions as sr
import unittest
import cv2
import os
import scipy.io as sio


class BinaryDetectorTester(unittest.TestCase):

    '''
    Tests for the binary detector
    '''

    def setUp(self):
        '''
        Load the binary image and the binary masks for the true regions.
        '''
        testdata_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'images/Binary/'))
        self.image_noise = cv2.imread(
            os.path.join(
                testdata_path,
                'Binary_all_types_noise.png'), cv2.IMREAD_GRAYSCALE)
        self.image_nested = cv2.imread(os.path.join(
            testdata_path,
            'Binary_nested.png'), cv2.IMREAD_GRAYSCALE)
        self.holes_true, self.islands_true, self.indents_true, self.prots_true = sr.read_matfile(
            os.path.join(testdata_path, 'Binary_all_types_noise_binregions.mat'), visualize=False)
        self.filled_image_noise_true = cv2.imread(
            os.path.join(
                testdata_path,
                'Binary_all_types_noise_filled.png'), cv2.IMREAD_GRAYSCALE)
        self.filled_image_nested_true = cv2.imread(
            os.path.join(
                testdata_path,
                'Binary_nested_filled.png'), cv2.IMREAD_GRAYSCALE)

        SE = sio.loadmat(
            os.path.join(
                testdata_path,
                "SE_neighb_all_other.mat"))['SE_n']
        lam = 50
        area_factor = 0.05
        connectivity = 4
        self.binarydetector = sr.BinaryDetector(
            SE=SE, lam=lam, area_factor=area_factor, connectivity=connectivity)

    def test_holes(self):
        '''
        Test the method `detect` for only holes.
        '''
        results = self.binarydetector.detect(
            self.image_noise,
            find_holes=True,
            find_islands=False,
            find_indentations=False,
            find_protrusions=False,
            visualize=False)
        holes_my = results['holes']
        assert sr.image_diff(self.holes_true, holes_my, visualize=False)

    def test_islands(self):
        '''
        Test the method `detect` for only islands.
        '''
        results = self.binarydetector.detect(
            self.image_noise,
            find_holes=False,
            find_islands=True,
            find_indentations=False,
            find_protrusions=False,
            visualize=False)
        islands_my = results['islands']
        assert sr.image_diff(self.islands_true, islands_my, visualize=False)

    def test_protrusions(self):
        '''
        Test the method `detect` for only protrusions.
        '''
        results = self.binarydetector.detect(
            self.image_noise,
            find_holes=False,
            find_islands=False,
            find_indentations=False,
            find_protrusions=True,
            visualize=False)
        prots_my = results['protrusions']
        assert sr.image_diff(self.prots_true, prots_my, visualize=False)

    def test_indentations(self):
        '''
        Test the method `detect` for only indentations.
        '''
        results = self.binarydetector.detect(
            self.image_noise,
            find_holes=False,
            find_islands=False,
            find_indentations=True,
            find_protrusions=False,
            visualize=False)
        indents_my = results['indentations']
        assert sr.image_diff(self.indents_true, indents_my, visualize=False)

    def test_holesislands(self):
        '''
        Test the method `detect` for holes and islands.
        '''
        results = self.binarydetector.detect(
            self.image_noise,
            find_holes=True,
            find_islands=True,
            find_indentations=False,
            find_protrusions=False,
            visualize=False)
        holes_my = results['holes']
        islands_my = results['islands']
        assert sr.image_diff(self.holes_true, holes_my, visualize=False)
        assert sr.image_diff(self.islands_true, islands_my, visualize=False)

    def test_protsindents(self):
        '''
        Test the method `detect` for protrusions and indentations.
        '''
        results = self.binarydetector.detect(
            self.image_noise,
            find_holes=False,
            find_islands=False,
            find_indentations=True,
            find_protrusions=True,
            visualize=False)
        assert sr.image_diff(
            self.indents_true,
            results['indentations'],
            visualize=False)
        assert sr.image_diff(
            self.prots_true,
            results['protrusions'],
            visualize=False)

    def test_detect(self):
        '''
        Test the method `detect` for all regions.
        '''
        results = self.binarydetector.detect(
            self.image_noise,
            find_holes=True,
            find_islands=True,
            find_indentations=True,
            find_protrusions=True,
            visualize=False)
        assert sr.image_diff(
            self.holes_true,
            results['holes'],
            visualize=False)
        assert sr.image_diff(
            self.islands_true,
            results['islands'],
            visualize=False)
        assert sr.image_diff(
            self.indents_true,
            results['indentations'],
            visualize=False)
        assert sr.image_diff(
            self.prots_true,
            results['protrusions'],
            visualize=False)

    def test_fill_image_noise(self):
        '''
        Test the helper method `fill_image`.
        '''
        filled = self.binarydetector._fill_image(self.image_noise)
        assert sr.image_diff(
            self.filled_image_noise_true,
            filled,
            visualize=False)

    def test_fill_image_nested(self):
        '''
        Test the helper method `fill_image` for an image with nested regions..
        '''
        filled = self.binarydetector._fill_image(self.image_nested)
        assert sr.image_diff(
            self.filled_image_nested_true,
            filled,
            visualize=False)
