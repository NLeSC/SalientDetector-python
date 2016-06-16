# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:10:40 2016

@author: dafne
"""
from __future__ import absolute_import
from .context import salientregions as sr
from .context import salientregions_binarydetector
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
        Load the image and features for both the noise image and the nested image.
        '''
        self.testdata_images_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'images/Binary/'))
        self.testdata_features_path = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'features/Binary/'))

        self.binary_noise = self.setUpImage('Binary_all_types_noise.png',
                                            'Binary_all_types_noise_filled.png',
                                            'Binary_all_types_noise_binregions.mat',
                                            'SE_all.mat',
                                            lam=50,
                                            area_factor=0.05,
                                            connectivity=4)

        self.binary_nested = self.setUpImage('Binary_nested.png',
                                             'Binary_nested_filled.png',
                                             'Binary_nested_binregions.mat',
                                             'SE_nested.mat',
                                             lam=80,
                                             area_factor=0.05,
                                             connectivity=4)

    def setUpImage(self, image_file, filled_image_file,
                   features_file, SE_file, lam, area_factor, connectivity):
        '''
        Set up a specific image: load the image, the filled image,
        the features, the SE and create a binary detector.
        '''
        image = cv2.imread(
            os.path.join(
                self.testdata_images_path,
                image_file), cv2.IMREAD_GRAYSCALE)
        filled_image = cv2.imread(
            os.path.join(
                self.testdata_images_path,
                filled_image_file), cv2.IMREAD_GRAYSCALE)
        holes_true, islands_true, indents_true, prots_true = sr.read_matfile(
            os.path.join(self.testdata_features_path, features_file), visualize=False)
        SE = sio.loadmat(
            os.path.join(
                self.testdata_features_path,
                SE_file))['SE_n']
        binarydetector = sr.BinaryDetector(
            SE=SE, lam=lam, area_factor=area_factor, connectivity=connectivity)

        return image, filled_image, holes_true, islands_true, indents_true, prots_true, binarydetector

    def test_holes(self):
        '''
        Test the method `detect` for only holes.
        '''
        for tup in [self.binary_noise, self.binary_nested]:
            image, filled_image, holes_true, islands_true, indents_true, prots_true, binarydetector = tup
            results = binarydetector.detect(
                image,
                find_holes=True,
                find_islands=False,
                find_indentations=False,
                find_protrusions=False,
                visualize=False)
            holes_my = results['holes']
            assert sr.image_diff(holes_true, holes_my, visualize=False)

    def test_islands(self):
        '''
        Test the method `detect` for only islands.
        '''
        for tup in [self.binary_noise, self.binary_nested]:
            image, filled_image, holes_true, islands_true, indents_true, prots_true, binarydetector = tup
            results = binarydetector.detect(
                image,
                find_holes=False,
                find_islands=True,
                find_indentations=False,
                find_protrusions=False,
                visualize=False)
            islands_my = results['islands']
            assert sr.image_diff(islands_true, islands_my, visualize=False)

    def test_protrusions(self):
        '''
        Test the method `detect` for only protrusions.
        '''
        for tup in [self.binary_noise, self.binary_nested]:
            image, filled_image, holes_true, islands_true, indents_true, prots_true, binarydetector = tup
            results = binarydetector.detect(
                image,
                find_holes=False,
                find_islands=False,
                find_indentations=False,
                find_protrusions=True,
                visualize=False)
            prots_my = results['protrusions']
            assert sr.image_diff(prots_true, prots_my, visualize=False)

    def test_indentations(self):
        '''
        Test the method `detect` for only indentations.
        '''
        for tup in [self.binary_noise, self.binary_nested]:
            image, filled_image, holes_true, islands_true, indents_true, prots_true, binarydetector = tup
            results = binarydetector.detect(
                image,
                find_holes=False,
                find_islands=False,
                find_indentations=True,
                find_protrusions=False,
                visualize=False)
            indents_my = results['indentations']
            assert sr.image_diff(indents_true, indents_my, visualize=False)

    def test_holesislands(self):
        '''
        Test the method `detect` for holes and islands.
        '''
        for tup in [self.binary_noise, self.binary_nested]:
            image, filled_image, holes_true, islands_true, indents_true, prots_true, binarydetector = tup
            results = binarydetector.detect(
                image,
                find_holes=True,
                find_islands=True,
                find_indentations=False,
                find_protrusions=False,
                visualize=False)
            holes_my = results['holes']
            islands_my = results['islands']
            assert sr.image_diff(holes_true, holes_my, visualize=False)
            assert sr.image_diff(islands_true, islands_my, visualize=False)

    def test_protsindents(self):
        '''
        Test the method `detect` for protrusions and indentations.
        '''
        for tup in [self.binary_noise, self.binary_nested]:
            image, filled_image, holes_true, islands_true, indents_true, prots_true, binarydetector = tup
            results = binarydetector.detect(
                image,
                find_holes=False,
                find_islands=False,
                find_indentations=True,
                find_protrusions=True,
                visualize=False)
            assert sr.image_diff(
                indents_true,
                results['indentations'],
                visualize=False)
            assert sr.image_diff(
                prots_true,
                results['protrusions'],
                visualize=False)

    def test_detect(self):
        '''
        Test the method `detect` for all regions.
        '''
        for tup in [self.binary_noise, self.binary_nested]:
            image, filled_image, holes_true, islands_true, indents_true, prots_true, binarydetector = tup
            results = binarydetector.detect(
                image,
                find_holes=True,
                find_islands=True,
                find_indentations=True,
                find_protrusions=True,
                visualize=False)
            assert sr.image_diff(
                holes_true,
                results['holes'],
                visualize=False)
            assert sr.image_diff(
                islands_true,
                results['islands'],
                visualize=False)
            assert sr.image_diff(
                indents_true,
                results['indentations'],
                visualize=False)
            assert sr.image_diff(
                prots_true,
                results['protrusions'],
                visualize=False)

    def test_fill_image(self):
        '''
        Test the helper method `fill_image`.
        '''
        for tup in [self.binary_noise, self.binary_nested]:
            image, filled_true, holes_true, islands_true, indents_true, prots_true, binarydetector = tup
            filled = salientregions_binarydetector._fill_image(image)
            assert sr.image_diff(
                filled_true,
                filled,
                visualize=False)
