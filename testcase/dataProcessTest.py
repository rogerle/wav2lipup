import unittest
from collections import Counter
from pathlib import Path

import cv2
from process_util.DataProcessor import DataProcessor

class DataProcessTest(unittest.TestCase):

    dataProcessor = DataProcessor()
    def testOpenCV(self):
        dp = self.dataProcessor
        dp.processVideoFile('../data/test_data/output/000001/000001/00050_00058.mp4',
                            device='gpu',
                            processed_data_root='../data/test_data/pr_data')
        face_path = Path('../data/test_data/pr_data/000001/000001_00050_00058')
        wavfile = Path('../data/test_data/pr_data/000001/000001_00050_00058/audio.wav')
        self.assertTrue(wavfile.exists())

        dp.processVideoFile('../data/test_data/output/000001/000001/00058_00063.mp4',
                            device='gpu',
                            processed_data_root='../data/test_data/pr_data')
        face_path = Path('../data/test_data/pr_data/000001/000001_00058_00063')
        wavfile = Path('../data/test_data/pr_data/000001/000001_00058_00063/audio.wav')
        self.assertTrue(wavfile.exists())


    def testffmpeg(self):
        pass




