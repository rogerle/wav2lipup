import unittest
from collections import Counter
from pathlib import Path

import cv2
from process_util.DataProcessor import DataProcessor

class DataProcessTest(unittest.TestCase):

    dataProcessor = DataProcessor()
    def testOpenCV(self):
        dp = self.dataProcessor
        dp.processVideoFile('../data/test_data/output/test/00021_00022.mp4',
                            device='gpu',
                            processed_data_root='../data/test_data/pr_data')
        #face_path = Path('../data/test_data/pr_data/000001/000001_00056_00058')
        #wavfile = Path('../data/test_data/pr_data/000001/000001_00056_00058/audio.wav')
        #self.assertTrue(wavfile.exists())




    def testffmpeg(self):
        pass




