import unittest
from collections import Counter
from pathlib import Path

import cv2
from process_util.DataProcessor import DataProcessor

class DataProcessTest(unittest.TestCase):

    dataProcessor = DataProcessor()
    def testOpenCV(self):
        dp = self.dataProcessor
        dp.processVideoFile('../data/test_data/output/000001/000001/00000_00006.mp4',
                            device='gpu',
                            processed_data_root='../data/test_data/pr_data')
        face_path = Path('../data/test_data/pr_data/000001/000001/00000_00006')
        wavfile = Path('../data/test_data/pr_data/000001/000001/00000_00006/audio.wav')
        path1f = (i.suffix for i in face_path.iterdir())
        facefiles= Counter(path1f)['.jpg']
        self.assertEqual(150,facefiles)
        self.assertTrue(wavfile.exists())






