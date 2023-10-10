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
        face_path = Path('../data/test_data/pr_data/000001/000001_00000_00006')
        wavfile = Path('../data/test_data/pr_data/000001/000001_00000_00006/audio.wav')
        self.assertTrue(wavfile.exists())

        dp.processVideoFile('../data/test_data/output/000001/000001/00006_00012.mp4',
                            device='gpu',
                            processed_data_root='../data/test_data/pr_data')
        face_path = Path('../data/test_data/pr_data/000001/000001_00006_00012')
        wavfile = Path('../data/test_data/pr_data/000001/000001_00006_00012/audio.wav')
        self.assertTrue(wavfile.exists())

        dp.processVideoFile('../data/test_data/output/000001/000001/00012_00017.mp4',
                            device='gpu',
                            processed_data_root='../data/test_data/pr_data')
        face_path = Path('../data/test_data/pr_data/000001/000001_00012_00017')
        wavfile = Path('../data/test_data/pr_data/000001/000001_00012_00017/audio.wav')
        self.assertTrue(wavfile.exists())

        dp.processVideoFile('../data/test_data/output/000001/000002/00000_00005.mp4',
                            device='gpu',
                            processed_data_root='../data/test_data/pr_data')
        face_path = Path('../data/test_data/pr_data/000001/000002_00000_00005')
        wavfile = Path('../data/test_data/pr_data/000001/000002_00000_00005/audio.wav')
        self.assertTrue(wavfile.exists())

        dp.processVideoFile('../data/test_data/output/000002/000001/00000_00005.mp4',
                            device='gpu',
                            processed_data_root='../data/test_data/pr_data')
        face_path2 = Path('../data/test_data/pr_data/000002/000001_00000_00005')
        wavfile2 = Path('../data/test_data/pr_data/000002/000001_00000_00005/audio.wav')
        self.assertTrue(wavfile2.exists())






