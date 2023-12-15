import unittest
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

from process_util.DataProcessor import DataProcessor

class DataProcessTest(unittest.TestCase):
    __input_dir__='../data/test_data/outputT'
    dataProcessor = DataProcessor()
    def testOpenCV(self):
        dp = self.dataProcessor
        files = []
        for file in Path.glob(Path(self.__input_dir__), '**/*.mp4'):
            if file.is_file():
                files.append(file.as_posix())
        files.sort()
        for video in tqdm(files):
            dp.processVideoFile(video,
                                processed_data_root='../data/test_data/pr_data')




    def testffmpeg(self):
        pass




