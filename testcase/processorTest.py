import unittest
from collections import Counter
from pathlib import Path

from process_util.PreProcessor import PreProcessor


class TestProcessor(unittest.TestCase):
    preProcessor = PreProcessor()
    __input_dir__ = '../data/test_data/input'
    __output_dir__ = '../data/test_data/output'
    __outputT_dir__ = '../data/test_data/outputT'

    def testVideosPreProcessByASR(self):
        processor = self.preProcessor
        processor.videosPreProcessByASR(input_dir=self.__input_dir__,
                                        output_dir=self.__output_dir__,
                                        ext='mp4')

    def testVideoPreProcessByTime(self):
        Path(self.__outputT_dir__).mkdir(exist_ok=True)
        processor = self.preProcessor

        files = []
        for file in Path.glob(Path(self.__input_dir__), '**/*.mp4'):
            if file.is_file():
                files.append(file)
        files.sort()
        for video in files:
            v = processor.videosPreProcessByTime(video,
                                                 s_time=5,
                                                 input_dir=self.__input_dir__,
                                                 output_dir=self.__outputT_dir__
                                                 )
