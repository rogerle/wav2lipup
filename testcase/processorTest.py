import json
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
        jsonfile1 = Path('../data/test_data/input/000001/000001.json')
        jsonfile2 = Path('../data/test_data/input/000002/000001.json')
        self.assertTrue(jsonfile1.exists())
        self.assertTrue(jsonfile2.exists())
        outputPath1 = Path('../data/test_data/output/000001/000001')
        outputPath2 = Path('../data/test_data/output/000002/000001')
        self.assertTrue(outputPath1.exists())
        self.assertTrue(outputPath2.exists())
        path1f = (i.suffix for i in outputPath1.iterdir())
        path2f = (i.suffix for i in outputPath2.iterdir())
        num1 = Counter(path1f)['.mp4']
        num2 = Counter(path2f)['.mp4']
        self.assertEqual(14, num1)
        self.assertEqual(11, num2)

    def testVideoPreProcessByTime(self):
        Path(self.__outputT_dir__).mkdir(exist_ok=True)
        processor = self.preProcessor

        processor.videosPreProcessByTime(input_dir=self.__input_dir__,
                                         output_dir=self.__outputT_dir__,
                                         s_time=5,
                                         ext='mp4')
        outputPath1 = Path('../data/test_data/outputT/000001/000001')
        outputPath2 = Path('../data/test_data/outputT/000002/000001')
        self.assertTrue(outputPath1.exists())
        self.assertTrue(outputPath2.exists())
        path1f = (i.suffix for i in outputPath1.iterdir())
        path2f = (i.suffix for i in outputPath2.iterdir())
        num1 = Counter(path1f)['.mp4']
        num2 = Counter(path2f)['.mp4']
        self.assertEqual(16, num1)
        self.assertEqual(13, num2)
