import unittest
from pathlib import Path

from process_util.PreProcessor import PreProcessor
class TestProcessor(unittest.TestCase):

    preProcessor = PreProcessor()
    __input_dir__ = '../data/test_data/input'
    __output_dir__ = '../data/test_data/output'
    def testAudioProcessBySilent(self):
        processor = self.preProcessor
        processor.audioProcessBySilent(audioType='mp4')

    def testVideosPreProcessByASR(self):
        processor =self.preProcessor
        processor.videosPreProcessByASR(input_dir=self.__input_dir__,
                                    output_dir=self.__output_dir__,
                                    ext='mp4')

    def testVideoPreProcessByTime(self):
        processor = self.preProcessor

        processor.videosPreProcessByTime(input_dir=self.__input_dir__,
                                         output_dir=self.__output_dir__,
                                         s_time=5,
                                         ext='mp4')
        dirs = []
        files = []
        for ob in Path(self.__output_dir__).glob('**/*'):
            if ob.is_dir():
                dirs.append(ob)
            elif ob.is_file():
                files.append(ob)
            else:
                continue

        self.assertIn(Path('../data/test_data/output/000001'),dirs)
        self.assertIn(Path('../data/test_data/output/000001/000001'), dirs)
        self.assertIn(Path('../data/test_data/output/000001/000002'), dirs)
        self.assertIn(Path('../data/test_data/output/000002'), dirs)
        self.assertIn(Path('../data/test_data/output/000002/000001'), dirs)
        self.assertIn(Path('../data/test_data/output/000002/000002'), dirs)
        self.assertIn(Path('../data/test_data/output/000001/000001/00000_00005.mp4'),files)
        self.assertIn(Path('../data/test_data/output/000001/000002/00000_00005.mp4'), files)
        self.assertIn(Path('../data/test_data/output/000002/000001/00000_00005.mp4'),files)
        self.assertIn(Path('../data/test_data/output/000002/000002/00000_00005.mp4'), files)
        splitVideos=[]
        for f in Path('../data/test_data/output/000001/000001').glob('**/*.mp4'):
            splitVideos.append(f)
        self.assertEqual(len(splitVideos),16,msg='000001/000001.mp4 split to 16 videos')
        splitVideos.clear()
        for f in Path('../data/test_data/output/000001/000002').glob('**/*.mp4'):
            splitVideos.append(f)
        self.assertEqual(len(splitVideos), 13, msg='000001/000002.mp4 split to 13 videos')
