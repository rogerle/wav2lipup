import unittest
from process_util.VideoExtractor import VideoExtractor

class MyTestCase(unittest.TestCase):
    def test_something(self):
        video_e = VideoExtractor('../data/temp',25)
        video_e.pipline_video('cctvf0000004/0001.mp4',data_root='../data/original_data')



