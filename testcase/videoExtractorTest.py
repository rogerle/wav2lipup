import unittest
from process_util.VideoExtractor import VideoExtractor

class MyTestCase(unittest.TestCase):
    def test_something(self):
        video_e = VideoExtractor('../data/temp',25)
        video_e.pipline_video('cctvm0000003/0003.mp4',data_root='../data/original_data')



