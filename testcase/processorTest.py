import unittest
from process_util.preProcessor import PreProcessor
class TestProcessor(unittest.TestCase):

    preProcessor = PreProcessor('../data/test_data')

    def testAudioProcessBySilent(self):
        processor = self.preProcessor
        processor.audioProcessBySilent(audioType='mp4')

    def testAudioProcessByASR(self):
        processor =self.preProcessor
        processor.audioProcessByASR(audioType='wav')