import unittest
from process_util.SyncnetScore import SyncnetScore

class SyncnetScoreTest(unittest.TestCase):
    def testscore(self):
        syncnet_score=SyncnetScore('H:/wav2lip_data/data/processed_data',8,'../data/syncnet_checkpoint/sync_checkpoint_step000028000.pth')
        syncnet_score.score_video()


if __name__ == '__main__':
    unittest.main()
