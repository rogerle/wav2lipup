import unittest
from process_util.SyncnetScore import SyncnetScore

class SyncnetScoreTest(unittest.TestCase):
    def testscore(self):
        syncnet_score=SyncnetScore('../data/test_data/pr_data',8,'../data/syncnet_checkpoint/sync_checkpoint_step000014000.pth')
        syncnet_score.score_video()


if __name__ == '__main__':
    unittest.main()
