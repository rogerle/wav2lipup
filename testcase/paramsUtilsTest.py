import unittest

from process_util.ParamsUtil import ParamsUtil


class TestParamsUtils(unittest.TestCase):

    def testParams(self):
        paramsU = ParamsUtil()

        self.assertEqual(80,paramsU.num_mels)
        self.assertEqual(0.9, paramsU.resacling_max)
        self.assertEqual('None', paramsU.frame_shift_ms)
        self.assertEqual(-100, paramsU.min_level_db)
        self.assertEqual(20, paramsU.ref_level_db)
        self.assertEqual(288, paramsU.img_size)
