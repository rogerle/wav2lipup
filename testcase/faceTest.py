import unittest

import cv2
from PIL import Image

from process_util.FaceDetector import FaceDetector


class TestFaceDetector(unittest.TestCase):
    faceDetector = FaceDetector()

    def test_imgPath(self):
        imgPath = '../data/test_data/test.jpg'

        result = self.faceDetector.faceDetec(imgPath)
        value = {'scores': [0.9578035473823547, 0.957781195640564, 0.9503945708274841, 0.9378407001495361,
                            0.8208725452423096],
                 'boxes': [[490.98687744140625, 81.58421325683594, 559.7579345703125, 179.40164184570312],
                           [205.03994750976562, 62.21476364135742, 273.06390380859375, 151.08944702148438],
                           [803.9385375976562, 157.96820068359375, 865.4893188476562, 235.7114715576172],
                           [611.2905883789062, 172.23983764648438, 661.6884155273438, 234.4497833251953],
                           [329.2081604003906, 179.24513244628906, 380.1270751953125, 244.6616973876953]],
                 'keypoints': None}
        self.assertEqual(result, value, 'testfiles')

    def test_img(self):
        img = cv2.imread('../data/test_data/input/test.jpg')
        result = self.faceDetector.faceBatchDetection([img])
        face,coords = result[0]
        print(coords)


if __name__ == '__main__':
    unittest.main()
