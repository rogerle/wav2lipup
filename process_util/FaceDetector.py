import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import logging


class FaceDetector():
    logging.basicConfig(level=logging.ERROR)
    # 使用modelscope的mogofacedetector模型进行人脸检测
    model_id = 'damo/cv_resnet50_face-detection_retinaface'
    #model_id = 'damo/cv_manual_face-detection_tinymog'

    # 初始化人脸热别模型，利用这个模型来识别人脸
    def __init__(self):
        # 初始化模型，人脸检测可以做
        self.mog_face_detection_func = pipeline(Tasks.face_detection, self.model_id)

    """
       在图片上进行人脸识别，回传人脸识别的原始数据，包含人脸的位置以及五官位置信息
       其中src_img_path是图片的物理路径
    """

    def faceDetec(self, *args):
        raw_result = self.mog_face_detection_func(args[0])
        return raw_result

    def faceBatchDetection(self, frames):
        frames = frames.copy()
        faces = []
        for frame in frames:
            face_result = self.mog_face_detection_func(frame)
            scores = face_result['scores']
            boxes = face_result['boxes']
            if scores is None or len(scores) == 0:
                print('No face detected')
                faces.append([-1,-1,-1,-1])
            else:
                idx = scores.index(max(scores))
                box = boxes[idx]
                x1, y1, x2, y2 = box
                coords= [int(x1),int(y1),int(x2),int(y2)]
                faces.append(coords)
        return faces

