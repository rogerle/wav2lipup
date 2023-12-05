from pathlib import Path

from moviepy.editor import *

from process_util.FaceDetector import FaceDetector


class SyncnetPScore():
    face_detector = FaceDetector()
    def __init__(self,pre_model,batch_size,tmp):
        self.pre_model=pre_model
        self.batch_size=batch_size
        self.tmp_dir=tmp
    def scoreVideo(self,v_file):
        video_file=v_file
        self.__extract_video(video_file)

    def __extract_video(self, video_file):
        videoC = VideoFileClip(video_file)
        v_end = int(videoC.duration)
        outputName = self.tmp_dir+'/'+Path(video_file).stem+'.mp4'
        clipVideo = videoC.subclip(0, v_end)
        clipVideo.write_videofile(outputName, fps=25)

        new_videoC=VideoFileClip(outputName)
        face_list={}
        for idx, frame in enumerate(new_videoC.iter_frames()):
            idx=idx +1
            face_result = self.face_detector.faceDetec(frame)
            scores = face_result['scores']
            boxes = face_result['boxes']
            if scores is None or len(scores) == 0:
                print('bad face video,drop it!')
                continue
            else:
                idx = scores.index(max(scores))
                box = boxes[idx]
                x1, y1, x2, y2 = box
                face = frame[int(y1):int(y2), int(x1):int(x2)]
                face_list['{}'.format(idx)] = face
                if np.size(face) == 0:
                    continue




