import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from inference_util.InferenceUtil import InferenceUtil
from models.FaceCreator import FaceCreator
from process_util.ParamsUtil import ParamsUtil

hp = ParamsUtil()


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(description="inference video")
    parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from',
                        required=True)
    parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use',
                        required=True)
    parser.add_argument('--audio', type=str,
                        help='Filepath of video/audio file to use as raw audio source', required=True)
    parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                        default='results/result_voice.mp4')
    parser.add_argument('--static', type=bool,
                        help='If True, then use only first video frame for inference', default=False)
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                        default=25., required=False)
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')

    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=64)
    parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=32)

    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')

    return parser.parse_args()


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = FaceCreator()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    args = parse_args()
    face_src = args.face
    fps = args.fps
    frame_names = []
    infer_util = InferenceUtil(args.fps, args.checkpoint_path)
    if not Path(face_src).exists():
        print('--face argument No such file or directory: {}'.format(face_src))
        extr_v = ''
    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        fps = args.fps
    else:
        extr_v = infer_util.src_video_process(face_src)
        frame_names = infer_util.load_img_names(extr_v)
        args.img_path = extr_v
        # full_frames = infer_util.get_frames(extr_v,frame_names)
        print('Number of frames for inference: {}'.format(len(frame_names)))
        """faces = infer_util.faces_detect(extr_v, args.face_det_batch_size)
        if faces is None:
            raise ValueError('No faces in video!Face not detected! Ensure the video contains a face in all the frames.')
        print('Number of frames and faces for inference: {}'.format(len(faces)))"""

    if not args.audio.endswith('.wav'):
        args.audio = infer_util.extract_audio(args.audio)

    mel = infer_util.get_mel(args.audio)
    print('The mel shape is {}'.format(mel.shape))

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    frame_names = frame_names[:len(mel_chunks)]
    batch_size = args.wav2lip_batch_size
    gen = infer_util.gen_data(frame_names, mel_chunks, args)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        if i == 0:
            model = load_model(args.checkpoint_path)
            print("Model loaded")

            frame_h, frame_w = frames[0].shape[:-1]
            out = cv2.VideoWriter(args.img_path+'/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.tensor(np.transpose(img_batch, (0, 3, 1, 2)), dtype=torch.float).to(device)
        mel_batch = torch.tensor(np.transpose(mel_batch, (0, 3, 1, 2)), dtype=torch.float).to(device)

        with torch.no_grad():
            pred = model(mel_batch,img_batch)

        pred = pred.cpu().numpy().transpose(0,2,3,1) * 255

        for p,f,c in zip(pred, frames,coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()
    command = 'ffmpeg -loglevel error -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, args.img_path+'/result.avi', args.outfile)
    output = subprocess.call(command, shell=True, stdout=None)

if __name__ == '__main__':
    main()
