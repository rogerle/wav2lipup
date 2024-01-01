import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from basicsr import imwrite
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan.utils import GFPGANer
from realesrgan import RealESRGANer
from torch import nn
from tqdm import tqdm

from inference_util import init_parser, swap_regions
from inference_util.InferenceUtil import InferenceUtil
from models.FaceCreator import FaceCreator
from process_util.ParamsUtil import ParamsUtil

hp = ParamsUtil()
mel_step_size = 16
pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(description="inference video")
    parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from',
                        required=True)
    parser.add_argument('--gfpgan_checkpoint', type=str, help='Name of saved checkpoint to load weights from',
                        required=True)
    parser.add_argument('--esrgan_checkpoint', type=str, help='Name of saved checkpoint to load weights from',
                        required=True)
    parser.add_argument('--segmentation_path', type=str,
                        help='Name of saved checkpoint of segmentation network', required=True)
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
    parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=64)
    parser.add_argument('--resize_factor', default=1, type=int,
                        help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                             'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                             'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                             'Use if you get a flipped result, despite feeding a normal looking video')
    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')
    parser.add_argument('--no_segmentation', default=False, action='store_true',
                        help='Prevent using face segmentation')
    parser.add_argument(
        '-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument(
        '--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument(
        '--bg_tile',
        type=int,
        default=400,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')

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


def set_realesrgan(args):
    use_half = False
    if torch.cuda.is_available():  # set False in CPU/MPS mode
        no_half_gpu_list = ['1650', '1660']  # set False for GPUs that don't support f16
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True
    model_net = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )

    up_sampler = RealESRGANer(
        scale=2,
        model_path=args.esrgan_checkpoint,
        model=model_net,
        tile=args.bg_tile,
        tile_pad=10,
        pre_pad=0,
        half=use_half
    )

    if not torch.cuda.is_available():  # CPU
        import warnings
        warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                      'The unoptimized RealESRGAN is slow on CPU. '
                      'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                      category=RuntimeWarning)
    return up_sampler


def main():
    args = parse_args()
    face_src = args.face
    fps = args.fps
    infer_util = InferenceUtil(args.fps, args.checkpoint_path)
    full_frames = []
    if not Path(face_src).exists():
        print('--face argument No such file or directory: {}'.format(face_src))
        extr_v = ''
    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg'] and Path(args.face).is_file():
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        ex_dir = Path(args.face).parent.as_posix()
        tmp_name = Path(args.face).stem +"_25.mp4"
        ffmpeg_cmd = "ffmpeg -loglevel error -y -i {0} -qscale:v 2 -async 1 -r {1} {2}/{3}".format(args.face, args.fps, ex_dir,tmp_name)
        output = subprocess.call(ffmpeg_cmd, shell=True, stdout=None)
        args.face = ex_dir+'/'+tmp_name
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('-------Reading video frames-------')
        while True:
            success, frame = video_stream.read()
            if not success:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

        # extr_v = infer_util.src_video_process(face_src)
        # frame_names = infer_util.load_img_names(extr_v)
        # args.img_path = extr_v
        # full_frames = infer_util.get_frames(extr_v,frame_names)
        print('Number of frames for inference: {}'.format(len(full_frames)))
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

    full_frames = full_frames[:len(mel_chunks)]
    batch_size = args.wav2lip_batch_size
    gen = infer_util.gen_data(full_frames.copy(), mel_chunks, args)
    print("Model loading {}".format(args.checkpoint_path))
    model = load_model(args.checkpoint_path)
    print("Model loaded")
    print("Loading segementation network...")
    seg_net = init_parser(args.segmentation_path)
    print("Model loaded")

    img_path = Path(args.outfile).parent.as_posix()
    output_path = img_path + '/output'

    j = 0
    prog_bar = tqdm(enumerate(gen), total=int(np.ceil(float(len(mel_chunks)) / batch_size)), leave=False)
    for i, (img_batch, mel_batch, frames, coords) in prog_bar:
        if i == 0:

            arch = 'clean'
            channel_multiplier = 2

            frame_h, frame_w = frames[0].shape[:-1]
            out = cv2.VideoWriter(img_path + '/result.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_w, frame_h))
            if args.bg_upsampler == 'realesrgan':
                bg_upsampler = set_realesrgan(args)
            else:
                bg_upsampler = None

            gfpgan = GFPGANer(model_path=args.gfpgan_checkpoint, upscale=1,
                              arch=arch, channel_multiplier=channel_multiplier,
                              bg_upsampler=bg_upsampler, device=device)

        img_batch = torch.tensor(np.transpose(img_batch, (0, 3, 1, 2)), dtype=torch.float).to(device)
        mel_batch = torch.tensor(np.transpose(mel_batch, (0, 3, 1, 2)), dtype=torch.float).to(device)

        print("batch write message:", len(img_batch), len(frames), len(coords))
        # cuda_ids = [int(d_id) for d_id in os.environ.get('CUDA_VISIBLE_DEVICES').split(',')]
        with torch.no_grad():
            # pred = MyDataParallel(model(mel_batch, img_batch), device_ids=cuda_ids)
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255

        for p, f, c in zip(pred, frames, coords):
            j += 1
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            if not args.no_segmentation:
                p = swap_regions(f[y1:y2, x1:x2], p, seg_net)
                head_h, head_w, _ = p.shape
                f[y1:y2, x1:x2] = p
            else:
                head_h, head_w, _ = p.shape
                width_cut = int(head_w * 0.2)
                f[y1:y2, x1 + width_cut:x2 - width_cut] = p[:, width_cut:head_w - width_cut]

            cf, rf, ri = gfpgan.enhance(f,
                                        has_aligned=args.aligned,
                                        only_center_face=args.only_center_face,
                                        paste_back=True,
                                        weight=args.weight)

            hd_f = np.clip(ri, 0, 255).astype(np.uint8)

            imwrite(f, output_path + '/{}_src.jpg'.format(j))
            #imwrite(hd_f, output_path + '/{}.jpg'.format(j))
            out.write(hd_f)

    out.release()

    # process_hd_video(args.img_path + '/result.avi', args)
    command = 'ffmpeg -f image2 -i {}/{}_src.jpg -tag:v DIVX {}'.format(output_path, '%d', img_path + '/result_src.avi')
    output = subprocess.call(command, shell=True, stdout=None)
    outfile_2 = Path(args.outfile).parent.as_posix()+'/output_src.mp4'
    command = 'ffmpeg -loglevel error -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio,
                                                                                  img_path + '/result_src.avi',
                                                                                  outfile_2)
    output = subprocess.call(command, shell=True, stdout=None)

    command = 'ffmpeg -loglevel error -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio,
                                                                                  img_path + '/result.avi',
                                                                                  args.outfile)
    output = subprocess.call(command, shell=True, stdout=None)
    print('Finish processed {}'.format(args.outfile))


if __name__ == '__main__':
    main()
