import random

import cv2
import numpy as np
import torch
from pathlib import Path

from torch.utils.data import Dataset


class SyncDataset(Dataset):

    def __init__(self,data_dir,
                 syncnet_T:int = 5,
                 type:str = 'train',
                 **kwargs):
        """
            :param data_dir: 数据文件的根目录
            :param video_info: 所有视频文件的目录信息，一般放在train.txt文件中。
        """
        self.data_dir = data_dir
        self.syncnet_T = syncnet_T
        self.type = type
        self.img_size = kwargs['img_size']
        self.dirlist = self.__get_split_video_list()


    def __getitem__(self, index):
        """
        循环去一段视频和一段错误视频进行网络对抗，这里就是取两个不同视频的方法
        :param index: the index of item
        :return: image
        """
        #随机抽取一个视频的文件进行处理
        image_names = self.__get_imgs(index)
        audio_index = index

        window,g_window = self.__get_gan_window(image_names)
        while len(image_names) <= 3 * self.syncnet_T or window is None or g_window is None:
            idx = random.randint(0,len(self.dirlist) - 1)
            audio_index = idx
            image_names = self.__get_imgs(idx)
            window, g_window = self.__get_gan_window(image_names)

        #对window进行范围缩小到0-1之间的array的处理
        window = self.__narray_window(window)
        x = window.copy()
        window[:,:,window.shape[2]//2:] = 0.

        g_window = self.__narray_window(g_window)
        y = np.concatenate([window,g_window],axis=0)

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        #对音频进行mel图谱化，并进行对应。
        vid = self.dirlist[audio_index]
        wavfile = vid + '/audio.wav'


        return x,y

    def __len__(self):
        return len(self.dirlist)
    """
        下面的方法都是内部的方法，用于数据装载时，对数据的处理
    """
    def __get_split_video_list(self):
        load_file= self.data_dir+'/{}.txt'.format(self.type)
        dirlist=[]
        with open(load_file,'r') as f:
           for line in f:
               line = line.strip()
               dirlist.append(line)

        return dirlist

    def __get_imgs(self,index):
        vid = self.dirlist[index]
        img_names = []
        for img in Path().joinpath(self.data_dir,vid).glob('**/*.jpg'):
            img_names.append(img)
        return img_names

    #随机找出对抗的两张脸的图，一个是正确的，一个是错误的
    def __get_gan_window(self, image_names):
        img_name = random.choice(image_names)
        g_img_name = random.choice(image_names)
        while img_name==g_img_name:
            g_img_name = random.choice(image_names)

        window_fnames = self.__get_window(img_name)
        g_window_fnames = self.__get_window(g_img_name)
        if window_fnames is None or g_window_fnames is None:
            return None,None
        window = self.__read_window(window_fnames)
        g_window = self.__read_window(g_window_fnames)

        return window,g_window

    def __get_window(self, img_name):
        start_id = int(Path(img_name).stem)
        seek_id = start_id +self.syncnet_T
        vidPath = Path(img_name).parent
        window_frames = []
        for frame_id in range(start_id,seek_id):
            frame = str(vidPath) + '/{}.jpg'.format(frame_id)
            if not Path(frame).is_file():
                return None
            window_frames.append(frame)
        return window_frames

    def __read_window(self, window_fnames):
        window = []
        for f_name in window_fnames:
            img = cv2.imread(f_name)
            if img is None:
                return None
            try:
                img = cv2.resize(img,(self.img_size,self.img_size))
            except Exception as e:
                print('Resize the face image error: {}'.format(e))
                return None
            window.append(img)
        return window

    def __narray_window(self, window):
        #数组转换3xTxHxW
        wa = np.asarray(window) / 255.
        wa = np.transpose(wa,(3,0,1,2))

        return wa