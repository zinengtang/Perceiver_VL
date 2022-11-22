import random
import time

import torch
import numpy as np
import cv2
import ffmpeg

from PIL import Image
from decord import VideoReader
from decord import cpu, gpu
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class RawVideoExtractorCV2():
    
    def __init__(self, centercrop=True, size=224, framerate=1, max_frames=8):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.max_frames = max_frames
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def video_to_tensor(self, video_file, preprocess, sample_fp=1, start_time=None, end_time=None):
        try:
            vr = VideoReader(video_file, ctx=cpu(), width=self.size, height=self.size)
            if random.random() < 0.5 and len(vr) >= self.max_frames:
                downsamlp_indices = random.sample(list(range(len(vr))), self.max_frames)                    
            else:
                downsamlp_indices = np.linspace(0, len(vr), self.max_frames, endpoint=False).astype(np.int)

            video = vr.get_batch(downsamlp_indices).asnumpy()/255.0
            video = torch.from_numpy(video).permute(0, 3, 1, 2)
            video = preprocess(video)
        except:
            video = preprocess(torch.zeros([self.max_frames, 3, self.size, self.size]))
        return video
    
    def get_video_data(self, video_path, start_time=None, end_time=None):
        return self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data
      
