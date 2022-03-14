import torch
import numpy as np
from PIL import Image
import random
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import cv2
import time
import ffmpeg
from decord import VideoReader
from decord import cpu, gpu

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
      
#         return Compose([
#             Resize(n_px, interpolation=Image.BICUBIC),
#             CenterCrop(n_px),
#             lambda image: image.convert("RGB"),
#             ToTensor(),
#             Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#             Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#         ])

#     def _get_video_dim(self, video_path):
#         probe = ffmpeg.probe(video_path)
#         video_stream = next((stream for stream in probe['streams']
#                              if stream['codec_type'] == 'video'), None)
#         width = int(video_stream['width'])
#         height = int(video_stream['height'])
        
#         return height, width
    
#     def _get_output_dim(self, h, w):
#         if isinstance(self.size, tuple) and len(self.size) == 2:
#             return self.size
#         elif h >= w:
#             return int(h * self.size / w), self.size
#         else:
#             return self.size, int(w * self.size / h)    
#     def video_to_tensor(self, video_file, preprocess, sample_fp=1, start_time=None, end_time=None):

# #         start=time.time()
#         try:
#             h, w = self._get_video_dim(video_file)
#             height, width = self._get_output_dim(h, w)
#             cmd = (
#                 ffmpeg
#                 .input(video_file)
#                 .filter('fps', fps=self.framerate)
#                 .filter('scale', height, width)
#             )

#             if self.centercrop:
#                 x = int((width - self.size) / 2.0)
#                 y = int((height - self.size) / 2.0)
#                 cmd = cmd.crop(x, y, self.size, self.size)

#             out, _ = (
#                 cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=self.max_frames)
#                 .run(capture_stdout=True, capture_stderr=True, quiet=True)
#             )

#             video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
#             if video.shape[0] > self.max_frames:
#                 if random.random() < 0.5 and len(video) >= self.max_frames:
#                     downsamlp_indices = random.sample(list(range(len(video))), self.max_frames)                    
#                 else:
#                     downsamlp_indices = np.linspace(0, len(video), self.max_frames, endpoint=False).astype(np.int)
#                 video = video[downsamlp_indices]

#             video = torch.from_numpy(video.astype('float32')).permute(0, 3, 1, 2)/255.0
#             video = preprocess(video)
#     #         print(time.time()-start)
#         except:
#             video = preprocess(torch.zeros([self.max_frames, 3, self.size, self.size]))
#         return video    
    
    
#     def video_to_tensor(self, video_file, preprocess, sample_fp=1, start_time=None, end_time=None):
#         if start_time is not None or end_time is not None:
#             assert isinstance(start_time, int) and isinstance(end_time, int) \
#                    and start_time > -1 and end_time > start_time
#         assert sample_fp > -1
        
#         try:
#             # Samples a frame sample_fp X frames.
#             cap = cv2.VideoCapture(video_file)

#             frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             fps = int(cap.get(cv2.CAP_PROP_FPS))        

#             total_duration = (frameCount + fps - 1) // fps
#             start_sec, end_sec = 0, total_duration

#             if start_time is not None:
#                 start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

#             interval = 1
#             if sample_fp > 0:
#                 interval = fps // sample_fp
#             else:
#                 sample_fp = fps
#             if interval == 0: interval = 1

#             inds = [ind for ind in np.arange(0, fps, interval)][:sample_fp]
#             assert len(inds) >= sample_fp
#             inds = inds[:sample_fp]

#             ret = True
#             images, included = [], []

#             for sec in np.arange(start_sec, end_sec + 1):
#                 if not ret: break
#                 sec_base = int(sec * fps)
#                 for ind in inds:
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
#                     ret, frame = cap.read()
#                     if not ret: break
#                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

#             cap.release()
#             if len(images) > 0:
#                 video_data = torch.tensor(np.stack(images))
#                 if video_data.size(0) > self.max_frames:
#                     downsamlp_indices = np.linspace(0, len(video_data), self.max_frames, endpoint=False).astype(np.int)
#                     video_data = video_data[downsamlp_indices]
#                     video_data = video_data[:self.max_frames]
#                 if video_data.size(0) < 1:
#                     video_data = torch.zeros([1, 3, self.size, self.size])+1e-8
#             else:
#                 video_data = torch.zeros([1, 3, self.size, self.size])+1e-8
        
#         except:
#             video_data = torch.zeros([1, 3, self.size, self.size])+1e-8
#         return video_data    

# An ordinary video frame extractor based CV2
# RawVideoExtractor = RawVideoExtractorCV2