from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2

class Logger():
    def __init__(self, log_dir, start=0) -> None:
        self.log_dir = log_dir

        self.global_step = start
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def add_scalars(self, scalars_dict):
        for key, value in scalars_dict.items():
            self.writer.add_scalar(key, value, self.global_step)
        self.writer.flush()
    
    def step(self):
        self.global_step += 1
    
    def log_video(self, img_arr, fps=30):
        img_arr = np.array(img_arr)
        video_path = self.log_dir + f"video_{self.global_step}.mp4"
        h,w = img_arr.shape[1:3]
        print(f"output video path: {video_path}")
        out = cv2.VideoWriter(filename=video_path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=(w,h))
        for t in range(img_arr.shape[0]):
            out.write(img_arr[t])
        out.release()