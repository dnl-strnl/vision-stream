from collections import deque
import cv2
import time

class DefaultStream:
    def __init__(self, device_id=1, frame_width=None, frame_height=None, fps_buffer_size=30):
        self.cap = cv2.VideoCapture(device_id)
        if frame_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        if frame_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.prop_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.true_fps = self.prop_fps

        self.fps_buffer = deque(maxlen=fps_buffer_size)
        self.last_frame_time = time.time()

    def __call__(self):
        ret, frame = self.cap.read()

        current_time = time.time()
        frame_time = current_time - self.last_frame_time

        if frame_time > 0:
            instantaneous_fps = 1 / frame_time
            self.fps_buffer.append(instantaneous_fps)
            self.true_fps = sum(self.fps_buffer) / len(self.fps_buffer)

        self.last_frame_time = current_time
        return frame

    def get_fps(self):
        return self.true_fps

    def __del__(self):
        self.cap.release()
