import mss
from screeninfo import get_monitors
import cv2
from time import time
import numpy as np
import os
import shutil


class ScreenRecorder:
    def __init__(self, base_dir):
        self.sct = mss.mss()

        self.base_dir = base_dir
        if os.path.exists(self.base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(self.base_dir)

        self.cur_img = 0

    def grab_screenshot(self, bbox=None):
        start_time = time()

        if not bbox:
            monitor = get_monitors()[0]
            bbox = [monitor.y, monitor.x, monitor.width, monitor.height]

        monitor = {'top': bbox[0], 'left': bbox[1], 'width': bbox[2], 'height': bbox[3],}
        img = cv2.cvtColor(np.array(self.sct.grab(monitor)), cv2.COLOR_BGRA2GRAY)

        processing_time = time() - start_time

        return img, processing_time

    def save_screenshot(self, img, size_factor=1.0, file_path=None):
        start_time = time()
        img = cv2.resize(img, None, fx=size_factor, fy=size_factor)

        if not file_path:
            file_path = self.base_dir + str(self.cur_img).zfill(5) + ".png"
        print(file_path)
        cv2.imwrite(file_path, img)

        return time() - start_time

if __name__ == "__main__":
    sr = ScreenRecorder("data/cat/")
    img, time1 = sr.grab_screenshot()
    time2 = sr.save_screenshot(img, size_factor=0.1)