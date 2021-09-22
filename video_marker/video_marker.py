import cv2
import logging
import numpy as np
import os

from . utils import CFEVideoConf, image_resize

logger = logging.getLogger(__name__)
_WINDOW_NAME = 'monitor'


class VideoMarker(object):
    VIDEO_RES = '720p'

    def __init__(self,
                 video_cap:str = "0",
                 resolution:str = VIDEO_RES,
                 save_path = 'output/output.avi',
                 frames_per_seconds = 50,
                 watermark_fpath = 'assets/logo.png',
                 pre_media = 'assets/intro.mp4',
                 post_media = 'assets/intro.mp4',
                 watermark_size = 50,
                 monitor = False, **kwargs
        ):
        video_source = int(video_cap) if video_cap.isdigit() else video_cap
        self.capture = cv2.VideoCapture(video_source)
        self.video_size = CFEVideoConf.STD_DIMENSIONS[resolution]

        self.watermark_fpath = watermark_fpath
        self.watermark_size = watermark_size
        self.pre_media = pre_media
        self.post_media = post_media
        self.monitor = monitor

        self.vconf = dict(
            res = resolution,
            filepath = save_path,
        )
        config = CFEVideoConf(self.capture, **self.vconf)
        self.video_writer = cv2.VideoWriter(save_path, config.video_type, frames_per_seconds, config.dims)

    def get_watermark(self):
        if not self.watermark_fpath:
            return
        logo = cv2.imread(self.watermark_fpath, cv2.IMREAD_UNCHANGED)
        if logo.shape[2] == 4:     # we have an alpha channel
          a1 = ~logo[:,:,3]        # extract and invert that alpha
          logo = cv2.add(cv2.merge([a1,a1,a1,a1]), logo)   # add up values (with clipping)
          logo = cv2.cvtColor(logo, cv2.COLOR_RGBA2RGB)    # strip alpha channel

        alpha = logo[:,:,1] # Channel 3
        result = np.dstack([logo, alpha]) # Add the alpha channel

        watermark = image_resize(result, height=self.watermark_size)
        watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
        return watermark

    def pre_post_video(self, media:str):
        if not media or not os.path.isfile(media):
            return

        intro = cv2.VideoCapture(media)
        while 1:
            ret, frame = intro.read()
            if ret:
                # not needed ...
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                # frame = cv2.resize(frame, VIDEO_SIZE)
                self.video_writer.write(frame)

                if self.monitor:
                    cv2.imshow(_WINDOW_NAME, frame)
                    if cv2.waitKey(20) & 0xFF == ord('q'):
                        break
            else:
                break
        intro.release()

    def record_video(self, watermark = None):
        if isinstance(watermark, np.ndarray):
            watermark_h, watermark_w, watermark_c = watermark.shape
            # replace overlay pixels with watermark pixel values
            h_range = range(0, watermark_h)
            w_range = range(0, watermark_w)
            _add_watermark = True
        else:
            _add_watermark = False

        while(True):
            # Capture frame-by-frame
            ret, frame = self.capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            frame_h, frame_w, frame_c = frame.shape
            # overlay with 4 channels BGR and Alpha
            overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')

            if _add_watermark:
                for i in h_range:
                    for j in w_range:
                        offset = 10
                        h_offset = frame_h - watermark_h - offset
                        w_offset = frame_w - watermark_w - offset
                        overlay[h_offset + i, w_offset+ j] = watermark[i,j]
                # add watermark here
                cv2.addWeighted(overlay, 0.25, frame, 1.0, 0, frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            resized = cv2.resize(frame, self.video_size)
            self.video_writer.write(resized)

            if self.monitor:
                # Display the resulting frame
                cv2.imshow(_WINDOW_NAME, resized)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

    def start(self):
        self.pre_post_video(media = self.pre_media)
        _watermark = self.get_watermark()
        self.record_video(watermark = _watermark)
        self.pre_post_video(media = self.post_media)

        # When everything done, release the capture
        self.capture.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
