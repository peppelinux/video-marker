import cv2
import logging
import numpy as np
import os
import signal
import time

from .exceptions import MissingFrame
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
                 watermark_inverted = False,
                 slow_down_by = 0,
                 exception_grace_period = 0.2,
                 max_recording_exceptions = 4,
                 flip = 0,
                 monitor = False, **kwargs
        ):
        self.video_source = int(video_cap) if video_cap.isdigit() else video_cap
        self.video_size = CFEVideoConf.STD_DIMENSIONS[resolution]

        self.slow_down_by = slow_down_by

        self.watermark_fpath = watermark_fpath
        self.watermark_size = watermark_size
        self.watermark_inverted = watermark_inverted
        self.pre_media = pre_media
        self.post_media = post_media
        self.monitor = monitor

        self.save_path = save_path
        self.frames_per_seconds = frames_per_seconds

        self.video_writer_config_params = dict(
            res = resolution,
            filepath = save_path
        )

        self.state = 0

        self.exception_grace_period = exception_grace_period
        self.max_recording_exceptions = max_recording_exceptions
        self.exceptions_counter = 0
        self.flip = flip

        self._load_video_source()

    def release_device(self):
        self.capture.release()
        self.video_writer.release()

    def _load_video_source(self):
        """
        Load and reload a video source runtime
        """
        if hasattr(self, 'capture'):
            del(self.capture)
        if hasattr(self, 'video_writer_config'):
            del(self.video_writer_config)
            del(self.video_writer)

        # do not overwrite existing recordings
        fpath = self.video_writer_config_params['filepath']
        while os.path.exists(fpath):
            logger.warning(
                f"Found an existing video in {fpath}, trying to not overwrite it"
            )
            exploded_path = list(fpath.rpartition('/'))
            exploded_path.insert(-1, '_')
            fpath = ''.join(exploded_path)
        self.save_path = fpath

        logger.info(f"Writing video to {fpath}")
        self.video_writer_config_params['filepath'] = fpath

        self.capture = cv2.VideoCapture(self.video_source)
        self.video_writer_config = CFEVideoConf(
            self.capture, **self.video_writer_config_params
        )

        self.video_writer = cv2.VideoWriter(
            self.save_path, self.video_writer_config.video_type,
            self.frames_per_seconds, self.video_writer_config.dims,
        )

    @staticmethod
    def _assert_frame(frame):
        if not hasattr(frame, 'any') or not frame.any():
            return False
        else:
            return True

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

        if self.watermark_inverted:
            # invert the array
            result = np.invert(result)

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

    def _assert_videodevice(self):
        # frame sample
        ret, frame = self.capture.read()
        if not self._assert_frame(frame):
            logger.critical("Missing frame in .capture.read() ... exit")
            raise MissingFrame("OpenCV frame is None, check video device")


    def record_video(self, watermark = None):
        if isinstance(watermark, np.ndarray):
            watermark_h, watermark_w, watermark_c = watermark.shape
            # replace overlay pixels with watermark pixel values
            h_range = range(0, watermark_h)
            w_range = range(0, watermark_w)
            _add_watermark = True
        else:
            _add_watermark = False

        slow_down_by = range(self.slow_down_by - 1)

        # optimizations
        cvtColor = cv2.cvtColor
        zeros = np.zeros
        addWeighted = cv2.addWeighted
        cv2_resize = cv2.resize
        COLOR_BGRA2BGR = cv2.COLOR_BGRA2BGR
        COLOR_BGR2BGRA = cv2.COLOR_BGR2BGRA

        ret, frame = self.capture.read()
        frame = cvtColor(frame, cv2.COLOR_BGR2BGRA)

        frame_h, frame_w, frame_c = frame.shape
        # overlay with 4 channels BGR and Alpha
        overlay = zeros((frame_h, frame_w, 4), dtype='uint8')

        if _add_watermark:
            for i in h_range:
                for j in w_range:
                    offset = 10
                    h_offset = frame_h - watermark_h - offset
                    w_offset = frame_w - watermark_w - offset
                    overlay[h_offset + i, w_offset+ j] = watermark[i,j]

        while (self.state >= 0):
            if self.exceptions_counter >=self.max_recording_exceptions:
                excp_msg = (
                    "Exception during video recording with "
                    f"exceptions_counter={self.exceptions_counter} "
                    f"and exception_grace_period={self.exception_grace_period}"
                )
                logger.critical(excp_msg)
                raise MissingFrame(
                    excp_msg
                )

            # Capture frame-by-frame
            ret, frame = self.capture.read()
            if not self._assert_frame(frame):
                self.exceptions_counter += 1
                time.sleep(self.exception_grace_period)
                self._load_video_source()
                continue
            else:
                self.exceptions_counter = 0

            frame = cvtColor(frame, COLOR_BGR2BGRA)

            if _add_watermark:
                addWeighted(overlay, 0.25, frame, 1.0, 0, frame)

            frame = cvtColor(frame, COLOR_BGRA2BGR)
            if self.flip:
                frame = cv2.flip(frame, self.flip)
            resized = cv2_resize(frame, self.video_size)

            self.video_writer.write(resized)
            for n in slow_down_by:
                self.video_writer.write(resized)

            if self.monitor:
                # Display the resulting frame
                cv2.imshow(_WINDOW_NAME, resized)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

    def repr_params(self):
        return '\n'.join([f"{k}: {v}" for k,v in self.__dict__.items()])

    def signal_handler(self, sig, frame):
        logger.info('INT Signal received, shutting down')
        self.state = -1
        # sys.exit(0)

    def start(self):
        self.state = 1
        signal.signal(signal.SIGTERM, self.signal_handler)
        logger.info(f"Starting video recording with {self.repr_params()}")
        self._assert_videodevice()
        self.pre_post_video(media = self.pre_media)
        _watermark = self.get_watermark()
        self.record_video(watermark = _watermark)
        self.pre_post_video(media = self.post_media)

        # When everything done, release the capture
        self.release_device()
        cv2.destroyAllWindows()
