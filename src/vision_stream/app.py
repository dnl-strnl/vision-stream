from collections import deque
import cv2
from flask import Flask, Response, jsonify, render_template, send_from_directory
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import logging as log
from omegaconf import DictConfig
import os
from os.path import basename, exists, join
from pathlib import Path
import queue
import subprocess
import sys
import time
import threading

from vision_stream.models import make_image_payload, send_image_request

class StreamServer:
    def __init__(
        self,
        video_stream,
        model_wrapper = None,
        jpeg_quality = 100,
        frame_buffer_maxlen = 10,
        video_dir = os.getcwd(),
        timeline_fps = 60,
    ):
        self.video_stream = video_stream
        self.model_wrapper = model_wrapper
        self.jpeg_quality = jpeg_quality
        self.video_dir = video_dir
        # each frame is held for as long as it actually took the pipeline to
        # produce the next one, so playback preserves the measured per-frame
        # timing instead of a single rate sampled when recording started.
        self.timeline_fps = timeline_fps
        self.session_frames = 0
        self._tick_error = 0.0

        self.frame_queue = queue.Queue()

        self.recording_thread = None
        self.is_recording = False
        self.video_path = None
        self.session_start = None
        self.process = None
        self.history = deque(maxlen=frame_buffer_maxlen)

        self.app = Flask(__name__)
        self.app.config['RECORDINGS_FOLDER'] = video_dir
        self._setup_routes()

    def get_history(self):
        return list(self.history)

    def start_recording(self, frame_size):
        if self.is_recording:
            return False

        self.session_frames = 0
        self._tick_error = 0.0

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'stream-{timestamp}.mp4'
        self.video_path = os.path.join(self.video_dir, filename)

        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{frame_size[1]}x{frame_size[0]}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.timeline_fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p',
            '-f', 'mp4',
            self.video_path
        ]

        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            self.is_recording = True
            self.session_start = time.time()
            self.recording_thread = threading.Thread(
                target=self._write_frames,
                daemon=True
            )
            self.recording_thread.start()

            filename = os.path.basename(self.video_path)
            self.history.append({
                'action': 'started',
                'timestamp': timestamp,
                'path': filename,
                'time': time.strftime('%H:%M:%S')
            })
            return True

        except Exception as e:
            print(f"Failed to start recording: {e}")
            return False

    def stop_recording(self):
        if not self.is_recording:
            return False

        self.is_recording = False
        # session ends when the user stops, not after the flush.
        stop_time = time.time()

        while not self.frame_queue.empty():
            time.sleep(0.1)

        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)

        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()

        session_duration = stop_time - self.session_start
        filename = os.path.basename(self.video_path)
        mean_fps = self.session_frames / session_duration if session_duration > 0 else 0.0

        self.history.append({
            'action': 'stopped',
            'path': filename,
            'duration': f'{session_duration:.2f}',
            'frames': self.session_frames,
            'mean_fps': f'{mean_fps:.2f}',
            'time': time.strftime('%H:%M:%S')
        })

        return {
            'path': filename,
            'duration': f'{session_duration:.2f}',
            'frames': self.session_frames,
            'mean_fps': f'{mean_fps:.2f}',
        }

    def _hold(self, frame_bytes, seconds):
        # the video is fixed-rate, so repeat the frame to fill the time it was
        # on screen. Rounding error carries over so total length stays exact.
        exact = seconds * self.timeline_fps + self._tick_error
        ticks = max(1, round(exact))
        self._tick_error = exact - ticks
        for _ in range(ticks):
            self.process.stdin.write(frame_bytes)

    def _write_frames(self):
        pending = None  # (frame_bytes, wall_clock_time) of the last frame seen
        try:
            while self.is_recording or not self.frame_queue.empty():
                try:
                    frame, frame_time = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if pending is not None:
                    self._hold(pending[0], frame_time - pending[1])
                pending = (frame.tobytes(), frame_time)
            if pending is not None:
                self._hold(pending[0], 0.0)  # final frame: one tick
        except Exception as frame_write_exception:
            print(f"{frame_write_exception=}")

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/recordings/<path:filename>')
        def serve_recording(filename):
            try:
                safe_filename = basename(filename)
                full_path = join(self.video_dir, safe_filename)

                if not exists(full_path):
                    return "Video file not found.", 404

                content_type = 'video/mp4' if filename.endswith('.mp4') else 'video/webm'
                response = send_from_directory(
                    self.video_dir,
                    safe_filename,
                    mimetype=content_type,
                    as_attachment=False
                )
                response.headers.add('Access-Control-Allow-Origin', '*')
                response.headers.add('Accept-Ranges', 'bytes')
                return response

            except Exception as video_serve_exception:
                return f"{video_serve_exception=}", 500

        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/record', methods=['POST'])
        def toggle_recording():
            if self.is_recording:
                result = self.stop_recording()
                return jsonify({
                    'status': 'stopped',
                    'video_path': result['path'],
                    'duration': result['duration'],
                    'history': self.get_history()
                })
            else:
                frame = self.video_stream()

                if frame is None:
                    return jsonify({'error': 'No frame available.'}), 400

                success = self.start_recording(frame_size=frame.shape)

                return jsonify({
                    'status': 'recording' if success else 'error',
                    'history': self.get_history()
                })

    def _generate_frames(self, show_output=True):
        while True:
            frame = self.video_stream()
            if frame is None:
                continue

            if self.model_wrapper and show_output:
                frame = self.model_wrapper(rgb_image=frame)

            _, buffer = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )

            if self.is_recording:
                self.frame_queue.put((frame, time.time()))
                self.session_frames += 1

            frame_header = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            yield frame_header + buffer.tobytes() + b"\r\n"

    def run(self, host, port):
        self.app.run(host=host, port=port, debug=False, threaded=True)

@hydra.main(version_base=None, config_path="config", config_name="app")
def main(cfg: DictConfig):

    def model_wrapper(rgb_image, **model_arguments):
        json_data = make_image_payload(rgb_image=rgb_image, **model_arguments)
        model_output = send_image_request(cfg.model_address, json_data=json_data)
        log.info(model_output)
        wrapper = instantiate(cfg.model_wrapper)
        image_output = wrapper.plot(rgb_image, model_output)
        return image_output

    try:
        try:
            video_stream = instantiate(cfg.video_stream)
            log.info("video stream: ✅")
        except Exception as video_stream_exception:
            log.info(f"{video_stream_exception=}")
            video_stream.close()
            sys.exit(1)

        StreamServer(
            video_stream,
            model_wrapper=model_wrapper,
            jpeg_quality=cfg.jpeg_quality,
            video_dir=HydraConfig.get().runtime.output_dir,
        ).run(host=cfg.host, port=cfg.port)

    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
