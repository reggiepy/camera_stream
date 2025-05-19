import os

from websockets import ConnectionClosedOK

# 参见 https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import snappy
import cv2
import asyncio
import base64
import threading
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class CameraStreamer:
    def __init__(self, camera_id: int, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera {camera_id} opened with resolution {actual_width}x{actual_height} at {actual_fps} FPS")

        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def _update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                logger.warning(f"读取摄像头 {self.camera_id} 帧失败")
                asyncio.run(asyncio.sleep(0.05))

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            ret, jpeg = cv2.imencode(".jpg", self.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            # ret, jpeg = cv2.imencode(".jpg", self.frame)
            if not ret:
                logger.error("JPEG 编码失败")
                return None
            return jpeg.tobytes()

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()
        logger.info(f"Camera {self.camera_id} released")


@app.get("/", response_class=HTMLResponse)
async def index():
    return RedirectResponse("/static/index.html")


@app.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket, camera_id: int = 0, width: int = 640, height: int = 480,
                           fps: int = 30):
    await websocket.accept()
    try:
        streamer = CameraStreamer(camera_id, width, height, fps)
    except RuntimeError as e:
        await websocket.send_text(f"ERROR: {e}")
        await websocket.close()
        return

    try:
        delay = 1.0 / fps
        while True:
            buffer = streamer.get_frame()
            if buffer is not None:
                # frame = base64.b64encode(buffer).decode("utf-8")
                # await websocket.send_text(frame)
                compressed = snappy.compress(buffer)  # 压缩 jpeg 二进制
                await websocket.send_bytes(compressed)  # 发送二进制数据
            await asyncio.sleep(delay)

    except WebSocketDisconnect:
        logger.info(f"WebSocket 客户端主动断开 (camera_id={camera_id})")
    except ConnectionClosedOK:
        logger.info(f"WebSocket 正常关闭 (camera_id={camera_id})")
    except Exception as e:
        logger.exception("WebSocket 内部发生异常")
    finally:
        streamer.release()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8011)
