import os

# 禁用 OpenCV 硬件加速（某些设备会崩溃）
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import snappy
import anyio
import asyncio
import logging
import uvicorn

from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from websockets.exceptions import ConnectionClosedOK

# 初始化日志
logger = logging.getLogger(__name__)

# FastAPI 应用及静态资源挂载
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# 摄像头管理类
class CameraStreamer:
    def __init__(self, camera_id: int, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.frame: Optional[bytes] = None
        self.lock = asyncio.Lock()
        self.subscriber_count = 0

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self._thread_stream: Optional[asyncio.Task] = None

    async def start(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        logger.info(f"Camera {self.camera_id} opened: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x"
                    f"{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {self.cap.get(cv2.CAP_PROP_FPS)} FPS")

        self.running = True
        self._thread_stream = asyncio.create_task(self._update_frame())

    async def _update_frame(self):
        delay = 1.0 / self.fps
        while self.running:
            ret, frame = await anyio.to_thread.run_sync(lambda: self.cap.read())
            if ret:
                async with self.lock:
                    self.frame = frame
            else:
                logger.warning(f"读取摄像头 {self.camera_id} 帧失败")
            await asyncio.sleep(delay)

    async def get_frame(self) -> Optional[bytes]:
        async with self.lock:
            if self.frame is None:
                return None
            ret, jpeg = cv2.imencode(".jpg", self.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            return jpeg.tobytes() if ret else None

    async def stop(self):
        self.running = False
        if self._thread_stream:
            await self._thread_stream
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f"Camera {self.camera_id} released")


# 全局摄像头实例缓存
camera_streamers: Dict[int, CameraStreamer] = {}
camera_streamers_lock = asyncio.Lock()


@app.get("/", response_class=HTMLResponse)
async def index():
    return RedirectResponse("/static/index.html")


@app.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket, camera_id: int = 0, width: int = 640, height: int = 480,
                           fps: int = 30):
    await websocket.accept()

    async with camera_streamers_lock:
        streamer = camera_streamers.get(camera_id)
        if not streamer:
            streamer = CameraStreamer(camera_id, width, height, fps)
            try:
                await streamer.start()
            except RuntimeError as e:
                await websocket.send_text(f"ERROR: {e}")
                await websocket.close()
                return
            camera_streamers[camera_id] = streamer
        streamer.subscriber_count += 1

    try:
        delay = 1.0 / fps
        while True:
            frame = await streamer.get_frame()
            if frame:
                await websocket.send_bytes(snappy.compress(frame))
            await asyncio.sleep(delay)

    except (WebSocketDisconnect, ConnectionClosedOK):
        logger.info(f"WebSocket 关闭 (camera_id={camera_id})")
    except Exception:
        logger.exception(f"WebSocket 错误 (camera_id={camera_id})")

    finally:
        async with camera_streamers_lock:
            streamer.subscriber_count -= 1
            if streamer.subscriber_count <= 0:
                await streamer.stop()
                camera_streamers.pop(camera_id, None)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    uvicorn.run(app, host="0.0.0.0", port=8011)
