import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

# ===================== PARÁMETROS DE LA CÁMARA =====================
DEV = "/dev/video0"
WIDTH = 1344
HEIGHT = 376
FPS = 15

# Abrimos la cámara una sola vez
cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

print("📷 Cámara abierta:", cap.isOpened())

app = FastAPI()


# ===================== GENERADOR DE FRAMES =====================
def generate_frames():
    """
    Genera frames JPEG en un stream tipo MJPEG.
    """
    while True:
        if not cap.isOpened():
            # Intentamos reabrir por si se perdió
            cap.open(DEV)

        ret, frame = cap.read()
        if not ret or frame is None:
            continue  # salteamos y seguimos intentando

        # Codificamos el frame como JPEG
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        # Formato MJPEG: multipart/x-mixed-replace
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )


# ===================== RUTA PRINCIPAL (HTML SIMPLE) =====================
@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Página HTML mínima que muestra el video.
    """
    return """
    <html>
      <head>
        <title>Stream ZED</title>
        <style>
          body { background: #111; color: #eee; text-align: center; font-family: sans-serif; }
          img { margin-top: 20px; border: 2px solid #555; }
        </style>
      </head>
      <body>
        <h1>Stream ZED UVC (/dev/video0)</h1>
        <p>Resolución: 1344x376 @ 15 fps (SBS)</p>
        <img src="/video" />
      </body>
    </html>
    """


# ===================== RUTA DEL VIDEO =====================
@app.get("/video")
async def video_feed():
    """
    Endpoint que devuelve el stream de video en MJPEG.
    """
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
