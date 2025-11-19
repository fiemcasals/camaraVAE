import os, time, yaml
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse

# ======================= ENV =======================
def getenvf(k, d):
    try:
        return float(os.getenv(k, str(d)))
    except:
        return float(d)

LEFT_DEV      = os.getenv("LEFT_DEV",  "/dev/video0")
RIGHT_DEV     = os.getenv("RIGHT_DEV", "/dev/video1")  # poné "" si es SBS puro
W             = int(os.getenv("WIDTH",  "1344"))       # tu ZED UVC: 1344x376 @15
H             = int(os.getenv("HEIGHT", "376"))
FPS           = int(os.getenv("FPS",    "15"))
THRESHOLD_M   = getenvf("THRESHOLD_M", 1.5)
ENABLE_PREVIEW= os.getenv("ENABLE_PREVIEW", "true").lower() == "true"
CALIB_YAML    = os.getenv("CALIB_YAML", "").strip()

# Pinhole (si no hay YAML)
BASELINE_M    = getenvf("BASELINE_M", 0.06)
FOCAL_PX      = getenvf("FOCAL_PX",   700)

# Banda de interés para /detect (en metros)
BAND_X_HALF_M = getenvf("BAND_X_HALF_M", 0.70)   # ±0.70m a los costados
BAND_Y_DOWN_M = getenvf("BAND_Y_DOWN_M", 0.70)   # 0.70m hacia abajo
BAND_Y_UP_M   = getenvf("BAND_Y_UP_M",   0.40)   # 0.40m hacia arriba
Z_MIN_M       = getenvf("Z_MIN_M", 0.20)
Z_MAX_M       = getenvf("Z_MAX_M", 10.0)

# Orientación/orden SBS
SWAP_HALVES   = os.getenv("SWAP_HALVES", "false").lower() == "true"
HFLIP_LEFT    = os.getenv("HFLIP_LEFT",  "false").lower() == "true"
HFLIP_RIGHT   = os.getenv("HFLIP_RIGHT", "false").lower() == "true"
VFLIP_LEFT    = os.getenv("VFLIP_LEFT",  "false").lower() == "true"
VFLIP_RIGHT   = os.getenv("VFLIP_RIGHT", "false").lower() == "true"

# Rectificación sin calibración (ayuda mucho si no hay YAML)
UNCAL_RECTIFY  = os.getenv("UNCAL_RECTIFY", "true").lower() == "true"
UNCAL_FEATURES = int(os.getenv("UNCAL_FEATURES", "1200"))

# >>>>>>>>>> Alineación manual fina (NUEVO) <<<<<<<<<<
MANUAL_ALIGN = os.getenv("MANUAL_ALIGN", "true").lower() == "true"
ROT_L_DEG = getenvf("ROT_L_DEG", 0.0)     # ej: -3.5
ROT_R_DEG = getenvf("ROT_R_DEG", 0.0)
SHIFT_L_X = int(float(os.getenv("SHIFT_L_X", "0")))   # px, der +
SHIFT_L_Y = int(float(os.getenv("SHIFT_L_Y", "0")))   # px, abajo +
SHIFT_R_X = int(float(os.getenv("SHIFT_R_X", "0")))
SHIFT_R_Y = int(float(os.getenv("SHIFT_R_Y", "0")))
SCALE_L   = getenvf("SCALE_L", 1.0)       # microzoom (1.0 = sin cambio)
SCALE_R   = getenvf("SCALE_R", 1.0)

app = FastAPI(title="Stereo CPU Obstacle Service")

# ================== CÁMARAS ==================
def open_cam(dev, w, h, fps=15):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Tu dispositivo reporta sólo YUYV en 1344x376@15
    fourcc = cv2.VideoWriter_fourcc(*'YUYV')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    return cap

capL = open_cam(LEFT_DEV, W, H, FPS)
capR = open_cam(RIGHT_DEV, W, H, FPS) if RIGHT_DEV else cv2.VideoCapture()  # vacío si SBS puro

use_sbs = False
if not capL.isOpened() and capR.isOpened():
    capL, capR = capR, capL

if not capL.isOpened():
    raise RuntimeError("No se pudo abrir cámara principal (LEFT_DEV).")

if not capR.isOpened():
    use_sbs = True  # un solo /dev/video con imagen lado-a-lado

# ========== Calibración/Rectificación (YAML) ==========
rectify = False
R1 = R2 = P1 = P2 = Q = None
mapLx = mapLy = mapRx = mapRy = None

if CALIB_YAML and os.path.exists(CALIB_YAML):
    with open(CALIB_YAML, "r") as f:
        data = yaml.safe_load(f)
    K1 = np.array(data["K1"]); D1 = np.array(data["D1"]).ravel()
    K2 = np.array(data["K2"]); D2 = np.array(data["D2"]).ravel()
    R  = np.array(data["R"]);  T  = np.array(data["T"]).ravel()
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, (W, H), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    mapLx, mapLy = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)
    rectify = True

# ========== Rectificación SIN calibración ==========
def _find_correspondences(imgL, imgR, max_pts=1200):
    ptsL = cv2.goodFeaturesToTrack(imgL, maxCorners=max_pts, qualityLevel=0.01, minDistance=7)
    if ptsL is None:
        return None, None
    ptsR, st, _ = cv2.calcOpticalFlowPyrLK(
        imgL, imgR, ptsL, None,
        winSize=(21,21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    if ptsR is None or st is None:
        return None, None
    st = st.reshape(-1)
    pL = ptsL.reshape(-1,2)[st==1]
    pR = ptsR.reshape(-1,2)[st==1]
    return pL, pR

def uncalibrated_rectify(grayL, grayR, max_pts=1200):
    pL, pR = _find_correspondences(grayL, grayR, max_pts=max_pts)
    if pL is None or len(pL) < 50:
        return grayL, grayR
    F, mask = cv2.findFundamentalMat(pL, pR, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None:
        return grayL, grayR
    h, w = grayL.shape
    ok, HL, HR = cv2.stereoRectifyUncalibrated(pL[mask.ravel()==1], pR[mask.ravel()==1], F, imgSize=(w,h))
    if not ok:
        return grayL, grayR
    rectL = cv2.warpPerspective(grayL, HL, (w,h))
    rectR = cv2.warpPerspective(grayR, HR, (w,h))
    return rectL, rectR

# ========== Alineación Manual (rotación/shift/escala) ==========
def _affine_rotate_shift(img, deg, sx, sy, scale=1.0):
    if (abs(deg) < 1e-3) and (sx == 0) and (sy == 0) and (abs(scale-1.0) < 1e-3):
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, scale)
    M[0,2] += sx
    M[1,2] += sy
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

def apply_manual_alignment(left, right):
    if not MANUAL_ALIGN:
        return left, right
    left  = _affine_rotate_shift(left,  ROT_L_DEG, SHIFT_L_X, SHIFT_L_Y, SCALE_L)
    right = _affine_rotate_shift(right, ROT_R_DEG, SHIFT_R_X, SHIFT_R_Y, SCALE_R)
    return left, right

# ========== SGBM (AJUSTES QUE PEDISTE) ==========
bm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,   # << antes 16*6
    blockSize=7,           # << antes 5
    P1=8*3*7**2,
    P2=32*3*7**2,
    speckleWindowSize=80,
    speckleRange=3,
    uniquenessRatio=8,
    disp12MaxDiff=1
)

# ================== FUNCIONES ==================
def grab_pair():
    okL, frameL = capL.read()
    if not okL:
        raise RuntimeError("No se pudo capturar de la cámara principal")

    if use_sbs:
        h, w = frameL.shape[:2]
        mid = w // 2
        l = frameL[:, :mid]
        r = frameL[:, mid:]

        # Corrige orden/orientación de mitades
        left, right = (r, l) if SWAP_HALVES else (l, r)
        if HFLIP_LEFT:  left  = cv2.flip(left,  1)
        if VFLIP_LEFT:  left  = cv2.flip(left,  0)
        if HFLIP_RIGHT: right = cv2.flip(right, 1)
        if VFLIP_RIGHT: right = cv2.flip(right, 0)

        # >>> Alineación manual previa (roll/shift/escala)
        left, right = apply_manual_alignment(left, right)

        grayL = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Rectificación (YAML o no calibrada)
        if rectify:
            grayL = cv2.remap(grayL, mapLx, mapLy, cv2.INTER_LINEAR)
            grayR = cv2.remap(grayR, mapRx, mapRy, cv2.INTER_LINEAR)
        elif UNCAL_RECTIFY:
            grayL, grayR = uncalibrated_rectify(grayL, grayR, UNCAL_FEATURES)

        return left, right, grayL, grayR

    # Modo dos /dev/video
    okR, frameR = capR.read()
    if not okR:
        raise RuntimeError("No se pudo capturar de la cámara secundaria")

    # flips si hiciera falta
    left, right = frameL, frameR
    if HFLIP_LEFT:  left  = cv2.flip(left,  1)
    if VFLIP_LEFT:  left  = cv2.flip(left,  0)
    if HFLIP_RIGHT: right = cv2.flip(right, 1)
    if VFLIP_RIGHT: right = cv2.flip(right, 0)

    # >>> Alineación manual previa
    left, right = apply_manual_alignment(left, right)

    grayL = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    if rectify:
        grayL = cv2.remap(grayL, mapLx, mapLy, cv2.INTER_LINEAR)
        grayR = cv2.remap(grayR, mapRx, mapRy, cv2.INTER_LINEAR)
    elif UNCAL_RECTIFY:
        grayL, grayR = uncalibrated_rectify(grayL, grayR, UNCAL_FEATURES)
    return left, right, grayL, grayR

def disparity_to_points3d_full(disp):
    disp_f = disp.astype(np.float32) / 16.0
    disp_f[disp_f <= 0.5] = np.nan
    if Q is not None:
        pts = cv2.reprojectImageTo3D(disp_f, Q)
        X = pts[:, :, 0]
        Y = pts[:, :, 1]
        Z = pts[:, :, 2]
        return X, Y, Z
    Z = (FOCAL_PX * BASELINE_M) / disp_f
    h, w = Z.shape
    cx = w * 0.5
    cy = h * 0.5
    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    X = (xs - cx) * (Z / FOCAL_PX)
    Y = (ys - cy) * (Z / FOCAL_PX)
    return X, Y, Z

def band_mask_xyz(X, Y, Z):
    mask = (np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z))
    mask &= (np.abs(X) <= BAND_X_HALF_M)
    mask &= (Y >= -BAND_Y_DOWN_M) & (Y <= BAND_Y_UP_M)
    mask &= (Z >= Z_MIN_M) & (Z <= Z_MAX_M)
    return mask

def min_distance_in_band(X, Y, Z):
    mask = band_mask_xyz(X, Y, Z)
    if not np.any(mask):
        return float("nan"), 0.0
    zvals = Z[mask]
    return float(np.nanmin(zvals)), float(np.mean(mask))

# ================== ENDPOINTS ==================

# 🔹 Página HTML simple que muestra el "video" usando /frame
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head>
        <title>Vista ZED</title>
        <style>
          body {
            background: #111;
            color: #eee;
            font-family: sans-serif;
            text-align: center;
          }
          h1 {
            margin-top: 20px;
          }
          img {
            margin-top: 20px;
            border: 2px solid #555;
            max-width: 100%;
          }
        </style>
      </head>
      <body>
        <h1>Stream ZED (/frame)</h1>
        <p>Resolución esperada: 1344x376 @ 15 FPS (SBS)</p>
        <img id="cam" src="/frame" alt="ZED frame" />
        <script>
          const img = document.getElementById('cam');
          function refresh() {
            img.src = '/frame?ts=' + Date.now();
          }
          setInterval(refresh, 100);  // ~10 fps
        </script>
      </body>
    </html>
    """

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "CPU OpenCV Stereo",
        "rectified_yaml": rectify,
        "rectified_uncal": (not rectify) and UNCAL_RECTIFY,
        "W": W, "H": H, "FPS": FPS,
        "has_Q": Q is not None,
        "devices": {"left": LEFT_DEV, "right": RIGHT_DEV, "use_sbs": use_sbs},
        "sbs_flags": {
            "swap": SWAP_HALVES,
            "hflip_left": HFLIP_LEFT, "hflip_right": HFLIP_RIGHT,
            "vflip_left": VFLIP_LEFT, "vflip_right": VFLIP_RIGHT
        },
        "manual_align": {
            "enabled": MANUAL_ALIGN,
            "rot_l_deg": ROT_L_DEG, "rot_r_deg": ROT_R_DEG,
            "shift_l": [SHIFT_L_X, SHIFT_L_Y], "shift_r": [SHIFT_R_X, SHIFT_R_Y],
            "scale_l": SCALE_L, "scale_r": SCALE_R
        },
        "band_m": {
            "x_half": BAND_X_HALF_M,
            "y_down": BAND_Y_DOWN_M,
            "y_up": BAND_Y_UP_M,
            "z_range": [Z_MIN_M, Z_MAX_M]
        }
    }

@app.get("/frame")
def frame():
    try:
        frameL, frameR, _, _ = grab_pair()
    except Exception as e:
        raise HTTPException(500, f"No se pudo capturar frame: {e}")
    if frameR is None:
        view = frameL
    else:
        view = np.hstack((frameL, frameR))
    ok, buf = cv2.imencode(".jpg", view)
    if not ok:
        raise HTTPException(500, "Error codificando frame.")
    return Response(content=buf.tobytes(), media_type="image/jpeg")

@app.get("/mono")
def mono(eye: str = "left"):
    """
    Devuelve solo una de las cámaras (mitad izquierda o derecha) ya procesada
    por grab_pair() (SBS, flips, rectificación, etc).

    Parámetros:
      - eye=left  (por defecto)
      - eye=right
    """
    try:
        frameL, frameR, _, _ = grab_pair()
    except Exception as e:
        raise HTTPException(500, f"No se pudo capturar frame: {e}")

    # Normalizamos el parámetro
    eye = (eye or "left").lower()

    # Elegimos qué mitad mostrar
    if eye == "right" and frameR is not None:
        view = frameR
    else:
        # por defecto usamos la izquierda
        view = frameL

    # Nos aseguramos de trabajar sobre una copia válida en BGR
    view = view.copy()

    ok, buf = cv2.imencode(".jpg", view)
    if not ok:
        raise HTTPException(500, "Error codificando frame mono.")
    return Response(content=buf.tobytes(), media_type="image/jpeg")

def mono_frame_generator(eye: str = "left"):
    """
    Generador de frames JPEG para el stream monocular.
    """
    eye = (eye or "left").lower()
    while True:
        try:
            frameL, frameR, _, _ = grab_pair()
        except Exception as e:
            # Si algo falla, seguimos intentando
            print(f"[mono_stream] error capturando frame: {e}")
            time.sleep(0.05)
            continue

        if eye == "right" and frameR is not None:
            view = frameR
        else:
            view = frameL

        ok, buf = cv2.imencode(".jpg", view)
        if not ok:
            continue

        frame_bytes = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )


@app.get("/mono_stream")
def mono_stream(eye: str = "left"):
    """
    Stream MJPEG de una sola cámara (monocular).
    Usar como:
      - /mono_stream
      - /mono_stream?eye=right
    Se puede ver directo en el navegador o en un <img>.
    """
    return StreamingResponse(
        mono_frame_generator(eye),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/grid")
def grid():
    """Vista de diagnóstico con líneas horizontales para chequear epipolaridad."""
    try:
        frameL, frameR, _, _ = grab_pair()
    except Exception as e:
        raise HTTPException(500, f"No se pudo capturar frame: {e}")
    def draw_lines(img, step=40):
        out = img.copy()
        h = out.shape[0]
        for y in range(step, h, step):
            cv2.line(out, (0,y), (out.shape[1]-1,y), (0,255,0), 1, cv2.LINE_AA)
        return out
    l = draw_lines(frameL)
    r = draw_lines(frameR) if frameR is not None else None
    view = np.hstack((l, r)) if r is not None else l
    ok, buf = cv2.imencode(".jpg", view)
    if not ok:
        raise HTTPException(500, "No se pudo codificar grid")
    return Response(content=buf.tobytes(), media_type="image/jpeg")

@app.get("/preview")
def preview():
    if not ENABLE_PREVIEW:
        raise HTTPException(404, "Preview disabled")

    frameL, frameR, grayL, grayR = grab_pair()
    disp = bm.compute(grayL, grayR)
    _, _, Z = disparity_to_points3d_full(disp)

    # Máscara de puntos con profundidad válida
    valid = np.isfinite(Z)

    # Máxima distancia a mostrar en el mapa (puede ser distinto de THRESHOLD_M*2 si querés)
    max_m = THRESHOLD_M * 2.0

    # Por defecto: todo “muy lejos”
    depth = np.full_like(Z, max_m, dtype=np.float32)

    # Donde hay datos válidos, usamos Z recortado
    depth[valid] = np.clip(Z[valid], 0.0, max_m)

    # Normalizamos 0..max_m -> 0..255
    depth_norm = (depth / max_m * 255.0).astype(np.uint8)

    # Invertimos para que cerca = rojo, lejos = azul
    depth_vis = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_JET)

    # Opcional: poner los inválidos directamente en negro para verlos claros
    depth_vis[~valid] = (0, 0, 0)

    ok, buf = cv2.imencode(".jpg", depth_vis)
    if not ok:
        raise HTTPException(500, "No se pudo codificar preview")
    return Response(content=buf.tobytes(), media_type="image/jpeg")

@app.get("/detect")
def detect():
    t0 = time.time()
    try:
        _, _, grayL, grayR = grab_pair()
    except Exception as e:
        raise HTTPException(500, str(e))
    disp = bm.compute(grayL, grayR)
    X, Y, Z = disparity_to_points3d_full(disp)
    zmin, cov = min_distance_in_band(X, Y, Z)
    obstacle = bool(np.isfinite(zmin) and zmin < THRESHOLD_M)
    dt = time.time() - t0
    return {
        "obstacle": obstacle,
        "min_distance_m": None if not np.isfinite(zmin) else round(zmin, 3),
        "coverage": round(cov, 3),
        "band_m": {
            "x_half": BAND_X_HALF_M,
            "y_down": BAND_Y_DOWN_M,
            "y_up": BAND_Y_UP_M,
            "z_range": [Z_MIN_M, Z_MAX_M]
        },
        "params": {
            "threshold_m": THRESHOLD_M,
            "rectified_yaml": rectify,
            "rectified_uncal": (not rectify) and UNCAL_RECTIFY,
            "baseline_m": BASELINE_M,
            "focal_px": FOCAL_PX
        },
        "latency_s": round(dt, 3)
    }

@app.get("/frame_info")
def frame_info():
    wL = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
    hL = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fpsL = capL.get(cv2.CAP_PROP_FPS)
    info = {"capL":{"W":wL,"H":hL,"FPS":fpsL,"use_sbs":use_sbs}}
    if not use_sbs and capR.isOpened():
        info["capR"] = {
            "W": int(capR.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "H": int(capR.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "FPS": capR.get(cv2.CAP_PROP_FPS)
        }
    return JSONResponse(info)
