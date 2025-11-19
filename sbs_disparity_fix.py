import cv2
import numpy as np

DEVICE = "/dev/video0"
W, H, FPS = 1344, 376, 15

# Estado runtime (podés cambiarlos con teclas)
swap_halves = True      # 'i' → alterna L↔R
hflip_left  = False     # 'a' → alterna espejo horizontal mitad izquierda
hflip_right = True      # 'd' → alterna espejo horizontal mitad derecha
vflip_left  = False     # 'w' → flip vertical izquierda
vflip_right = False     # 's' → flip vertical derecha
do_uncal_rect = False   # 'r' → alterna rectificación no calibrada (ORB+F)

# StereoSGBM básico
bm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*6,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    speckleWindowSize=50,
    speckleRange=2,
    uniquenessRatio=5,
    disp12MaxDiff=1
)

def open_cam(dev, w, h, fps):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    fourcc = cv2.VideoWriter_fourcc(*'YUYV')  # tu cámara reporta YUYV
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    return cap

# --- ORB + F + rectificación no calibrada ---
orb = cv2.ORB_create(1200)
H1 = H2 = None

def compute_uncal_rect(gL, gR):
    # detecta y matchea puntos
    k1, d1 = orb.detectAndCompute(gL, None)
    k2, d2 = orb.detectAndCompute(gR, None)
    if not k1 or not k2 or d1 is None or d2 is None:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(d1, d2)
    if len(m) < 20:
        return None, None
    m = sorted(m, key=lambda x: x.distance)[:200]
    pts1 = np.float32([k1[x.queryIdx].pt for x in m])
    pts2 = np.float32([k2[x.trainIdx].pt for x in m])
    F, inl = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None:
        return None, None
    h, w = gL.shape
    ret, H1, H2 = cv2.stereoRectifyUncalibrated(pts1[inl.ravel()==1], pts2[inl.ravel()==1], F, (w, h))
    if not ret:
        return None, None
    return H1, H2

def apply_flips(img, hflip=False, vflip=False):
    if hflip and vflip: return cv2.flip(img, -1)
    if hflip:           return cv2.flip(img, 1)
    if vflip:           return cv2.flip(img, 0)
    return img

def main():
    global swap_halves, hflip_left, hflip_right, vflip_left, vflip_right, do_uncal_rect, H1, H2

    cap = open_cam(DEVICE, W, H, FPS)
    if not cap.isOpened():
        print("❌ No se pudo abrir", DEVICE)
        return

    print("✅ Capturando SBS de", DEVICE)
    print("Teclas: q salir | i swap L/R | a/d hflip L/R | w/s vflip L/R | r rectificación")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Falló captura")
            break

        h, w = frame.shape[:2]
        mid = w // 2
        left = frame[:, :mid]
        right = frame[:, mid:]

        if swap_halves:
            left, right = right, left

        # flips por mitad
        left  = apply_flips(left,  hflip_left,  vflip_left)
        right = apply_flips(right, hflip_right, vflip_right)

        gL = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # rectificación no calibrada (opcional)
        if do_uncal_rect:
            if H1 is None or H2 is None:
                H1, H2 = compute_uncal_rect(gL, gR)
            if H1 is not None and H2 is not None:
                gL = cv2.warpPerspective(gL, H1, (gL.shape[1], gL.shape[0]))
                gR = cv2.warpPerspective(gR, H2, (gR.shape[1], gR.shape[0]))

        disp = bm.compute(gL, gR).astype(np.float32) / 16.0
        disp[disp < 0] = 0

        # visual
        disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        both = np.hstack([left, right])

        # overlay estado
        txt = f"swap={swap_halves}  hL={hflip_left} hR={hflip_right}  vL={vflip_left} vR={vflip_right}  rect={do_uncal_rect}"
        cv2.putText(both, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("SBS (L|R) + estado", both)
        cv2.imshow("Disparidad", disp_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('i'): swap_halves = not swap_halves; H1=H2=None
        elif key == ord('a'): hflip_left  = not hflip_left;  H1=H2=None
        elif key == ord('d'): hflip_right = not hflip_right; H1=H2=None
        elif key == ord('w'): vflip_left  = not vflip_left;  H1=H2=None
        elif key == ord('s'): vflip_right = not vflip_right; H1=H2=None
        elif key == ord('r'): do_uncal_rect = not do_uncal_rect; H1=H2=None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
