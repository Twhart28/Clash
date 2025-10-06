# ===== All-in-one GUI Macro (Window Picker + Highlight + OCR + Actions) =====
# Requires: easyocr pillow numpy mss pyautogui keyboard pypiwin32
# Install:  py -m pip install easyocr pillow numpy mss pyautogui keyboard pypiwin32

import os, re, time, math, threading, ctypes, statistics
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageFilter
import mss
import pyautogui as pag
import keyboard

# Win32
import win32gui, win32con, win32api

# GUI
import tkinter as tk
from tkinter import ttk, messagebox

# High-DPI awareness (like AHK's SetProcessDPIAware)
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

import sys  # up with the other imports

# --- Hard-stop on Escape (global) ---
def _hard_stop():
    try:
        _stop_flag.set()
    except Exception:
        pass
    os._exit(0)  # instant process kill (no cleanup)

keyboard.add_hotkey('esc', _hard_stop, suppress=True)
# ---------------- Config ----------------
KEEP_SNAPSHOTS = False   # False = delete OCR BMPs after use
DEBUG = False            # console logs

# Allowed colors (exact)
ALLOWED_LOOT   = ["#FFFBCC", "#FFE8FD", "#F3F3F3","#E0DCB3","#E0CCDE","#D5D5D5"]
Menu_Closed_Chat_Color= ["#EA8A3B","#CE7934"]
Menu_Open_Chat_Color = ["#75451E","#673D1A"]
Next_Raid_Color = ["#FCBB36","#DDA32F"]
Surrender_Okay_Color = ["#D5F376","#BFD96B"] #Color of okay surrender and return home button

# Use the resolution you used when taking AHK measurements:
MEASURE_W, MEASURE_H = 1920, 1080
def PXW(x: int) -> float: return round(x / MEASURE_W, 6)
def PYW(y: int) -> float: return round(y / MEASURE_H, 6)

# Regions (raw pixels from your AHK measurements)
LOOT_BOX_PX   = (57,  105, 250, 263)

def percent_box(box_px: Tuple[int,int,int,int]) -> Tuple[float,float,float,float]:
    x1, y1, x2, y2 = box_px
    return (PXW(x1), PYW(y1), PXW(x2), PYW(y2))

# Default thresholds (GUI can change)
GOLD_MIN   = 1_000_000
ELIXIR_MIN = 1_000_000
DARK_MIN   = 0
TOTAL_MIN  = 0
USE_TOTAL  = False

# ---------------- Window selection / geometry ----------------
_selected_hwnd = None        # set by GUI
_stop_flag = threading.Event()

def list_visible_windows():
    wins = []
    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd): return
        title = win32gui.GetWindowText(hwnd)
        if not title: return
        # skip tool windows without real client area
        rect = win32gui.GetClientRect(hwnd)
        if rect == (0,0,0,0): return
        wins.append((hwnd, title))
    win32gui.EnumWindows(cb, None)
    # Dedup by title, but keep hwnd
    return wins

def get_client_rect_screen(hwnd) -> Tuple[int,int,int,int]:
    if not hwnd or not win32gui.IsWindow(hwnd): return 0,0,0,0
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    (L, T) = win32gui.ClientToScreen(hwnd, (0,0))
    (R, B) = win32gui.ClientToScreen(hwnd, (right, bottom))
    return L, T, R-L, B-T

def set_foreground(hwnd):
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass

def set_base_from_monitor(hwnd):
    """Set BASE_W/BASE_H to the physical resolution of the monitor that contains hwnd."""
    try:
        hmon = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
        info = win32api.GetMonitorInfo(hmon)
        (L, T, R, B) = info["Monitor"]
        w, h = R - L, B - T
        if w > 0 and h > 0:
            global BASE_W, BASE_H
            BASE_W, BASE_H = w, h
            if DEBUG: print(f"[monitor] BASE set to {BASE_W}x{BASE_H}")
    except Exception as e:
        if DEBUG: print("Failed to set base from monitor:", e)

def abs_from_pct(px: float, py: float) -> Tuple[int,int]:
    L,T,W,H = get_client_rect_screen(_selected_hwnd)
    if W<=0 or H<=0: return 0,0
    return L + round(px*W), T + round(py*H)

# ---------------- Mouse / wheel / pixel ----------------
def click_pct(px: float, py: float):
    x,y = abs_from_pct(px, py)
    pag.moveTo(x, y, duration=0); pag.click()

def drag_pct(x1p: float, y1p: float, x2p: float, y2p: float, steps: int=0, delay_ms: int=0):
    x1,y1 = abs_from_pct(x1p, y1p); x2,y2 = abs_from_pct(x2p, y2p)
    pag.moveTo(x1, y1, duration=0); pag.mouseDown()
    if steps>0:
        dx = (x2-x1)/steps; dy = (y2-y1)/steps; cx,cy=float(x1),float(y1)
        for _ in range(steps):
            cx += dx; cy += dy
            pag.moveTo(int(round(cx)), int(round(cy)), duration=0)
            if delay_ms>0: time.sleep(delay_ms/1000.0)
    else:
        pag.moveTo(x2, y2, duration=0)
    pag.mouseUp()

def wheel_down(notches=1):
    for _ in range(max(1, int(notches))):
        win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, -120, 0)
        time.sleep(0.01)

def parse_hex_color(c) -> Tuple[int, int, int]:
    """Convert '#RRGGBB' / '0xRRGGBB' / int / (r,g,b) into (r, g, b)."""
    # Already an RGB triple?
    if isinstance(c, (list, tuple)) and len(c) == 3:
        r, g, b = c
        return int(r) & 255, int(g) & 255, int(b) & 255

    # Hex given as int or string
    if isinstance(c, int):
        v = c
    else:
        s = str(c).strip()
        if s.startswith("#"):
            s = s[1:]
        if s.lower().startswith("0x"):
            s = s[2:]
        v = int(s, 16)
    return (v >> 16) & 255, (v >> 8) & 255, v & 255

def parse_hex_colors(c) -> List[Tuple[int,int,int]]:
    """Accept a single hex (str/int) or a list/tuple of them; return list of (r,g,b)."""
    if isinstance(c, (list, tuple, set)):
        return [parse_hex_color(x) for x in c]
    return [parse_hex_color(c)]

def color_close(c1, c2, tol=12):
    return max(abs(c1[0]-c2[0]), abs(c1[1]-c2[1]), abs(c1[2]-c2[2])) <= tol

def wait_pixel_color_pct(px: float, py: float, color_or_colors, timeout_ms=10000, tol=0, poll_ms=50) -> bool:
    targets = parse_hex_colors(color_or_colors)
    x, y = abs_from_pct(px, py)
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        if _stop_flag.is_set(): return False
        try:
            r, g, b = pag.pixel(x, y)
            if any(color_close((r, g, b), t, tol) for t in targets):
                return True
        except Exception:
            pass
        time.sleep(poll_ms / 1000.0)
    return False

def wait_all_pixels_pct(pxs: List[Tuple[float,float,Any]], timeout_ms=55000, tol=0, poll_ms=50) -> bool:
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        if _stop_flag.is_set(): return False
        allok = True
        for (px, py, hx) in pxs:
            x, y = abs_from_pct(px, py)
            targets = parse_hex_colors(hx)
            ok = False
            try:
                r, g, b = pag.pixel(x, y)
                ok = any(color_close((r, g, b), t, tol) for t in targets)
            except Exception:
                ok = False
            if not ok:
                allok = False
                break
        if allok:
            return True
        time.sleep(poll_ms / 1000.0)
    return False

# ---------------- Screenshot + B/W whitelist ----------------
def make_snap_path(directory: str, prefix: str="snap") -> str:
    os.makedirs(directory, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + f"{int(datetime.now().microsecond/1000):03d}"
    return os.path.join(directory, f"{prefix}_{ts}.bmp")

def hex_to_rgb_int(hx: str) -> int:
    s = hx.strip()
    if s.startswith("#"): s = s[1:]
    return int(s, 16)

def snap_client_region_pct_bw(x1p: float, y1p: float, x2p: float, y2p: float,
                              allowed_hex_list: List[str], save_dir: str, prefix: str) -> str:
    L,T,W,H = get_client_rect_screen(_selected_hwnd)
    if W<=0 or H<=0: return ""
    x1 = L + round(x1p*W); y1 = T + round(y1p*H)
    x2 = L + round(x2p*W); y2 = T + round(y2p*H)
    if x2<x1: x1,x2 = x2,x1
    if y2<y1: y1,y2 = y2,y1
    w = x2-x1; h = y2-y1
    if w<=0 or h<=0: return ""
    with mss.mss() as sct:
        img = sct.grab({"left": x1, "top": y1, "width": w, "height": h})
        arr = np.array(img)[:,:,:3][:,:,::-1]  # BGRA -> RGB
    allowed = set(hex_to_rgb_int(h) for h in allowed_hex_list)
    rgb_int = (arr[:,:,0].astype(np.uint32)<<16) | (arr[:,:,1].astype(np.uint32)<<8) | arr[:,:,2].astype(np.uint32)
    mask = np.isin(rgb_int, list(allowed))
    bw = np.where(mask[:,:,None], 0, 255).astype(np.uint8)
    bw = np.repeat(bw, 3, axis=2)
    im = Image.fromarray(bw, mode="RGB")
    path = make_snap_path(save_dir, prefix)
    im.save(path, format="BMP")
    return path if (os.path.exists(path) and os.path.getsize(path)>1000) else ""

# ---- EasyOCR lazy init (fast GUI startup) ----
_READER = None
_READER_READY = threading.Event()
_READER_ERR = None

def _init_easyocr():
    global _READER, _READER_ERR
    try:
        import easyocr
        _READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    except Exception as e:
        _READER_ERR = e
    finally:
        _READER_READY.set()

def ensure_reader():
    # If background init is done, use it; otherwise wait (only when called)
    if _READER is not None:
        return _READER
    _READER_READY.wait()
    if _READER is None:
        raise RuntimeError(f"EasyOCR init failed: {_READER_ERR}")
    return _READER

READ_KW = dict(
    detail=1,
    paragraph=False,
    allowlist='0123456789 ',
    decoder='greedy',
    mag_ratio=3.0,
    add_margin=0.08,
    contrast_ths=0.08,
    adjust_contrast=0.8,
    ycenter_ths=0.6,
    height_ths=0.6,
    width_ths=0.6,
    slope_ths=0.35,
    rotation_info=[0],
)

LINE_Y_TOL_FRAC = 0.6
ISLAND_MIN_AREA_FRAC = 0.002
ISLAND_CLOSE_SIZE = 3
ISLAND_MARGIN_PIX = 2

def _normalize_digits(s: str) -> str:
    return (s or "").replace('I','1').replace('l','1').replace('|','1')

def _bbox_center(bbox):
    xs = [float(p[0]) for p in bbox]; ys = [float(p[1]) for p in bbox]
    cx = sum(xs)/4.0; cy = sum(ys)/4.0
    w = max(xs)-min(xs); h = max(ys)-min(ys)
    return cx, cy, w, h

def _group_by_line(dets):
    if not dets: return []
    dets_sorted = sorted(dets, key=lambda d: d['cy'])
    median_h = statistics.median(d['h'] for d in dets_sorted) or 1.0
    y_tol = LINE_Y_TOL_FRAC * median_h
    lines, current, current_y = [], [], None
    for d in dets_sorted:
        if not current: current=[d]; current_y=d['cy']; continue
        if abs(d['cy']-current_y) <= y_tol:
            current.append(d)
            current_y = (current_y*(len(current)-1)+d['cy'])/len(current)
        else:
            current.sort(key=lambda z: z['cx']); lines.append(current)
            current=[d]; current_y=d['cy']
    if current:
        current.sort(key=lambda z: z['cx']); lines.append(current)
    lines.sort(key=lambda L: sum(p['cy'] for p in L)/len(L))
    return lines

def _concat_line(line):
    raw = " ".join(_normalize_digits(d['text']) for d in line if d['text'])
    return re.sub(r'\D+', '', raw)

def _line_union_bbox(line, img_w, img_h):
    xs=[]; ys=[]
    for d in line:
        cx,cy,w,h = d['cx'], d['cy'], d['w'], d['h']
        xs += [cx-w/2, cx+w/2]; ys += [cy-h/2, cy+h/2]
    x1 = max(0, int(min(xs) - ISLAND_MARGIN_PIX))
    y1 = max(0, int(min(ys) - ISLAND_MARGIN_PIX))
    x2 = min(int(max(xs) + ISLAND_MARGIN_PIX), img_w-1)
    y2 = min(int(max(ys) + ISLAND_MARGIN_PIX), img_h-1)
    return x1,y1,x2,y2

def _count_black_islands(line_crop: Image.Image) -> int:
    g = line_crop.convert('L')
    if ISLAND_CLOSE_SIZE and ISLAND_CLOSE_SIZE>1:
        g = g.filter(ImageFilter.MinFilter(size=ISLAND_CLOSE_SIZE))
        g = g.filter(ImageFilter.MaxFilter(size=ISLAND_CLOSE_SIZE))
    a = np.array(g, dtype=np.uint8)
    black = (a <= 128).astype(np.uint8)
    h,w = black.shape
    area = h*w
    min_area = max(1, int(ISLAND_MIN_AREA_FRAC*area))
    visited = np.zeros_like(black, dtype=np.uint8)
    islands = 0
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for y in range(h):
        for x in range(w):
            if black[y,x]==1 and not visited[y,x]:
                stack=[(y,x)]; visited[y,x]=1; size=0
                while stack:
                    cy,cx = stack.pop(); size+=1
                    for dy,dx in nbrs:
                        ny,nx = cy+dy, cx+dx
                        if 0<=ny<h and 0<=nx<w and black[ny,nx]==1 and not visited[ny,nx]:
                            visited[ny,nx]=1; stack.append((ny,nx))
                if size>=min_area: islands+=1
    return islands

def read_digits_from_image(path: str, rois=None) -> List[str]:
    img = Image.open(path).convert('RGB'); img_w, img_h = img.size
    values = []
    def process(results):
        dets=[]
        for bbox, text, conf in results:
            text=_normalize_digits(text)
            cx,cy,w,h = _bbox_center(bbox)
            dets.append({'text':text,'conf':float(conf),'cx':cx,'cy':cy,'w':w,'h':h})
        lines = _group_by_line(dets)
        line_vals=[]
        for line in lines:
            concat = _concat_line(line)
            x1,y1,x2,y2 = _line_union_bbox(line, img_w, img_h)
            crop = img.crop((x1,y1,x2,y2))
            islands = _count_black_islands(crop)
            corrected = concat
            if isinstance(concat, str) and (islands - len(concat)) == 1:
                corrected = '1' + concat
            line_vals.append(corrected)
        return line_vals
    if rois:
        for (x,y,w,h) in rois:
            crop = img.crop((x, y, x+w, y+h))
            reader = ensure_reader()
            results = reader.readtext(np.array(crop), **READ_KW)
            values.extend(process(results))
    else:
        reader = ensure_reader()
        results = reader.readtext(np.array(img), **READ_KW)
        values = process(results)
    return values

# ---------------- Loot parsing / filters ----------------
def parse_amount(s: Any):
    if s is None: return ""
    s = str(s).strip().replace(",", "")
    s = s.replace("o","0").replace("O","0").replace("l","1").replace("I","1")
    m = re.match(r'^\s*(\d+(?:\.\d+)?)\s*([km]?)\s*$', s)
    if not m: return ""
    val = float(m.group(1)); unit = m.group(2).lower()
    if unit=="k": val*=1_000
    elif unit=="m": val*=1_000_000
    return int(round(val))

def try_normalize_loot(loot_list: List[str]) -> Tuple[bool, Dict[str,int], str]:
    if not isinstance(loot_list, (list,tuple)) or len(loot_list)==0:
        return False, {}, "No loot array returned"
    g = parse_amount(loot_list[0]) if len(loot_list)>=1 else ""
    e = parse_amount(loot_list[1]) if len(loot_list)>=2 else ""
    d = parse_amount(loot_list[2]) if len(loot_list)>=3 else ""
    if g=="" or e=="" or d=="": return False, {}, "Missing or malformed Gold/Elixir/DE"
    return True, {"gold":g, "elixir":e, "dark":d}, ""

def should_attack(loot: Dict[str,int]) -> bool:
    global GOLD_MIN, ELIXIR_MIN, DARK_MIN, TOTAL_MIN, USE_TOTAL
    active, ok = 0, True
    if GOLD_MIN>0:   active+=1; ok = ok and (loot["gold"]   >= GOLD_MIN)
    if ELIXIR_MIN>0: active+=1; ok = ok and (loot["elixir"] >= ELIXIR_MIN)
    if DARK_MIN>0:   active+=1; ok = ok and (loot["dark"]   >= DARK_MIN)
    if USE_TOTAL and TOTAL_MIN>0:
        active+=1; ok = ok and ((loot["gold"]+loot["elixir"]+loot["dark"]) >= TOTAL_MIN)
    return ok if active>0 else False

# ---------------- Actions (from your AHK) ----------------
def start_raid():
    click_pct(PXW(106),   PYW(970))
    _ = wait_pixel_color_pct(PXW(537), PYW(1034), Next_Raid_Color, 10000, 0, 100)
    click_pct(PXW(537), PYW(1034))
    _ = wait_pixel_color_pct(PXW(1877), PYW(801), Next_Raid_Color, 12000, 30, 50)

def next_raid():
    click_pct(PXW(1877), PYW(801)); time.sleep(1.0)
    _ = wait_pixel_color_pct(PXW(1877), PYW(801), Next_Raid_Color, 12000, 30, 50)

def drag_attack():
    set_foreground(_selected_hwnd)
    steps, delay = 10, 5
    for _ in range(20): wheel_down(1)
    drag_pct(PXW(1130), PYW(788), PXW(493), PYW(310), steps=1, delay_ms=10)

    keyboard.send("r"); time.sleep(1)
    click_pct(PXW(1688), PYW(359))
    keyboard.send("r")

    keyboard.send("w"); time.sleep(1)
    click_pct(PXW(911), PYW(869))
    keyboard.send("w")

    keyboard.send("z"); time.sleep(1)
    pag.click()

    keyboard.send("2"); time.sleep(1)
    pag.click()

    keyboard.send("1"); time.sleep(1)
    click_pct(PXW(1049), PYW(844))
    click_pct(PXW(1627), PYW(398)); time.sleep(10.0)

    startXp, startYp = PXW(1402), PYW(538)
    endXp,   endYp   = PXW(1268), PYW(656)
    clickCount, totalSteps = 13, 13
    clickInterval = round(totalSteps/clickCount)
    dxp = (endXp-startXp)/totalSteps; dyp = (endYp-startYp)/totalSteps
    cxp, cyp = startXp, startYp
    x,y = abs_from_pct(cxp, cyp); pag.moveTo(x,y, duration=0)
    for i in range(1, totalSteps+1):
        if _stop_flag.is_set(): return
        cxp += dxp; cyp += dyp
        x,y = abs_from_pct(cxp, cyp); pag.moveTo(x,y, duration=0)
        if (i % clickInterval) == 0: pag.click()
        time.sleep(delay/1000.0)

    keyboard.send("2"); time.sleep(1)
    clickCount = 13; clickInterval = round(totalSteps/clickCount)
    cxp, cyp = startXp, startYp
    x,y = abs_from_pct(cxp, cyp); pag.moveTo(x,y, duration=0)
    for i in range(1, totalSteps+1):
        if _stop_flag.is_set(): return
        cxp += dxp; cyp += dyp
        x,y = abs_from_pct(cxp, cyp); pag.moveTo(x,y, duration=0)
        if (i % clickInterval) == 0: pag.click()
        time.sleep(delay/1000.0)

    keyboard.send("q"); time.sleep(1); click_pct(PXW(1335), PYW(597))
    keyboard.send("e"); time.sleep(1); click_pct(PXW(1335), PYW(597))
    time.sleep(10)
    keyboard.send("q"); keyboard.send("e")

    keyboard.send("a"); time.sleep(1)
    click_pct(PXW(921), PYW(492))
    click_pct(PXW(1040), PYW(401))
    click_pct(PXW(1201), PYW(307));time.sleep(5.0)
    click_pct(PXW(903), PYW(361))
    click_pct(PXW(1030), PYW(260))
    keyboard.send("s"); time.sleep(1)
    click_pct(PXW(946), PYW(339))

    tri_ok = wait_all_pixels_pct([
        (PXW(347), PYW(555), "#000000"),
        (PXW(1569), PYW(624), "#000000"),
    ], 55000, 0, 100)

    if tri_ok:
        click_pct(PXW(1051), PYW(908))
    else:
        click_pct(PXW(113),  PYW(858))
        _ = wait_pixel_color_pct(PXW(1229), PYW(649), Surrender_Okay_Color, 4000, 30, 100)
        click_pct(PXW(1229), PYW(649)); time.sleep(2)
        wait_all_pixels_pct([
            (PXW(347), PYW(555), "#000000"),
            (PXW(1569), PYW(624), "#000000"),
        ], 10000, 0, 100)
        click_pct(PXW(1051),  PYW(908))
    no_star_bonus = wait_pixel_color_pct(PXW(65), PYW(520), Menu_Closed_Chat_Color, 6000, 0, 100)
    if not no_star_bonus:
        click_pct(PXW(958),  PYW(862))
    _ = wait_pixel_color_pct(PXW(65), PYW(520), Menu_Closed_Chat_Color, 10000, 0, 100)

# ---------------- OCR wrappers ----------------
def get_loot(save_dir: str) -> List[str]:
    time.sleep(0.1)
    x1p, y1p, x2p, y2p = percent_box(LOOT_BOX_PX)
    imgBW = snap_client_region_pct_bw(x1p, y1p, x2p, y2p, ALLOWED_LOOT, save_dir, "loot_bw")
    if not imgBW: return []
    try:
        vals = read_digits_from_image(imgBW)
        out=[]
        for v in vals:
            if v:
                out.append(v)
                if len(out)==3: break
        return out
    finally:
        if not KEEP_SNAPSHOTS and imgBW and os.path.exists(imgBW):
            try: os.remove(imgBW)
            except: pass

# ---------------- Highlight overlay ----------------
class HighlightOverlay:
    """Draws a breathing neon border around a target window's client rect.
       Auto-hides after 3 seconds or when selection changes."""
    def __init__(self, root, border_px: int = 6, duration_ms: int = 3000):
        self.root = root
        self.border_px = border_px
        self.duration_ms = duration_ms
        self._wins = []        # [top, right, bottom, left] Toplevels
        self._pulse_job = None
        self._hide_job = None
        self.visible = False

    def _make_windows(self):
        if self._wins:
            return
        for _ in range(4):
            w = tk.Toplevel(self.root)
            w.overrideredirect(True)
            w.attributes("-topmost", True)
            # make clicks pass through (optional): keep normal so user can close app
            w.configure(bg="#FFFF33")  # initial neon yellow
            self._wins.append(w)

    def _place(self, L, T, W, H):
        b = self.border_px
        self._wins[0].geometry(f"{W}x{b}+{L}+{T}")             # top
        self._wins[1].geometry(f"{b}x{H}+{L+W-b}+{T}")         # right
        self._wins[2].geometry(f"{W}x{b}+{L}+{T+H-b}")         # bottom
        self._wins[3].geometry(f"{b}x{H}+{L}+{T}")             # left

    def _set_color(self, hexcolor: str):
        for w in self._wins:
            w.configure(bg=hexcolor)

    def _pulse(self):
        # Breathing between bright yellow and warmer yellow
        t = time.time()
        phase = (math.sin(t * 3.3) + 1) / 2.0  # 0..1
        # Interpolate between two yellows
        # c1 = (255, 255, 70), c2 = (255, 208, 0)
        r, g, b = 255, int(255 * (0.82 + 0.18 * phase)), int(70 + (0 * (1 - phase)))
        hexcol = f"#{r:02X}{g:02X}{b:02X}"
        self._set_color(hexcol)
        self._pulse_job = self.root.after(50, self._pulse)

    def show(self, hwnd):
        L, T, W, H = get_client_rect_screen(hwnd)
        if W <= 0 or H <= 0:
            self.hide()
            return
        self._make_windows()
        self._place(L, T, W, H)
        for w in self._wins:
            w.deiconify()
        self.visible = True

        # start pulse
        if self._pulse_job is None:
            self._pulse()

        # auto-hide timer reset
        if self._hide_job is not None:
            self.root.after_cancel(self._hide_job)
        self._hide_job = self.root.after(self.duration_ms, self.hide)

    def hide(self):
        if not self._wins:
            self.visible = False
            return
        if self._pulse_job is not None:
            self.root.after_cancel(self._pulse_job)
            self._pulse_job = None
        if self._hide_job is not None:
            self.root.after_cancel(self._hide_job)
            self._hide_job = None
        for w in self._wins:
            try:
                w.withdraw()
            except Exception:
                pass
        self.visible = False

# ---------------- GUI ----------------
class ControlPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        # --- Theming ---
        self.title("Macro Control")
        self.attributes("-topmost", True)
        self.resizable(False, False)

        # Colors
        BG      = "#1E1F2A"
        PANEL   = "#26293A"
        FG      = "#EAEAEA"
        ACCENT  = "#FFD54A"
        ACCENT2 = "#FFEA7F"

        style = ttk.Style()
        try:
            style.theme_use("clam")  # allows custom colors reliably
        except Exception:
            pass

        # Root background
        self.configure(bg=BG)

        # Frames and labels
        style.configure("TFrame", background=BG)
        style.configure("TLabelframe", background=PANEL, foreground=FG, borderwidth=1)
        style.configure("TLabelframe.Label", background=PANEL, foreground=FG)
        style.configure("TLabel", background=BG, foreground=FG)

        # Buttons
        style.configure("TButton", background=ACCENT, foreground="#000000", padding=6)
        style.map("TButton",
                background=[("active", ACCENT2)],
                relief=[("pressed", "sunken"), ("!pressed", "raised")])

        # Checkbutton / Entry
        style.configure("TCheckbutton", background=PANEL, foreground=FG)
        style.configure("TEntry", fieldbackground="#2D3047", foreground=FG)

        # Make labelframes sit nicely against root
        # (You already use ttk.LabelFrame; backgrounds set above)

        # Tweak listbox colors (tk widget, not ttk)

        self.bind_all('<Escape>', lambda e: _hard_stop())
        self.title("Macro Control")
        self.attributes("-topmost", True)
        self.resizable(False, False)

        self.overlay = HighlightOverlay(self)
        self.window_items = []  # (hwnd, title)
        self.running = False

        # --- Window picker ---
        frm_top = ttk.LabelFrame(self, text="Target Window")
        frm_top.grid(row=0, column=0, padx=10, pady=8, sticky="ew")

        self.win_list = tk.Listbox(frm_top, width=50, height=8, exportselection=False)
        self.win_list.grid(row=0, column=0, columnspan=3, padx=6, pady=6, sticky="ew")
        self.win_list.bind("<<ListboxSelect>>", lambda e: self.overlay.hide())
        self.win_list.configure(
            bg="#202331", fg=FG,
            selectbackground=ACCENT, selectforeground="#000000",
            highlightthickness=0, relief="flat"
        )

        ttk.Button(frm_top, text="Refresh", command=self.refresh_windows).grid(row=1, column=0, padx=6, pady=4)
        self.btn_highlight = ttk.Button(frm_top, text="Highlight", command=self.highlight_once)
        self.btn_highlight.grid(row=1, column=1, padx=6, pady=4)
        self.btn_focus = ttk.Button(frm_top, text="Focus", command=self.focus_selected)
        self.btn_focus.grid(row=1, column=2, padx=6, pady=4)

        # --- Thresholds ---
        frm_th = ttk.LabelFrame(self, text="Loot Thresholds")
        frm_th.grid(row=1, column=0, padx=10, pady=6, sticky="ew")

        row=0
        ttk.Label(frm_th, text="Gold ≥").grid(row=row, column=0, sticky="e")
        self.var_gold = tk.StringVar(value=str(GOLD_MIN))
        ttk.Entry(frm_th, textvariable=self.var_gold, width=12).grid(row=row, column=1, padx=6)

        ttk.Label(frm_th, text="Elixir ≥").grid(row=row, column=2, sticky="e")
        self.var_elix = tk.StringVar(value=str(ELIXIR_MIN))
        ttk.Entry(frm_th, textvariable=self.var_elix, width=12).grid(row=row, column=3, padx=6)

        row+=1
        ttk.Label(frm_th, text="Dark ≥").grid(row=row, column=0, sticky="e")
        self.var_dark = tk.StringVar(value=str(DARK_MIN))
        ttk.Entry(frm_th, textvariable=self.var_dark, width=12).grid(row=row, column=1, padx=6)

        ttk.Label(frm_th, text="Total ≥").grid(row=row, column=2, sticky="e")
        self.var_total = tk.StringVar(value=str(TOTAL_MIN))
        ttk.Entry(frm_th, textvariable=self.var_total, width=12).grid(row=row, column=3, padx=6)

        self.var_use_total = tk.BooleanVar(value=USE_TOTAL)
        ttk.Checkbutton(frm_th, text="Use total rule", variable=self.var_use_total).grid(row=row, column=4, padx=6)

        row+=1
        ttk.Button(frm_th, text="Preset 1: G+E ≥ 1,600,000", command=self.preset1).grid(row=row, column=0, columnspan=2, padx=4, pady=4)
        ttk.Button(frm_th, text="Preset 2: Gold ≥ 1,000,000", command=self.preset2).grid(row=row, column=2, padx=4, pady=4)
        ttk.Button(frm_th, text="Preset 3: Elixir ≥ 1,000,000", command=self.preset3).grid(row=row, column=3, padx=4, pady=4)
        ttk.Button(frm_th, text="Preset 4: Dark ≥ 20,000", command=self.preset4).grid(row=row, column=4, padx=4, pady=4)

        # --- Controls ---
        frm_ctl = ttk.Frame(self)
        frm_ctl.grid(row=2, column=0, padx=10, pady=8, sticky="ew")
        self.btn_start = ttk.Button(frm_ctl, text="Start", command=self.start_macro)
        self.btn_start.grid(row=0, column=0, padx=6)

        # --- Debug / OCR quick test ---
        ttk.Button(frm_ctl, text="Test Loot OCR", command=self.test_loot_ocr).grid(row=0, column=3, padx=6)


        self.status = ttk.Label(self, text="Ready.", anchor="w")
        self.status.grid(row=3, column=0, padx=10, pady=(0,10), sticky="ew")

        # Loading spinner while OCR initializes
        self.pb = ttk.Progressbar(self, mode="indeterminate", length=140)
        self.pb.grid(row=3, column=0, sticky="e", padx=10)   # next to your status
        self.pb.start(40)

        # Keep Start disabled until OCR is ready
        self.btn_start.config(state="disabled")

        # Kick off OCR init in the background, then poll readiness
        threading.Thread(target=_init_easyocr, daemon=True).start()
        self.after(100, self._check_ocr_ready)
        self.refresh_windows()

        # Clean up overlay when closing
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._center_on_screen()

    def _check_ocr_ready(self):
        if _READER_READY.is_set():
            self.pb.stop()
            self.pb.destroy()
            if _READER_ERR:
                self.status.config(text=f"OCR failed: {_READER_ERR}")
                # (Leave Start disabled; or enable if you want to allow non-OCR actions)
            else:
                self.status.config(text="Ready. OCR initialized.")
                self.btn_start.config(state="normal")
            return
        self.after(100, self._check_ocr_ready)

    # Window picker actions
    def refresh_windows(self):
        self.window_items = list_visible_windows()
        self.win_list.delete(0, tk.END)

        default_idx = None
        for i, (_, title) in enumerate(self.window_items):
            self.win_list.insert(tk.END, title)
            if default_idx is None and "clash of clans" in title.lower():
                default_idx = i

        if default_idx is not None:
            self.win_list.selection_clear(0, tk.END)
            self.win_list.selection_set(default_idx)
            self.win_list.see(default_idx)
            # optional: reflect in status, but don't steal focus from the user
            self.status.config(text=f"Default target: {self.window_items[default_idx][1]}")

    def selected_hwnd(self):
        sel = self.win_list.curselection()
        if not sel: return None
        idx = sel[0]
        return self.window_items[idx][0]

    def highlight_once(self):
        hwnd = self.selected_hwnd()
        if not hwnd:
            messagebox.showinfo("Pick a window", "Select a window in the list first.")
            return
        # One-shot highlight; auto-hides after ~3s via HighlightOverlay
        self.overlay.show(hwnd)
        # Optional: briefly disable button to prevent spam clicks
        try:
            self.btn_highlight.state(['disabled'])
            self.after(3500, lambda: self.btn_highlight.state(['!disabled']))
        except Exception:
            pass

    def focus_selected(self):
        hwnd = self.selected_hwnd()
        if not hwnd:
            messagebox.showinfo("Pick a window", "Select a window in the list first.")
            return
        set_foreground(hwnd)
        self.status.config(text=f"Focused: {win32gui.GetWindowText(hwnd)}")
        self.overlay.show(hwnd)  # optional: confirm focus with a quick border

    # Presets
    def preset1(self):
        self.var_gold.set("0"); self.var_elix.set("0"); self.var_dark.set("0")
        self.var_total.set("1600000"); self.var_use_total.set(True)

    def preset2(self):
        self.var_gold.set("1000000"); self.var_elix.set("0"); self.var_dark.set("0")
        self.var_total.set("0"); self.var_use_total.set(False)

    def preset3(self):
        self.var_gold.set("0"); self.var_elix.set("1000000"); self.var_dark.set("0")
        self.var_total.set("0"); self.var_use_total.set(False)

    def preset4(self):
        self.var_gold.set("0"); self.var_elix.set("0"); self.var_dark.set("20000")
        self.var_total.set("0"); self.var_use_total.set(False)

    # Start/Stop
    def start_macro(self):
        global _selected_hwnd, GOLD_MIN, ELIXIR_MIN, DARK_MIN, TOTAL_MIN, USE_TOTAL
        hwnd = self.selected_hwnd()
        if not hwnd:
            messagebox.showerror("No window", "Select a window to target.")
            return
        _selected_hwnd = hwnd

        # ⛔️ DO NOT change PXW/PYW base here.
        # (Optional) Just log the monitor size for visibility.
        try:
            hmon = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
            info = win32api.GetMonitorInfo(hmon)
            L, T, R, B = info["Monitor"]
            if DEBUG:
                print(f"[monitor] target monitor is {R-L}x{B-T}; using AHK base {MEASURE_W}x{MEASURE_H}")
        except Exception as e:
            if DEBUG: print("monitor size lookup failed:", e)

        # thresholds
        try:
            GOLD_MIN   = int(self.var_gold.get() or "0")
            ELIXIR_MIN = int(self.var_elix.get() or "0")
            DARK_MIN   = int(self.var_dark.get() or "0")
            TOTAL_MIN  = int(self.var_total.get() or "0")
            USE_TOTAL  = bool(self.var_use_total.get())
        except ValueError:
            messagebox.showerror("Invalid threshold", "Please enter integer amounts (you can use presets).")
            return

        self.status.config(text=f"Running on: {win32gui.GetWindowText(hwnd)}")
        self.btn_start.config(state="disabled")

        # Hide the UI (Esc still hard-stops)
        try:
            self.withdraw()
        except Exception:
            pass

        _stop_flag.clear()
        threading.Thread(target=self.run_loop, daemon=True).start()

    def stop_macro(self):
        _stop_flag.set()
        self.status.config(text="Stopping...")
        self.btn_start.config(state="normal")

    def on_close(self):
        _stop_flag.set()
        try: self.overlay.hide()
        except: pass
        self.destroy()

    # Core macro ported from your AHK
    def run_loop(self):
        screenshot_dir = os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)

        # Search loop for bases
        start_raid()
        while not _stop_flag.is_set():
            next_raid()
            vals = get_loot(screenshot_dir)
            success, loot_obj, why = try_normalize_loot(vals)
            if not success:
                if DEBUG: print("Loot OCR failed:", why, " -> skipping base")
                continue
            # Decision
            if should_attack(loot_obj):
                drag_attack()
                break
            else:
                continue

    def _center_on_screen(self):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"+{x}+{y}")

        # Ensure OCR is available
        if not _READER_READY.is_set() or _READER is None:
            try:
                ensure_reader()
            except Exception as e:
                messagebox.showerror("OCR not ready", f"EasyOCR init failed:\n{e}")
                return

        # Use the selected window
        global _selected_hwnd, KEEP_SNAPSHOTS
        _selected_hwnd = hwnd
        set_foreground(hwnd)

        # Where snapshots go
        screenshot_dir = os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)

    def test_loot_ocr(self):
        hwnd = self.selected_hwnd()
        if not hwnd:
            messagebox.showerror("No window", "Select a window to target first.")
            return

        # Ensure OCR is available
        if not _READER_READY.is_set() or _READER is None:
            try:
                ensure_reader()
            except Exception as e:
                messagebox.showerror("OCR not ready", f"EasyOCR init failed:\n{e}")
                return

        # Use the selected window
        global _selected_hwnd, KEEP_SNAPSHOTS
        _selected_hwnd = hwnd
        set_foreground(hwnd)

        # Where snapshots go
        screenshot_dir = os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)

        # Force-keep the snapshot for this test
        prev_keep = KEEP_SNAPSHOTS
        KEEP_SNAPSHOTS = True
        try:
            vals = get_loot(screenshot_dir)  # returns up to 3 strings
        finally:
            KEEP_SNAPSHOTS = prev_keep

        # Build message
        if not vals:
            messagebox.showwarning(
                "Loot OCR",
                "No loot digits detected.\n\n" +
                "Tips:\n• Ensure the game window is visible\n" +
                "• Confirm LOOT_BOX coords\n" +
                "• Try higher brightness / disable filters"
            )
            return

        # Ensure exactly 3 lines shown (pad with blanks if needed)
        v = (vals + ["", "", ""])[:3]
        msg = (
            "Loot OCR (Gold / Elixir / Dark)\n"
            f"1) {v[0]}\n"
            f"2) {v[1]}\n"
            f"3) {v[2]}\n\n"
            "Saved a loot snapshot in your Screenshots folder."
        )
        messagebox.showinfo("Loot OCR", msg)

# ---------------- Run GUI ----------------
if __name__ == "__main__":
    app = ControlPanel()
    app.mainloop()
