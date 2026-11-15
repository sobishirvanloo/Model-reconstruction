# MeshBench - Copyright (c) 2026 Sobhan Shirvanlou. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Loading screen and logo of MeshBench.

The splash is painted by hand (QPainter, plain PyQt5): a dark card with
a slowly drifting point field, a turning geodesic sphere that starts as
a point cloud and gains its mesh edges as loading advances - what the
program does, in one picture - the name, a status line and a glowing
progress bar.  ``app_icon`` draws the same sphere as the window icon.
"""

import math
import random
import sys

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import (
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget

from modules import branding

BG_TOP = QColor("#0b1424")
BG_BOTTOM = QColor("#11284a")
CYAN = QColor("#38d6ff")
BLUE = QColor("#2f7dff")
TEXT = QColor("#eef4fb")
MUTED = QColor("#8ea3bd")


# =========================
# Geodesic sphere (icosahedron, subdivided once)
# =========================
def _geodesic():
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [_unit(v) for v in verts]
    cache = {}

    def middle(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            va, vb = verts[a], verts[b]
            verts.append(_unit(tuple((x + y) / 2 for x, y in zip(va, vb))))
            cache[key] = len(verts) - 1
        return cache[key]

    edges = set()
    for a, b, c in faces:
        ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
        for tri in ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)):
            for i in range(3):
                p, q = tri[i], tri[(i + 1) % 3]
                edges.add((min(p, q), max(p, q)))
    return verts, sorted(edges)


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


VERTS, EDGES = _geodesic()


def _project(angle, tilt, center, radius):
    """Screen points (x, y, depth 0..1) of the turned sphere."""
    ca, sa, ct, st = math.cos(angle), math.sin(angle), math.cos(tilt), math.sin(tilt)
    points = []
    for x, y, z in VERTS:
        x, z = x * ca + z * sa, -x * sa + z * ca          # turn about Y
        y, z = y * ct - z * st, y * st + z * ct          # tilt about X
        scale = radius * (1.0 + 0.12 * z)                 # a little perspective
        points.append((center.x() + x * scale, center.y() - y * scale, (z + 1) / 2))
    return points


def paint_sphere(painter, center, radius, angle, tilt=0.42, mesh=1.0,
                 point_size=2.4):
    """The sphere as points, with its edges at ``mesh`` (0..1) strength."""
    points = _project(angle, tilt, center, radius)
    if mesh > 0:
        for a, b in EDGES:
            depth = (points[a][2] + points[b][2]) / 2
            color = QColor(CYAN)
            color.setAlphaF(max(0.0, min(1.0, mesh)) * (0.12 + 0.55 * depth))
            painter.setPen(QPen(color, 1.1))
            painter.drawLine(QPointF(points[a][0], points[a][1]),
                             QPointF(points[b][0], points[b][1]))
    painter.setPen(Qt.NoPen)
    for x, y, depth in sorted(points, key=lambda p: p[2]):
        glow = QColor(CYAN)
        glow.setAlphaF(0.10 + 0.20 * depth)
        painter.setBrush(glow)
        painter.drawEllipse(QPointF(x, y), point_size * 2.4, point_size * 2.4)
        core = QColor(TEXT if depth > 0.55 else CYAN)
        core.setAlphaF(0.45 + 0.55 * depth)
        painter.setBrush(core)
        painter.drawEllipse(QPointF(x, y), point_size * (0.7 + 0.5 * depth),
                            point_size * (0.7 + 0.5 * depth))


def app_icon():
    """Window / taskbar icon: the sphere on a rounded blue tile."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        tile = QRectF(0, 0, size, size).adjusted(size * 0.04, size * 0.04,
                                                 -size * 0.04, -size * 0.04)
        gradient = QLinearGradient(tile.topLeft(), tile.bottomRight())
        gradient.setColorAt(0, BG_BOTTOM)
        gradient.setColorAt(1, BG_TOP)
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(tile, size * 0.22, size * 0.22)
        paint_sphere(painter, tile.center(), size * 0.32, 0.6, mesh=1.0,
                     point_size=max(0.6, size / 70))
        painter.end()
        icon.addPixmap(pixmap)
    return icon


# =========================
# Splash
# =========================
class SplashScreen(QWidget):
    """Frameless, always-on-top loading window that fades into the main
    window."""

    WIDTH, HEIGHT = 760, 420
    NUM_STEPS = 100           # progress steps
    STEP_INTERVAL_MS = 20     # 20 ms x 100 = 2 s minimum
    FRAME_MS = 16             # ~60 frames per second
    MESSAGES = (
        "Starting the 3D engine",
        "Loading the reconstruction algorithms",
        "Preparing the comparison methods",
        "Warming up the computation helper",
        "Ready",
    )

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            | Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle(branding.APP_NAME)
        self.setWindowIcon(app_icon())
        self.resize(self.WIDTH, self.HEIGHT)
        screen = QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            self.move(area.center().x() - self.WIDTH // 2,
                      area.center().y() - self.HEIGHT // 2)
        self._step = 0
        self._frame = 0
        self._main_window = None
        # a calm field of background points (fixed pattern, drifting)
        rng = random.Random(7)
        self._field = [(rng.uniform(0, self.WIDTH), rng.uniform(0, self.HEIGHT),
                        rng.uniform(0.5, 1.6)) for _ in range(90)]

        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(self.STEP_INTERVAL_MS)
        self.progress_timer.timeout.connect(self._update_progress)
        self.frame_timer = QTimer(self)
        self.frame_timer.setInterval(self.FRAME_MS)
        self.frame_timer.timeout.connect(self._next_frame)
        self.frame_timer.start()

    # ---------- animation ----------
    @property
    def progress(self):
        return self._step / self.NUM_STEPS

    @property
    def message(self):
        index = min(int(self.progress * len(self.MESSAGES)), len(self.MESSAGES) - 1)
        return self.MESSAGES[index]

    def start_animation(self):
        self.progress_timer.start()

    def _next_frame(self):
        self._frame += 1
        self.update()

    def _update_progress(self):
        self._step += 1
        if self._step >= self.NUM_STEPS:
            self._step = self.NUM_STEPS
            self.progress_timer.stop()
            if self._main_window is not None:
                self.finish(self._main_window)

    # ---------- painting ----------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        card = QRectF(6, 6, self.width() - 12, self.height() - 12)
        clip = QPainterPath()
        clip.addRoundedRect(card, 18, 18)

        background = QLinearGradient(card.topLeft(), card.bottomRight())
        background.setColorAt(0, BG_TOP)
        background.setColorAt(1, BG_BOTTOM)
        painter.fillPath(clip, background)
        painter.setClipPath(clip)

        # soft light behind the sphere
        sphere_center = QPointF(card.left() + 215, card.center().y() - 6)
        light = QRadialGradient(sphere_center, 230)
        light.setColorAt(0, QColor(56, 214, 255, 70))
        light.setColorAt(1, QColor(56, 214, 255, 0))
        painter.fillRect(card, light)

        # drifting point field
        painter.setPen(Qt.NoPen)
        for x, y, speed in self._field:
            px = (x + self._frame * 0.15 * speed) % card.width() + card.left()
            dot = QColor(MUTED)
            dot.setAlphaF(0.18 * speed)
            painter.setBrush(dot)
            painter.drawEllipse(QPointF(px, y + card.top()), 1.1, 1.1)

        # the sphere: points first, the mesh appears with the progress
        paint_sphere(painter, sphere_center, 118, self._frame * 0.012,
                     mesh=min(1.0, 0.08 + self.progress * 1.1))

        # name: "Mesh" white, "Bench" cyan
        left = card.left() + 395
        room = card.right() - 34 - left
        title_font = QFont(self.font())
        title_font.setBold(True)
        title_font.setLetterSpacing(QFont.AbsoluteSpacing, 0.5)
        for size in range(52, 30, -2):              # the largest that fits
            title_font.setPixelSize(size)
            painter.setFont(title_font)
            if painter.fontMetrics().horizontalAdvance(branding.APP_NAME) <= room:
                break
        painter.setPen(TEXT)
        painter.drawText(QPointF(left, card.top() + 165), "Mesh")
        mesh_width = painter.fontMetrics().horizontalAdvance("Mesh")
        bench = QLinearGradient(left + mesh_width, 0, left + mesh_width + 160, 0)
        bench.setColorAt(0, CYAN)
        bench.setColorAt(1, BLUE)
        painter.setPen(QPen(bench, 1))
        painter.drawText(QPointF(left + mesh_width, card.top() + 165), "Bench")

        small = QFont(self.font())
        small.setPixelSize(14)
        painter.setFont(small)
        painter.setPen(MUTED)
        painter.drawText(QRectF(left, card.top() + 182, card.right() - left - 30, 40),
                         Qt.TextWordWrap, branding.TAGLINE)

        # status + progress bar
        status = QFont(self.font())
        status.setPixelSize(12)
        painter.setFont(status)
        dots = "." * (1 + (self._frame // 20) % 3) if self.progress < 1 else ""
        painter.setPen(TEXT)
        painter.drawText(QPointF(left, card.bottom() - 92), self.message + dots)
        painter.setPen(MUTED)
        percent = f"{int(self.progress * 100)}%"
        painter.drawText(QRectF(left, card.bottom() - 106, card.right() - left - 30, 20),
                         Qt.AlignRight | Qt.AlignVCenter, percent)

        track = QRectF(left, card.bottom() - 78, card.right() - left - 30, 6)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 28))
        painter.drawRoundedRect(track, 3, 3)
        if self.progress > 0:
            done = QRectF(track.left(), track.top(),
                          max(6.0, track.width() * self.progress), track.height())
            fill = QLinearGradient(done.topLeft(), done.topRight())
            fill.setColorAt(0, BLUE)
            fill.setColorAt(1, CYAN)
            glow = QColor(CYAN)
            glow.setAlpha(60)
            painter.setBrush(glow)
            painter.drawRoundedRect(done.adjusted(-2, -3, 2, 3), 6, 6)
            painter.setBrush(fill)
            painter.drawRoundedRect(done, 3, 3)

        painter.setFont(status)
        painter.setPen(MUTED)
        painter.drawText(QPointF(left, card.bottom() - 30), f"Version {branding.VERSION}")

        painter.setClipping(False)
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(card, 18, 18)
        painter.end()

    # ---------- hand-off ----------
    def finish_when_done(self, main_window):
        """Fade to ``main_window`` once the progress is complete
        (immediately if it already is)."""
        if self.progress_timer.isActive():
            self._main_window = main_window
        else:
            self.finish(main_window)

    def finish(self, main_window, fade_steps=20, fade_interval=25):
        """Fade out, then show the main window."""
        self.progress_timer.stop()
        counter = [0]

        def fade_step():
            counter[0] += 1
            self.setWindowOpacity(max(0.0, 1.0 - counter[0] / fade_steps))
            if counter[0] >= fade_steps:
                fade_timer.stop()
                self.frame_timer.stop()
                self.hide()
                self.deleteLater()
                main_window.show()

        fade_timer = QTimer(self)
        fade_timer.setInterval(fade_interval)
        fade_timer.timeout.connect(fade_step)
        fade_timer.start()


def create_splash_and_show(main_window_class):
    """Show the splash, build the main window, then fade to it."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setWindowIcon(app_icon())

    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # Build the main window first (this blocks the event loop), then run
    # the progress so the bar never freezes half way.
    main_win = main_window_class()
    # Keep a reference: a parentless local widget would be garbage
    # collected on return, killing its timers (and the hand-off).
    main_win._splash = splash

    splash.start_animation()
    splash.finish_when_done(main_win)
    return main_win, app


if __name__ == "__main__":
    class DummyMain(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(branding.APP_NAME)
            self.resize(1182, 881)

    app = QApplication(sys.argv)
    window, _ = create_splash_and_show(DummyMain)
    sys.exit(app.exec_())
