from PyQt5.QtWidgets import QSplashScreen, QApplication, QProgressBar, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, QObject
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QPen, QConicalGradient, QPainterPath
import sys
import math


class SplashScreen(QSplashScreen):
    def __init__(self):
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor("#00171f"))
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.FramelessWindowHint)

        self._progress = 0
        self._arc_angle = 0

        # Timer for progress animation (5 seconds)
        self._step = 0
        self._total_steps = 100
        self.progress_timer = QTimer()
        self.progress_timer.setInterval(50)  # 50ms * 100 steps = 5 seconds
        self.progress_timer.timeout.connect(self._update_progress)

        # Timer for arc animation
        self.arc_timer = QTimer()
        self.arc_timer.setInterval(30)
        self.arc_timer.timeout.connect(self._update_arc)

        # Dot animation
        self._dot_phase = 0
        self.dot_timer = QTimer()
        self.dot_timer.setInterval(400)
        self.dot_timer.timeout.connect(self._update_dots)
        self._active_dot = 0

        # Messages shown during loading
        self._messages = [
            "Sobhan Shirvanloo"
        ]
        self._current_msg = 0

    def start_animation(self):
        self.progress_timer.start()
        self.arc_timer.start()
        self.dot_timer.start()

    def _update_progress(self):
        self._step += 1
        self._progress = int((self._step / self._total_steps) * 100)
        self._current_msg = min(self._step // 20, len(self._messages) - 1)
        self.repaint()
        if self._step >= self._total_steps:
            self.progress_timer.stop()

    def _update_arc(self):
        self._arc_angle += 3
        if self._arc_angle > 360:
            self._arc_angle = 0
        self.repaint()

    def _update_dots(self):
        self._active_dot = (self._active_dot + 1) % 3
        self.repaint()

    def drawContents(self, painter: QPainter):
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor("#00171f"))

        cx, cy = w // 2, h // 2 - 40

        # Outer ring
        pen = QPen(QColor("#003459"), 5)
        painter.setPen(pen)
        painter.drawEllipse(cx - 45, cy - 45, 90, 90)

        # Progress arc
        span = int(self._progress * 3.6 * 16)  # Qt uses 1/16 degree units
        pen2 = QPen(QColor("#00a8e8"), 5)
        pen2.setCapStyle(Qt.RoundCap)
        painter.setPen(pen2)
        painter.drawArc(cx - 45, cy - 45, 90, 90, 90 * 16, -span)

        # Triangle icon inside circle
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#00a8e8"))
        path = QPainterPath()
        path.moveTo(cx, cy - 18)
        path.lineTo(cx + 16, cy + 10)
        path.lineTo(cx - 16, cy + 10)
        path.closeSubpath()
        painter.drawPath(path)

        # Base rectangle
        painter.setBrush(QColor("#007ea7"))
        painter.drawRoundedRect(cx - 10, cy + 14, 20, 5, 2, 2)

        # Title
        painter.setPen(QColor("#00a8e8"))
        font = QFont("Segoe UI", 18, QFont.Medium)
        painter.setFont(font)
        painter.drawText(0, cy + 65, w, 30, Qt.AlignCenter, "Model Reconstruction")

        # Subtitle / status message
        painter.setPen(QColor("#007ea7"))
        font2 = QFont("Segoe UI", 10)
        painter.setFont(font2)
        msg = self._messages[self._current_msg]
        painter.drawText(0, cy + 90, w, 20, Qt.AlignCenter, msg.upper())

        # Progress bar background
        bar_x = cx - 130
        bar_y = cy + 125
        bar_w = 260
        bar_h = 6
        painter.setBrush(QColor("#003459"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 3, 3)

        # Progress bar fill
        fill_w = int(bar_w * self._progress / 100)
        if fill_w > 0:
            painter.setBrush(QColor("#00a8e8"))
            painter.drawRoundedRect(bar_x, bar_y, fill_w, bar_h, 3, 3)

        # Percentage text
        painter.setPen(QColor("#007ea7"))
        font3 = QFont("Segoe UI", 9)
        painter.setFont(font3)
        painter.drawText(0, bar_y + 12, w, 20, Qt.AlignCenter, f"{self._progress}%")

        # Animated dots
        dot_y = bar_y + 40
        dot_colors = ["#003459", "#003459", "#003459"]
        dot_colors[self._active_dot] = "#00a8e8"
        for i, color in enumerate(dot_colors):
            dx = cx - 12 + i * 12
            painter.setBrush(QColor(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(dx, dot_y, 8, 8)


def create_splash_and_show(main_window_class):
    """
    Usage:
        from splash_screen import create_splash_and_show
        from main_window import MainWindow

        app = QApplication(sys.argv)
        window = create_splash_and_show(MainWindow)
        sys.exit(app.exec_())
    """
    app = QApplication.instance() or QApplication(sys.argv)

    splash = SplashScreen()
    splash.show()
    splash.start_animation()
    app.processEvents()

    main_win = main_window_class()

    def finish_loading():
        splash.arc_timer.stop()
        splash.dot_timer.stop()

        # Fade out over 5 seconds using opacity via window flags workaround
        fade_steps = 50
        fade_interval = 100  # ms -> 5 seconds total

        counter = [0]
        def fade_step():
            counter[0] += 1
            alpha = int(255 * (1 - counter[0] / fade_steps))
            px = QPixmap(splash.size())
            px.fill(QColor(0, 23, 31, max(0, alpha)))
            # Simply close after full fade
            if counter[0] >= fade_steps:
                fade_timer.stop()
                splash.finish(main_win)
                main_win.show()

        fade_timer = QTimer()
        fade_timer.setInterval(fade_interval)
        fade_timer.timeout.connect(fade_step)
        fade_timer.start()

    # After 5 seconds loading + 5 seconds fade
    QTimer.singleShot(5000, finish_loading)

    return main_win, app


# ── Standalone demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from PyQt5.QtWidgets import QMainWindow

    class DummyMain(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Model Reconstruction")
            self.resize(1182, 881)
            self.setStyleSheet("background-color:#00171f;")

    app = QApplication(sys.argv)
    window, _ = create_splash_and_show(DummyMain)
    sys.exit(app.exec_())