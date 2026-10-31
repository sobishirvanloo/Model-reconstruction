"""Splash screen for Model Reconstruction (PyQt5-compatible).

The previous implementation relied on Qt6 / Qt Quick classes that do not
exist in PyQt5 (``QSplashScreen``, ``QPropertyAnimation``, ``QEasingCurve``,
``pyqtProperty``, ``QPainterPath``, ``QConicalGradient``) which stopped the
whole application from starting.  This version is built exclusively from
standard PyQt5 widgets so it always runs.
"""

import sys

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class SplashScreen(QWidget):
    """Frameless, always-on-top splash window.

    Shows the application title, a status message and a progress bar,
    then fades out and hands control over to the main window.
    """

    # Same palette as the light theme (modules/theme.py)
    BG_COLOR = "#ffffff"
    ACCENT = "#0a78c2"
    ACCENT_DARK = "#5d6b7b"
    TRACK = "#e1e8ef"

    NUM_STEPS = 100          # total progress steps
    STEP_INTERVAL_MS = 20    # 20 ms * 100 = 2 s minimum loading animation
    DOT_INTERVAL_MS = 400

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.resize(600, 400)
        self.setStyleSheet(
            f"""
            QWidget#splashRoot {{ background-color: {self.BG_COLOR};
                                border: 2px solid {self.TRACK};
                                border-radius: 12px; }}
            QLabel#splashTitle {{ color: {self.ACCENT};
                                  font: bold 20px "Segoe UI"; }}
            QLabel#splashMsg {{ color: {self.ACCENT_DARK};
                               font: 11px "Segoe UI"; }}
            QLabel#splashPct {{ color: {self.ACCENT_DARK};
                                font: 10px "Segoe UI"; }}
            QProgressBar {{ background-color: {self.TRACK};
                            border: none; border-radius: 3px;
                            min-height: 10px; max-height: 10px; }}
            QProgressBar::chunk {{ background-color: {self.ACCENT};
                                   border-radius: 3px; }}
            """
        )
        self.setObjectName("splashRoot")

        self._messages = ["Sobhan Shirvanloo"]
        self._step = 0
        self._progress = 0
        self._current_msg = 0
        self._dots = 0
        self._main_window = None

        title = QLabel("Model Reconstruction")
        title.setObjectName("splashTitle")
        title.setAlignment(Qt.AlignCenter)

        self.msg = QLabel(self._messages[0].upper())
        self.msg.setObjectName("splashMsg")
        self.msg.setAlignment(Qt.AlignCenter)

        self.bar = QProgressBar()
        self.bar.setRange(0, self.NUM_STEPS)
        self.bar.setTextVisible(False)

        self.pct = QLabel("0%")
        self.pct.setObjectName("splashPct")
        self.pct.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)
        layout.addWidget(title)
        layout.addWidget(self.msg)
        layout.addStretch()
        layout.addWidget(self.bar)
        layout.addWidget(self.pct)

        # Timers
        self.progress_timer = QTimer()
        self.progress_timer.setInterval(self.STEP_INTERVAL_MS)
        self.progress_timer.timeout.connect(self._update_progress)

        self.dot_timer = QTimer()
        self.dot_timer.setInterval(self.DOT_INTERVAL_MS)
        self.dot_timer.timeout.connect(self._update_dots)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    def start_animation(self):
        self.progress_timer.start()
        self.dot_timer.start()

    def _update_dots(self):
        self._dots = (self._dots + 1) % 4
        self.msg.setText(
            self._messages[self._current_msg].upper() + "." * self._dots
        )

    def _update_progress(self):
        self._step += 1
        if self._step > self.NUM_STEPS:
            self.progress_timer.stop()
            if self._main_window is not None:
                self.finish(self._main_window)
            return
        self._progress = int((self._step / self.NUM_STEPS) * 100)
        self.bar.setValue(self._step)
        self.pct.setText(f"{self._progress}%")
        if self._step % 20 == 0:
            self._current_msg = min(self._step // 20, len(self._messages) - 1)
            self._update_dots()

    # ------------------------------------------------------------------
    # Cleanup / transition
    # ------------------------------------------------------------------
    def finish_when_done(self, main_window):
        """Fade to ``main_window`` once the progress animation completes
        (immediately if it already has)."""
        if self.progress_timer.isActive():
            self._main_window = main_window
        else:
            self.finish(main_window)

    def finish(self, main_window, fade_steps=20, fade_interval=25):
        """Stop the animation, fade out, then show the main window."""
        self.progress_timer.stop()
        self.dot_timer.stop()

        counter = [0]

        def fade_step():
            counter[0] += 1
            alpha = max(0.0, 1.0 - counter[0] / fade_steps)
            self.setWindowOpacity(alpha)
            if counter[0] >= fade_steps:
                fade_timer.stop()
                self.hide()
                self.deleteLater()
                main_window.show()

        fade_timer = QTimer(self)
        fade_timer.setInterval(fade_interval)
        fade_timer.timeout.connect(fade_step)
        fade_timer.start()


def create_splash_and_show(main_window_class):
    """Show the splash, build the main window, then fade to it.

    Usage:
        from splash_screen import create_splash_and_show
        from main_window import MainWindow

        app = QApplication(sys.argv)
        window, app = create_splash_and_show(MainWindow)
        sys.exit(app.exec_())
    """
    app = QApplication.instance() or QApplication(sys.argv)

    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # Build the main window first (this blocks the event loop), then run
    # the animation so the progress bar never freezes half way.
    main_win = main_window_class()
    # Keep a reference: a parentless local widget would be garbage
    # collected on return, killing its timers (and the hand-off).
    main_win._splash = splash

    splash.start_animation()
    splash.finish_when_done(main_win)

    return main_win, app


# ----------------------------------------------------------------------
# Standalone demo
# ----------------------------------------------------------------------
if __name__ == "__main__":

    class DummyMain(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Model Reconstruction")
            self.resize(1182, 881)

    app = QApplication(sys.argv)
    window, _ = create_splash_and_show(DummyMain)
    sys.exit(app.exec_())