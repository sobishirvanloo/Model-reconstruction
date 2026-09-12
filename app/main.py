import os
import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi

from modules.import_mesh import Geometry
from modules.point_processing import Processing
from modules.poisson_reconstruction import PoissonReconstruction
from modules.splash_screen import SplashScreen


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Resolve the .ui file relative to this file so the app can be
        # launched from anywhere (not only from inside app/).
        ui_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ui", "main.ui"
        )
        loadUi(ui_path, self)

        self.geometry = Geometry(self)
        self.processing = Processing(self)
        self.poisson = PoissonReconstruction(self)


# ---------- Main ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    splash = SplashScreen()
    splash.show()
    splash.start_animation()
    app.processEvents()

    window = MainWindow()

    # After the 5 s loading animation, fade into the main window.
    QTimer.singleShot(5000, lambda: splash.finish(window))

    sys.exit(app.exec_())