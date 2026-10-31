import os
import sys

from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi

from modules.import_mesh import Geometry
from modules.point_processing import Processing
from modules.poisson_reconstruction import PoissonReconstruction
from modules.splash_screen import create_splash_and_show


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

    def send_point_cloud(self, data, target):
        """Hand a ``PointCloudData`` to another tab in memory and show it.

        ``target`` is ``"processing"`` or ``"reconstruction"``.
        """
        tab, page = {
            "processing": (self.processing, self.tab_2),
            "reconstruction": (self.poisson, self.tab_5),
        }[target]
        self.tabWidget.setCurrentWidget(page)
        tab.receive_data(data)


# ---------- Main ----------
if __name__ == "__main__":
    window, app = create_splash_and_show(MainWindow)
    sys.exit(app.exec_())
