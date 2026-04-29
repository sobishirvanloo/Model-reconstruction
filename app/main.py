import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from modules.import_mesh import Geometry
from modules.point_processing import Processing
from modules.poisson_reconstruction import PoissonReconstruction  


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/main.ui", self)

        # Initialize Geometry
        self.geometry = Geometry(self)
        self.processing = Processing(self)
        self.poisson = PoissonReconstruction(self)  # ← اضافه کن


# ---------- Main ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())