from PyQt5.QtWidgets import  QProgressDialog
from PyQt5.QtCore import Qt

class ProgressDialog(QProgressDialog):
    def __init__(self, title, description, minimum, maximum, parent):
        super().__init__(parent)
        self.setWindowModality(Qt.WindowModal)
        self.resize(500, 40)
        self.setAutoReset(False)
        self.setAutoClose(True)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setWindowFlags(Qt.Window| Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.setCancelButton(None)