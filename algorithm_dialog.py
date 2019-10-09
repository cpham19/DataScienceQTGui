from PyQt5.QtWidgets import QHBoxLayout, QDialog, QComboBox, QDialogButtonBox
from PyQt5.QtCore import Qt

# A Dialog
class AlgorithmDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Dialog stuff
        self.setWindowTitle("Select an algorithm")
        self.setWindowFlags(Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

        layout = QHBoxLayout()
        self.combobox = QComboBox()
        self.combobox.addItems(["Decision Tree", "Random Forest", "Multilayer Perception", "K-Nearest Neighbors", "Linear Regression", "Logistic Regression", "Support Vector Machine"])
        layout.addWidget(self.combobox)

        selectButtonBox = QDialogButtonBox(self)
        selectButtonBox.setOrientation(Qt.Horizontal)
        selectButtonBox.setStandardButtons(QDialogButtonBox.Ok)
        selectButtonBox.clicked.connect(self.accept)

        layout.addWidget(selectButtonBox)
        self.setLayout(layout)

    def accept(self):
        print(self.combobox.currentText())
        self.close()