import os, pandas as pd
from selection_dialog import SelectionFeaturesDialog
from tabwidget import GroupBox
from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget, QGridLayout, QMainWindow, QAction, QFileDialog, QMessageBox
from PyQt5.QtCore import QUrl

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.counter = 1
        self.initUI()

    def initUI(self):
        # Main Window stuff
        self.statusBar().showMessage('Ready')
        self.resize(1280, 720)
        self.setWindowTitle('Data Science GUI')

        # Menu bar
        menuBar = self.menuBar()

        # File menu option
        menuFile = menuBar.addMenu("File")

        newAction = QAction("Open a CSV File", self)
        newAction.triggered.connect(self.newTab)
        menuFile.addAction(newAction)

        # Main Widget
        mainWidget = QWidget(self)
        layout = QGridLayout()
        mainWidget.setLayout(layout)
        self.setCentralWidget(mainWidget)

        # Tab Widget
        self.mainTabWidget = QTabWidget(mainWidget)
        self.mainTabWidget.setTabsClosable(True)
        self.mainTabWidget.setCurrentIndex(-1)
        self.mainTabWidget.tabCloseRequested.connect(self.closeTab)
        self.mainTabWidget.setStyleSheet("QTabBar::tab { height: 50px; width: 300px;}")
        layout.addWidget(self.mainTabWidget)

    def newTab(self, currentIndex):
        # File Dialog (uses csv directory)
        directory = os.path.dirname(os.path.realpath(__file__)) + "\csv"
        if not os.path.exists(directory):
            os.makedirs(directory)

        QApplication.beep()
        # Path to Selected CSV File
        path = QFileDialog.getOpenFileName(self, "Open CSV File", directory, "Comma-Separated Values File (*.csv)")[0]
        if not path:
            return
        filename = QUrl.fromLocalFile(path).fileName()

        QApplication.beep()
        # Dialog for user to select features and label
        selectionFeaturesDialog = SelectionFeaturesDialog(path)
        dialogCode = selectionFeaturesDialog.exec_()


        # If Dialog was cancelled, then avoid opening a new tab
        if dialogCode == 0:
            return
        elif (len(selectionFeaturesDialog.getFeatures()) == 0 or len(selectionFeaturesDialog.getLabel()) == 0):
            QApplication.beep()
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Please open the CSV file again select at least one feature and select the label!')
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        features = selectionFeaturesDialog.getFeatures()
        label = selectionFeaturesDialog.getLabel()
        tabGroupBox = GroupBox(path, features, label)
        self.mainTabWidget.addTab(tabGroupBox, filename)
        self.counter += 1
        self.mainTabWidget.setCurrentWidget(tabGroupBox)

    def closeTab(self, currentIndex):
        tabWidgetToClose = self.mainTabWidget.widget(currentIndex)
        tabWidgetToClose.deleteLater()
        self.mainTabWidget.removeTab(currentIndex)


def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()