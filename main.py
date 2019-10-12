import os
from tabwidget import MainTabWidget
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QMainWindow, QAction, QStyleFactory
from PyQt5.QtGui import QIcon

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
        pathToWindowIcon = os.path.dirname(os.path.realpath(__file__)) + "\\assets" + "\\data_science_icon.png"
        self.setWindowIcon(QIcon(pathToWindowIcon))

        self.setStyleSheet("""
                QMenuBar {
                    background-color: rgba(49,49,49, 0.5);
                    color: rgb(255,255,255);
                    border: 1px solid #000;
                    font-size:14px;
                    font-weight: bold;
                }

                QMenuBar::item {
                    background-color: rgba(49,49,49, 0.5);
                    color: rgb(255,255,255);
                }

                QMenuBar::item::selected {
                    background-color: rgb(255,127,80);
                }

                QMenu {
                    background-color: rgba(49,49,49, 0.5);
                    color: rgb(255,255,255);
                    border: 1px solid #000;
                    font-size:14px;
                    font-weight: bold;         
                }

                QMenu::item::selected {
                    background-color: rgb(255,127,80);
                }
        """)

        # Menu bar
        menuBar = self.menuBar()

        # File menu option
        menuFile = menuBar.addMenu("File")

        pathToNewCSVIcon = os.path.dirname(os.path.realpath(__file__)) + "\\assets" + "\\csv_icon.png"
        newAction = QAction(QIcon(pathToNewCSVIcon),"Open a CSV File", self)
        newAction.setShortcut("Ctrl+N")
        menuFile.addAction(newAction)

        # Main Widget
        mainWidget = QWidget(self)
        layout = QGridLayout()
        mainWidget.setLayout(layout)
        self.setCentralWidget(mainWidget)

        # Tab Widget
        self.mainTabWidget = MainTabWidget()
        newAction.triggered.connect(self.mainTabWidget.newTab)
        layout.addWidget(self.mainTabWidget)

def main():
    app = QApplication([])
    print(QStyleFactory.keys())
    app.setStyle('')

    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()