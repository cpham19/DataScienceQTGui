import os, numpy as np, pandas as pd
from functools import partial

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QTabWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QPushButton, QMainWindow, QGroupBox, QMenuBar, QMenu, QAction, QDialog, QComboBox, QDialogButtonBox, QFileDialog, QTableView, QTableWidget, QAbstractItemView, QHeaderView, QCheckBox, QRadioButton
from PyQt5.QtCore import Qt, QUrl, QVariant, QAbstractItemModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem

class SelectionFeaturesDialog(QDialog):
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path, sep=',')
        self.columns = list(self.df.columns.values)
        self.data = self.df.to_numpy().tolist()
        self.initUI()

    def initUI(self):
        # Dialog stuff
        self.setWindowTitle("Select features")
        self.setWindowFlags(Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.resize(1280, 720)

        # Outer layout to contain the tableview and the groupbox for selections
        layout = QVBoxLayout()

        # Datatable
        self.tableView = TableView(self.data, self.columns)

        # Groupbox to split into two groupboxes for selections
        groupbox = QGroupBox("Select your options")
        groupboxLayout = QHBoxLayout()
        groupbox.setLayout(groupboxLayout)

        featureGroupbox = QGroupBox("Select your feature columns")
        featureGroupboxLayout = QVBoxLayout()
        featureGroupbox.setLayout(featureGroupboxLayout)

        labelGroupbox = QGroupBox("Select your label")
        labelGroupboxLayout = QVBoxLayout()
        labelGroupbox.setLayout(labelGroupboxLayout)

        groupboxLayout.addWidget(featureGroupbox)
        groupboxLayout.addWidget(labelGroupbox)

        self.arrayOfFeatureCheckboxes = []
        self.arrayOfLabelRadioButtons = []
        for column in self.columns:
            checkbox = QCheckBox(column)
            radioButton = QRadioButton(column)

            checkbox.stateChanged.connect(partial(self.changeCheckboxState, checkbox))
            radioButton.toggled.connect(partial(self.changeRadioButtonState, radioButton))

            self.arrayOfFeatureCheckboxes.append(checkbox)
            self.arrayOfLabelRadioButtons.append(radioButton)

            featureGroupboxLayout.addWidget(checkbox)
            labelGroupboxLayout.addWidget(radioButton)


        layout.addWidget(self.tableView)
        layout.addWidget(groupbox)

        self.setLayout(layout)

    # Invoke when a checkbox is clicked
    def changeCheckboxState(self, checkbox):
        if (checkbox.isChecked() == True):
            for radioButton in self.arrayOfLabelRadioButtons:
                if (radioButton.text() == checkbox.text()):
                    radioButton.setDisabled(True)
        elif (checkbox.isChecked() == False):
            for radioButton in self.arrayOfLabelRadioButtons:
                if (radioButton.text() == checkbox.text()):
                    radioButton.setDisabled(False)

    # Invoke when a radio button is clicked
    def changeRadioButtonState(self, radioButton):
        if (radioButton.isChecked() == True):
            for checkbox in self.arrayOfFeatureCheckboxes:
                if (radioButton.text() == checkbox.text()):
                    checkbox.setDisabled(True)
                else:
                    checkbox.setDisabled(False)


class TableView(QTableView):
    def __init__(self, data, headers):
        super().__init__()
        self.data = data
        self.headers = headers
        self.initUI()

    def initUI(self):
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.verticalHeader().setDefaultSectionSize(50)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        tableModel = TableModel(self.data, self.headers)
        self.setModel(tableModel)

# Used for creating a new tab, contains the table of the CSV data and the machine learning algorithms
class GroupBox(QGroupBox):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.initUI()

    def initUI(self):
        tabLayout = QVBoxLayout()

        # Datatable
        self.df = pd.read_csv(self.path, sep=',')
        self.headers = list(self.df.columns.values)
        self.data = self.df.to_numpy().tolist()
        self.tabTableView = TableView(self.data, self.headers)

        # Tab Widget
        tabWidget = TabWidget()
        tabLayout.addWidget(self.tabTableView)
        tabLayout.addWidget(tabWidget)
        self.setLayout(tabLayout)

class TableModel(QStandardItemModel):
    def __init__(self, data, headers):
        super().__init__()
        self.data = data
        self.headers = headers
        self.setupData()

    def rowCount(self, parent):
        return len(self.data)

    def columnCount(self, parent):
        return len(self.data[0])

    def headerData(self, col, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[col]
            elif orientation == Qt.Vertical:
                return col
        return None

    def setupData(self):
        rowLength = len(self.data)
        colLength = len(self.data[0])
        for row in range(rowLength):
            for column in range(colLength):
                item = QStandardItem(str(self.data[row][column]))
                self.setItem(row, column, item)


# An Inner QTabWidget that has different tabs for each machine learning algorithm
class TabWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setTabsClosable(False)
        self.setCurrentIndex(-1)
        self.setStyleSheet("QTabBar::tab { height: 25px; width: 150px;}")

        self.createTabOne()
        self.createTabTwo()
        self.createTabThree()

    def createTabOne(self):
        groupBox = QGroupBox("Decision Tree")
        layout = QGridLayout()
        groupBox.setLayout(layout)
        self.addTab(groupBox, "Decision Tree")

    def createTabTwo(self):
        groupBox = QGroupBox("Random Forest")
        layout = QGridLayout()
        groupBox.setLayout(layout)
        self.addTab(groupBox, "Random Forest")

    def createTabThree(self):
        groupBox = QGroupBox("K-Nearest Neighbors")
        layout = QGridLayout()
        groupBox.setLayout(layout)
        self.addTab(groupBox, "K-Nearest Neighbors")


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

        newAction = QAction("Create a new tab", self)
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
        # File Dialog
        currentDirectory = os.path.dirname(os.path.realpath(__file__)) + "\csv"
        path = QFileDialog.getOpenFileName(self, "Open CSV File", currentDirectory, "Comma-Separated Values File (*.csv)")[0]
        if not path:
            return
        filename = QUrl.fromLocalFile(path).fileName()

        selectionFeaturesDialog = SelectionFeaturesDialog(path)
        selectionFeaturesDialog.exec_()

        tabGroupBox = GroupBox(path)
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