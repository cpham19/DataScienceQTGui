import pandas as pd, os
from selection_dialog import SelectionFeaturesDialog
from tableview import TableView
from algorithm_groupboxes import KNearestNeighborsGroupBox, DecisionTreeGroupBox, RandomForestGroupBox, LinearRegressionGroupBox, LogisticRegressionGroupBox, LinearSVMGroupBox, MLPGroupBox
from progress_dialog import ProgressDialog
from PyQt5.QtWidgets import QApplication, QTabWidget, QVBoxLayout, QGroupBox, QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QUrl

class MainTabWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.initUi()

    def initUi(self):
        self.setTabsClosable(True)
        self.setCurrentIndex(-1)
        self.tabBar().setMovable(True)
        self.tabCloseRequested.connect(self.closeTab)
        self.setStyleSheet("""
                        QTabWidget::pane {
                            border: 1px solid black;
                            background: white;
                        } 

                        QTabBar::tab {
                            height: 50px;
                            width: 300px;
                            font-size: 14px;
                            font-weight: bold;
                            color: black;
                        }

                        QTabBar::tab:selected {
                            background-color: rgba(49,49,49, 0.5);
                            color:white;
                        }
                """)

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

        # Dialog for user to select features and label
        progressDialog = ProgressDialog(None, None, 0, 100, self)
        progressDialog.setWindowTitle("Reading " + filename + "...")
        progressDialog.setValue(0)
        progressDialog.show()
        QApplication.processEvents()

        selectionFeaturesDialog = SelectionFeaturesDialog(path)

        progressDialog.setWindowTitle("Done")
        progressDialog.setValue(100)
        progressDialog.close()
        QApplication.processEvents()
        QApplication.beep()

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
        tabGroupBox = AlgorithmGroupBox(path, features, label)

        pathToNewCSVIcon = os.path.dirname(os.path.realpath(__file__)) + "\\assets" + "\\csv_icon.png"
        self.addTab(tabGroupBox, QIcon(pathToNewCSVIcon), filename)
        self.setCurrentWidget(tabGroupBox)

    def closeTab(self, currentIndex):
        tabWidgetToClose = self.widget(currentIndex)
        tabWidgetToClose.deleteLater()
        self.removeTab(currentIndex)

# Used for creating a new tab, contains the table of the CSV data and the machine learning algorithms
class AlgorithmGroupBox(QGroupBox):
    def __init__(self, path, features, label):
        super().__init__()
        self.path = path
        self.features = features
        self.label = label

        self.df = pd.read_csv(path, sep=',')
        self.X = self.df[self.features]
        self.y = self.df[self.label[0]]
        self.columns = self.features + self.label
        self.df = self.df[self.columns]
        self.data = self.df.to_numpy().tolist()
        self.initUI()

    def initUI(self):
        tabLayout = QVBoxLayout()
        self.tabTableView = TableView(self.data, self.columns)
        # Tab Widget
        tabWidget = TabWidget(self.X, self.y)
        tabWidget.setStyleSheet(""" 
                QTabBar::tab {
                    height: 25px;
                    width: 150px;
                    font-size: 10px;
                    color: black;
                }

                QTabBar::tab:selected {
                    background-color: rgba(49,49,49, 0.7);
                    color:white;
                    border: 1px solid orange;
                }
        """)
        tabLayout.addWidget(self.tabTableView)
        tabLayout.addWidget(tabWidget)
        self.setLayout(tabLayout)

# An Inner QTabWidget that has different tabs for each machine learning algorithm
class TabWidget(QTabWidget):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.initUI()

    def initUI(self):
        self.setTabsClosable(False)
        self.setCurrentIndex(-1)
        self.setStyleSheet("QTabBar::tab { height: 25px; width: 150px;}")
        self.createTabs()

    def createTabs(self):
        self.algorithms = ["K-Nearest Neighbors", "Decision Tree", "Random Forest", "Linear Regression", "Logistic Regression", "Linear SVC", "Multilayer Perceptron"]

        for algorithm in self.algorithms:
            pathToIcon = os.path.dirname(os.path.realpath(__file__)) + "\\assets" + "\\" + algorithm + ".png"
            groupBox = None

            if algorithm == "K-Nearest Neighbors":
                groupBox = KNearestNeighborsGroupBox(self.X, self.y)
            elif algorithm == "Decision Tree":
                groupBox = DecisionTreeGroupBox(self.X, self.y)
            elif algorithm == "Random Forest":
                groupBox = RandomForestGroupBox(self.X, self.y)
            elif algorithm == "Linear Regression":
                groupBox = LinearRegressionGroupBox(self.X, self.y)
            elif algorithm == "Logistic Regression":
                groupBox = LogisticRegressionGroupBox(self.X, self.y)
            elif algorithm == "Linear SVC":
                groupBox = LinearSVMGroupBox(self.X, self.y)
            elif algorithm == "Multilayer Perceptron":
                groupBox = MLPGroupBox(self.X, self.y)

            # groupBox = ComputeGroupBox(algorithm, self.X, self.y)
            self.addTab(groupBox, QIcon(pathToIcon), algorithm)
