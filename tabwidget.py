import pandas as pd, os
from tableview import TableView
from PyQt5.QtWidgets import QApplication, QTabWidget, QVBoxLayout, QGridLayout, QGroupBox, QHBoxLayout, QFormLayout, QSpinBox, QLabel, QPushButton, QSpinBox, QComboBox,QDoubleSpinBox, QProgressDialog
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp, Qt
# Scikit Learn library
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import scale, LabelEncoder

# Used for creating a new tab, contains the table of the CSV data and the machine learning algorithms
class GroupBox(QGroupBox):
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
        tabLayout.addWidget(self.tabTableView)
        tabLayout.addWidget(tabWidget)
        self.setLayout(tabLayout)


# Compute GroupBox
class ComputeGroupBox(QGroupBox):
    def __init__(self, name, X, y):
        super().__init__()
        self.setTitle(name)
        self.algorithm = name
        self.X = X
        self.y = y

        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Groupbox to contain the line edits for parameters
        parameterGroupBox = QGroupBox("Parameters")
        parameterGroupBox.setFixedWidth(300)
        parameterGroupBoxLayout = QFormLayout()
        parameterGroupBox.setLayout(parameterGroupBoxLayout)

        # Base line edits that are used by every algorithm
        self.testSizeSpinBox = QDoubleSpinBox()
        self.testSizeSpinBox.setDecimals(2)
        self.testSizeSpinBox.setSingleStep(0.05)
        self.testSizeSpinBox.setMinimum(0.05)
        self.testSizeSpinBox.setMaximum(0.95)
        self.testSizeSpinBox.setValue(0.3)
        self.testSizeSpinBox.setFixedWidth(100)

        self.randomStateSpinBox = QSpinBox()
        self.randomStateSpinBox.setMinimum(1)
        self.randomStateSpinBox.setMaximum(100)
        self.randomStateSpinBox.setValue(3)
        self.randomStateSpinBox.setFixedWidth(100)

        self.cvSpinBox = QSpinBox()
        self.cvSpinBox.setMinimum(2)
        self.cvSpinBox.setMaximum(20)
        self.cvSpinBox.setValue(10)
        self.cvSpinBox.setFixedWidth(100)

        parameterGroupBoxLayout.addRow(QLabel("test_size"), self.testSizeSpinBox)
        parameterGroupBoxLayout.addRow(QLabel("random_state"), self.randomStateSpinBox)
        parameterGroupBoxLayout.addRow(QLabel("cross_validation"), self.cvSpinBox)

        # Base columns are test_size, random_state, cross_validation, accuracy, and accuracy(cv)
        self.data = [['', '', '', '', '']]
        self.columns = ["test_size", "random_state", "cross_validation"]

        if self.algorithm == "K-Nearest Neighbors":
            self.numberOfNeighborsSpinBox = QSpinBox()
            self.numberOfNeighborsSpinBox.setMinimum(1)
            self.numberOfNeighborsSpinBox.setMaximum(100)
            self.numberOfNeighborsSpinBox.setValue(3)
            self.numberOfNeighborsSpinBox.setFixedWidth(100)
            parameterGroupBoxLayout.addRow(QLabel("n_neighbors"), self.numberOfNeighborsSpinBox)
            self.columns.append("n_neighbors")
            self.data[0].append('')
        elif self.algorithm == "Random Forest":
            self.numberOfEstimatorsSpinBox = QSpinBox()
            self.numberOfEstimatorsSpinBox.setMinimum(1)
            self.numberOfEstimatorsSpinBox.setMaximum(100)
            self.numberOfEstimatorsSpinBox.setValue(19)
            self.numberOfEstimatorsSpinBox.setFixedWidth(100)
            parameterGroupBoxLayout.addRow(QLabel("n_estimators"), self.numberOfEstimatorsSpinBox)
            self.data[0].append('')
            self.columns.append("n_estimators")
        elif self.algorithm == "Logistic Regression":
            self.solverComboBox = QComboBox()
            self.solverComboBox.addItems(["lbfgs", "newton-cg", "liblinear", "sag", "saga"])
            self.solverComboBox.setCurrentText("lbfgs")
            self.solverComboBox.setFixedWidth(100)
            parameterGroupBoxLayout.addRow(QLabel("solver"), self.solverComboBox)
            self.data[0].append('')
            self.columns.append("solver")
        elif self.algorithm == "Multilayer Perceptron":
            self.numberOfMaxIterationsSpinBox = QSpinBox()
            self.numberOfMaxIterationsSpinBox.setSingleStep(100)
            self.numberOfMaxIterationsSpinBox.setMinimum(1)
            self.numberOfMaxIterationsSpinBox.setMaximum(100000)
            self.numberOfMaxIterationsSpinBox.setValue(1000)
            self.numberOfMaxIterationsSpinBox.setFixedWidth(100)
            parameterGroupBoxLayout.addRow(QLabel("max_iter"), self.numberOfMaxIterationsSpinBox)
            self.columns.append("max_iter")
            self.data[0].append('')

            self.alphaSpinBox = QDoubleSpinBox()
            self.alphaSpinBox.setDecimals(5)
            self.alphaSpinBox.setValue(0.005)
            self.alphaSpinBox.setSingleStep(0.005)
            self.alphaSpinBox.setMinimum(0.00001)
            self.alphaSpinBox.setMaximum(10000.000)
            self.alphaSpinBox.setFixedWidth(100)
            parameterGroupBoxLayout.addRow(QLabel("alpha"), self.alphaSpinBox)
            self.columns.append("alpha")
            self.data[0].append('')

            self.numberOfHiddenLayerSizesSpinBox = QSpinBox()
            self.numberOfHiddenLayerSizesSpinBox.setValue(3)
            self.numberOfHiddenLayerSizesSpinBox.setFixedWidth(100)
            parameterGroupBoxLayout.addRow(QLabel("hidden_layer_sizes"), self.numberOfHiddenLayerSizesSpinBox)
            self.columns.append("hidden_layer_sizes")
            self.data[0].append('')

        self.computeButton = QPushButton("Compute")
        self.computeButton.setFixedWidth(175)

        parameterGroupBoxLayout.addRow(self.computeButton)
        self.computeButton.clicked.connect(self.compute)

        resultsGroupBox = QGroupBox("Results")
        resultsGroupBoxLayout = QGridLayout()
        resultsGroupBox.setLayout(resultsGroupBoxLayout)

        self.columns.append("accuracy")
        self.columns.append("accuracy (cv)")

        self.resultsTableView = TableView(self.data, self.columns)
        self.resultTableModel = self.resultsTableView.getModel()
        # For some reason, the model doesn't like an empty list such as [], so you have to make a matrix with [['', '', '', '' ,'', '']] and then remove it, in order to add new rows
        self.resultTableModel.removeRow(0)

        resultsGroupBoxLayout.addWidget(self.resultsTableView)

        layout.addWidget(parameterGroupBox)
        layout.addWidget(resultsGroupBox)

    def compute(self):
        #Progress bar
        progressDialog = QProgressDialog(None, None, 0, 100, self)
        progressDialog.setWindowModality(Qt.WindowModal)
        progressDialog.resize(500, 40)
        progressDialog.setAutoReset(False)
        progressDialog.setAutoClose(True)
        progressDialog.setMinimum(0)
        progressDialog.setMaximum(100)
        progressDialog.setWindowTitle("Computing...")
        progressDialog.show()

        # Split the dataframe dataset.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=float(self.testSizeSpinBox.text()),
                                                            random_state=int(self.randomStateSpinBox.text()))

        self.recordParametersRow = [self.testSizeSpinBox.text(), self.randomStateSpinBox.text(), self.cvSpinBox.text()]

        # Create classifier object
        self.classifier = None
        if self.algorithm == "Decision Tree":
            # Create classifier object
            self.classifier = DecisionTreeClassifier()
        elif self.algorithm == "K-Nearest Neighbors":
            self.classifier = KNeighborsClassifier(n_neighbors=int(self.numberOfNeighborsSpinBox.text()))
            self.recordParametersRow.append(str(self.numberOfNeighborsSpinBox.text()))
        elif self.algorithm == "Random Forest":
            self.classifier = RandomForestClassifier(n_estimators=int(self.numberOfEstimatorsSpinBox.text()), bootstrap=True)
            self.recordParametersRow.append(str(self.numberOfEstimatorsSpinBox.text()))
        elif self.algorithm == "Linear Regression":
            self.classifier = LinearRegression()
        elif self.algorithm == "Logistic Regression":
            self.classifier = LogisticRegression(solver=self.solverComboBox.currentText())
            self.recordParametersRow.append(self.solverComboBox.currentText())
        elif self.algorithm == "Linear SVC":
            self.classifier = LinearSVC()
        elif self.algorithm == "Multilayer Perceptron":
            self.classifier = MLPClassifier(activation='logistic', solver='adam', max_iter=int(self.numberOfMaxIterationsSpinBox.text()),
                                            learning_rate_init=0.002, alpha=float(self.alphaSpinBox.text()),
                                            hidden_layer_sizes=(2,2))
            self.recordParametersRow.append(str(self.numberOfMaxIterationsSpinBox.text()))
            self.recordParametersRow.append(self.alphaSpinBox.text())
            self.recordParametersRow.append(self.numberOfHiddenLayerSizesSpinBox.text())

        progressDialog.setWindowTitle("Training the model...")
        progressDialog.setValue(0)
        QApplication.processEvents()

        # fit the model with the training set
        self.classifier.fit(X_train, y_train)

        progressDialog.setWindowTitle("Creating predictions to generate accuracy of predictive model...")
        progressDialog.setValue(25)
        QApplication.processEvents()

        # Predict method is used for creating a prediction on testing data
        y_predict = self.classifier.predict(X_test)

        progressDialog.setWindowTitle("Calculating accuracy...")
        progressDialog.setValue(50)
        QApplication.processEvents()

        # Accuracy of testing data on predictive model
        accuracy = accuracy_score(y_test, y_predict)

        progressDialog.setWindowTitle("Computing cross-validation...")
        progressDialog.setValue(75)
        QApplication.processEvents()

        # Add #-fold Cross Validation with Supervised Learning
        accuracy_list = cross_val_score(self.classifier, self.X, self.y, cv=int(self.cvSpinBox.text()), scoring='accuracy')

        self.recordParametersRow.append(str(accuracy))
        self.recordParametersRow.append(str(accuracy_list.mean()))
        self.resultTableModel.insertNewRow(self.recordParametersRow)

        progressDialog.setWindowTitle("Done...")
        progressDialog.setValue(100)
        QApplication.processEvents()
        progressDialog.close()


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
            groupBox = ComputeGroupBox(algorithm, self.X, self.y)
            self.addTab(groupBox, algorithm)
