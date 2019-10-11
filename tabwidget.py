import pandas as pd, os
from selection_dialog import SelectionFeaturesDialog
from tableview import TableView
from progress_dialog import ProgressDialog
from PyQt5.QtWidgets import QApplication, QTabWidget, QVBoxLayout, QGridLayout, QGroupBox, QHBoxLayout, QFormLayout, QSpinBox, QLabel, QPushButton, QSpinBox, QComboBox,QDoubleSpinBox, QProgressDialog, QTabBar, QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QUrl
from PyQt5 import QtCore
# Scikit Learn library
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import scale, LabelEncoder

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
                            background-color: rgba(49,49,49, 0.7);
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


# Compute GroupBox for Sklearn algorithms
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
        self.parameterGroupBoxLayout = QFormLayout()
        parameterGroupBox.setLayout(self.parameterGroupBoxLayout)

        # Base line edits that are used by every algorithm
        testSizeLabel = QLabel("test_size")
        testSizeLabel.setToolTip("test_size : float, int or None, optional (default=None)\nIf float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.")

        self.testSizeSpinBox = QDoubleSpinBox()
        self.testSizeSpinBox.setDecimals(2)
        self.testSizeSpinBox.setSingleStep(0.05)
        self.testSizeSpinBox.setMinimum(0.05)
        self.testSizeSpinBox.setMaximum(0.95)
        self.testSizeSpinBox.setValue(0.3)
        self.testSizeSpinBox.setFixedWidth(150)

        randomStateLabel = QLabel("random_state")
        randomStateLabel.setToolTip("random_state : int, RandomState instance or None, optional (default=None)\nIf int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.")

        self.randomStateSpinBox = QSpinBox()
        self.randomStateSpinBox.setMinimum(1)
        self.randomStateSpinBox.setMaximum(100)
        self.randomStateSpinBox.setValue(3)
        self.randomStateSpinBox.setFixedWidth(150)

        cvLabel = QLabel("cv")
        cvLabel.setToolTip("In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”: \nA model is trained using  of the folds as training data\nthe resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).")

        self.cvSpinBox = QSpinBox()
        self.cvSpinBox.setMinimum(2)
        self.cvSpinBox.setMaximum(20)
        self.cvSpinBox.setValue(10)
        self.cvSpinBox.setFixedWidth(150)

        self.parameterGroupBoxLayout.addRow(testSizeLabel, self.testSizeSpinBox)
        self.parameterGroupBoxLayout.addRow(randomStateLabel, self.randomStateSpinBox)
        if (self.algorithm != "Linear Regression"):
            self.parameterGroupBoxLayout.addRow(cvLabel, self.cvSpinBox)

        # Base columns are test_size, random_state, cross_validation, accuracy, and accuracy(cv)
        if (self.algorithm != "Linear Regression"):
            self.data = [['', '', '', '', '']]
            self.columns = ["test_size", "random_state", "cross_validation"]
        else:
            self.data = [['', '']]
            self.columns = ["test_size", "random_state"]

        if self.algorithm == "K-Nearest Neighbors":
            numberOfNeighborsLabel = QLabel("n_neighbors")
            numberOfNeighborsLabel.setToolTip("n_neighbors : int, optional (default = 5) \nNumber of neighbors to use by default for kneighbors queries.")

            self.numberOfNeighborsSpinBox = QSpinBox()
            self.numberOfNeighborsSpinBox.setMinimum(1)
            self.numberOfNeighborsSpinBox.setMaximum(100)
            self.numberOfNeighborsSpinBox.setValue(3)
            self.numberOfNeighborsSpinBox.setFixedWidth(150)
            self.parameterGroupBoxLayout.addRow(numberOfNeighborsLabel, self.numberOfNeighborsSpinBox)
            self.columns.append("n_neighbors")
            self.data[0].append('')
        elif self.algorithm == "Random Forest":
            numberOfEstimatorsLabel = QLabel("n_estimators")
            numberOfEstimatorsLabel.setToolTip("n_estimators : integer, optional (default=10) \nThe number of trees in the forest. Changed in version 0.20: The default value of n_estimators will change from 10 in version 0.20 to 100 in version 0.22.")

            self.numberOfEstimatorsSpinBox = QSpinBox()
            self.numberOfEstimatorsSpinBox.setMinimum(1)
            self.numberOfEstimatorsSpinBox.setMaximum(100)
            self.numberOfEstimatorsSpinBox.setValue(19)
            self.numberOfEstimatorsSpinBox.setFixedWidth(150)
            self.parameterGroupBoxLayout.addRow(numberOfEstimatorsLabel, self.numberOfEstimatorsSpinBox)
            self.data[0].append('')
            self.columns.append("n_estimators")
        elif self.algorithm == "Linear Regression":
            # There are no fields but we still need to add three empty columns to accomdate for coefficients, mean squared error, and r2 score
            self.data[0].append('')
            self.data[0].append('')
            self.data[0].append('')
            self.columns.append("coefficients")
            # The mean squared error
            self.columns.append("mean squared error")
            # Explained variance score: 1 is perfect prediction
            self.columns.append("variance score")
        elif self.algorithm == "Logistic Regression":
            solverLabel = QLabel("solver")
            solverLabel.setToolTip("solver : str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional (default=’liblinear’). \nAlgorithm to use in the optimization problem. \nFor small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones. \nFor multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes. \n‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty \n‘liblinear’ and ‘saga’ also handle L1 penalty \n‘saga’ also supports ‘elasticnet’ penalty \n‘liblinear’ does not handle no penalty")

            self.solverComboBox = QComboBox()
            self.solverComboBox.addItems(["lbfgs", "newton-cg", "liblinear", "sag", "saga"])
            self.solverComboBox.setCurrentText("lbfgs")
            self.solverComboBox.setFixedWidth(150)
            self.parameterGroupBoxLayout.addRow(solverLabel, self.solverComboBox)
            self.data[0].append('')
            self.columns.append("solver")
        elif self.algorithm == "Multilayer Perceptron":
            numberOfMaxIterationsLabel = QLabel("max_iter")
            numberOfMaxIterationsLabel.setToolTip("max_iter : int, optional, default 200 \nMaximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps")

            self.numberOfMaxIterationsSpinBox = QSpinBox()
            self.numberOfMaxIterationsSpinBox.setSingleStep(100)
            self.numberOfMaxIterationsSpinBox.setMinimum(1)
            self.numberOfMaxIterationsSpinBox.setMaximum(100000)
            self.numberOfMaxIterationsSpinBox.setValue(1000)
            self.numberOfMaxIterationsSpinBox.setFixedWidth(150)
            self.parameterGroupBoxLayout.addRow(numberOfMaxIterationsLabel, self.numberOfMaxIterationsSpinBox)
            self.columns.append("max_iter")
            self.data[0].append('')

            alphaLabel = QLabel("alpha")
            alphaLabel.setToolTip("alpha : float, optional, default 0.0001\nL2 penalty (regularization term) parameter.")

            self.alphaSpinBox = QDoubleSpinBox()
            self.alphaSpinBox.setDecimals(5)
            self.alphaSpinBox.setValue(0.005)
            self.alphaSpinBox.setSingleStep(0.005)
            self.alphaSpinBox.setMinimum(0.00001)
            self.alphaSpinBox.setMaximum(10000.000)
            self.alphaSpinBox.setFixedWidth(150)
            self.parameterGroupBoxLayout.addRow(QLabel("alpha"), self.alphaSpinBox)
            self.columns.append("alpha")
            self.data[0].append('')

            solverLabel = QLabel("solver")
            solverLabel.setToolTip("solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’ \nThe solver for weight optimization.\n‘lbfgs’ is an optimizer in the family of quasi-Newton methods.\n‘sgd’ refers to stochastic gradient descent.\b‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba\nNote: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, ‘lbfgs’ can converge faster and perform better.")

            self.solverMlpComboBox = QComboBox()
            self.solverMlpComboBox.addItems(["lbfgs", "sgd", "adam"])
            self.solverMlpComboBox.setCurrentText("adam")
            self.solverMlpComboBox.setFixedWidth(150)
            self.parameterGroupBoxLayout.addRow(solverLabel, self.solverMlpComboBox)
            self.data[0].append('')
            self.columns.append("solver")

            hiddenLayerSizesLabel = QLabel("number of hidden layers")
            hiddenLayerSizesLabel.setToolTip("hidden_layer_sizes : tuple, length = n_layers - 2, default (100,) \nThe ith element represents the number of neurons in the ith hidden layer.")

            self.numberOfHiddenLayerSizesSpinBox = QSpinBox()
            self.numberOfHiddenLayerSizesSpinBox.setMinimum(1)
            self.numberOfHiddenLayerSizesSpinBox.setMaximum(15)
            self.numberOfHiddenLayerSizesSpinBox.setValue(1)
            self.numberOfHiddenLayerSizesSpinBox.setFixedWidth(150)
            self.numberOfHiddenLayerSizesSpinBox.valueChanged.connect(self.hiddenLayerValueChange)
            self.parameterGroupBoxLayout.addRow(hiddenLayerSizesLabel, self.numberOfHiddenLayerSizesSpinBox)

            self.nodeLabels = []
            self.nodeSpinBoxes = []

            self.numberOfNodesSpinBox = QSpinBox()
            self.numberOfNodesSpinBox.setMinimum(1)
            self.numberOfNodesSpinBox.setValue(1)
            self.numberOfNodesSpinBox.setFixedWidth(150)

            label = QLabel("number of nodes (layer 1)")

            self.parameterGroupBoxLayout.addRow(label, self.numberOfNodesSpinBox)
            self.nodeLabels.append(label)
            self.nodeSpinBoxes.append(self.numberOfNodesSpinBox)

            self.columns.append("hidden_layer_sizes")
            self.data[0].append('')

        self.computeButton = QPushButton("Compute")
        self.computeButton.adjustSize()

        self.parameterGroupBoxLayout.addRow(self.computeButton)
        self.computeButton.clicked.connect(self.compute)

        resultsGroupBox = QGroupBox("Results")
        resultsGroupBoxLayout = QGridLayout()
        resultsGroupBox.setLayout(resultsGroupBoxLayout)

        if (self.algorithm != "Linear Regression"):
            self.columns.append("accuracy")
            self.columns.append("accuracy (cv)")

        self.resultsTableView = TableView(self.data, self.columns)
        self.resultTableModel = self.resultsTableView.getModel()
        # For some reason, the model doesn't like an empty list such as [], so you have to make a matrix with [['', '', '', '' ,'', '']] and then remove it, in order to add new rows
        self.resultTableModel.removeRow(0)

        resultsGroupBoxLayout.addWidget(self.resultsTableView)

        layout.addWidget(parameterGroupBox)
        layout.addWidget(resultsGroupBox)

    def hiddenLayerValueChange(self):
        numberOfLayers = int(self.numberOfHiddenLayerSizesSpinBox.text())
        numberOfSpinBoxes = len(self.nodeSpinBoxes)

        self.parameterGroupBoxLayout.removeWidget(self.computeButton)
        self.parameterGroupBoxLayout.removeRow(self.parameterGroupBoxLayout.rowCount() - 1)

        # Removing hidden layer sizes
        if (numberOfLayers <= numberOfSpinBoxes):
            for i in range(numberOfLayers, numberOfSpinBoxes):
                nodeLabelToRemove = self.nodeLabels.pop()
                nodeSpinBoxToRemove = self.nodeSpinBoxes.pop()
                self.parameterGroupBoxLayout.removeRow(self.parameterGroupBoxLayout.rowCount() - 1)
        else:
            # Adding hidden layer sizes
            for i in range(numberOfSpinBoxes, numberOfLayers):
                newNodeSpinBox = QSpinBox()
                newNodeSpinBox.setMinimum(1)
                newNodeSpinBox.setValue(1)
                newNodeSpinBox.setFixedWidth(150)

                label = QLabel("number of nodes (layer" + str(i + 1) + ")")
                self.parameterGroupBoxLayout.addRow(label, newNodeSpinBox)
                self.nodeLabels.append(label)
                self.nodeSpinBoxes.append(newNodeSpinBox)


        self.parameterGroupBoxLayout.addRow(self.computeButton)

    def compute(self):
        #Progress bar
        progressDialog = ProgressDialog(None, None, 0, 100, self)
        progressDialog.setWindowTitle("Computing...")
        progressDialog.show()

        # Split the dataframe dataset.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=float(self.testSizeSpinBox.text()),
                                                            random_state=int(self.randomStateSpinBox.text()))

        if (self.algorithm != "Linear Regression"):
            self.recordParametersRow = [self.testSizeSpinBox.text(), self.randomStateSpinBox.text(), self.cvSpinBox.text()]
        else:
            self.recordParametersRow = [self.testSizeSpinBox.text(), self.randomStateSpinBox.text()]

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
            # Get the tuple of nodes and hidden layer sizes
            arrayOfNodesForEachHiddenLayer = list([int(spinbox.text()) for spinbox in self.nodeSpinBoxes])
            s = [str(i) for i in arrayOfNodesForEachHiddenLayer]
            hiddenLayersAndNodes = ", ".join(s)
            tupleOfNodes = tuple(arrayOfNodesForEachHiddenLayer)


            self.classifier = MLPClassifier(activation='logistic', solver=self.solverMlpComboBox.currentText(), max_iter=int(self.numberOfMaxIterationsSpinBox.text()),
                                            learning_rate_init=0.002, alpha=float(self.alphaSpinBox.text()),
                                            hidden_layer_sizes=tupleOfNodes)

            self.recordParametersRow.append(self.solverMlpComboBox.currentText())
            self.recordParametersRow.append(self.numberOfMaxIterationsSpinBox.text())
            self.recordParametersRow.append(self.alphaSpinBox.text())
            self.recordParametersRow.append(hiddenLayersAndNodes)

        progressDialog.setWindowTitle("Training the model...")
        progressDialog.setValue(0)
        QApplication.processEvents()

        # training the model with the training set
        self.classifier.fit(X_train, y_train)

        progressDialog.setWindowTitle("Creating predictions to generate accuracy of predictive model...")
        progressDialog.setValue(25)
        QApplication.processEvents()

        # Making predictions sing the testing data
        y_predict = self.classifier.predict(X_test)

        progressDialog.setWindowTitle("Calculating accuracy...")
        progressDialog.setValue(50)
        QApplication.processEvents()

        # Accuracy of testing data on predictive model
        if self.algorithm != "Linear Regression":
            accuracy = accuracy_score(y_test, y_predict)
            self.recordParametersRow.append(str(accuracy))

            progressDialog.setWindowTitle("Computing cross-validation...")
            progressDialog.setValue(75)
            QApplication.processEvents()

            accuracy_list = cross_val_score(self.classifier, self.X, self.y, cv=int(self.cvSpinBox.text()),scoring='accuracy')
            self.recordParametersRow.append(str(accuracy_list.mean()))
        else:
            arrayOfCoefficients = ["{0:.2f}".format(i) for i in self.classifier.coef_]
            string = ", ".join(arrayOfCoefficients)
            self.recordParametersRow.append(string)
            self.recordParametersRow.append("{0:.2f}".format(mean_squared_error(y_test, y_predict)))
            self.recordParametersRow.append("{0:.2f}".format(r2_score(y_test, y_predict)))


        self.resultTableModel.insertNewRow(self.recordParametersRow)

        progressDialog.setWindowTitle("Done...")
        progressDialog.setValue(100)
        QApplication.processEvents()
        QApplication.beep()
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
            pathToIcon = os.path.dirname(os.path.realpath(__file__)) + "\\assets" + "\\" + algorithm + ".png"
            groupBox = ComputeGroupBox(algorithm, self.X, self.y)
            self.addTab(groupBox, QIcon(pathToIcon), algorithm)
