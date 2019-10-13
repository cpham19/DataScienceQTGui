from PyQt5.QtWidgets import QApplication, QGridLayout, QGroupBox, QHBoxLayout, QFormLayout, QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QMessageBox
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

from progress_dialog import ProgressDialog
from tableview import TableView

class KNearestNeighborsGroupBox(QGroupBox):
    def __init__(self, X, y):
        super().__init__()
        self.name = "K-Nearest Neighbors"
        self.setTitle(self.name)
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
        testSizeLabel.setToolTip(
            "test_size : float, int or None, optional (default=None)\nIf float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.")

        self.testSizeSpinBox = QDoubleSpinBox()
        self.testSizeSpinBox.setDecimals(2)
        self.testSizeSpinBox.setSingleStep(0.05)
        self.testSizeSpinBox.setMinimum(0.05)
        self.testSizeSpinBox.setMaximum(0.95)
        self.testSizeSpinBox.setValue(0.3)
        self.testSizeSpinBox.adjustSize()

        randomStateLabel = QLabel("random_state")
        randomStateLabel.setToolTip("random_state : int, RandomState instance or None, optional (default=None)\nIf int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.")

        self.randomStateSpinBox = QSpinBox()
        self.randomStateSpinBox.setMinimum(1)
        self.randomStateSpinBox.setMaximum(100)
        self.randomStateSpinBox.setValue(3)
        self.randomStateSpinBox.adjustSize()

        cvLabel = QLabel("cross_validation")
        cvLabel.setToolTip("In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”: \nA model is trained using  of the folds as training data\nthe resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).")

        self.cvSpinBox = QSpinBox()
        self.cvSpinBox.setMinimum(2)
        self.cvSpinBox.setMaximum(20)
        self.cvSpinBox.setValue(10)
        self.cvSpinBox.adjustSize()

        numberOfNeighborsLabel = QLabel("n_neighbors")
        numberOfNeighborsLabel.setToolTip("n_neighbors : int, optional (default = 5) \nNumber of neighbors to use by default for kneighbors queries.")

        self.numberOfNeighborsSpinBox = QSpinBox()
        self.numberOfNeighborsSpinBox.setMinimum(1)
        self.numberOfNeighborsSpinBox.setMaximum(100)
        self.numberOfNeighborsSpinBox.setValue(3)
        self.numberOfNeighborsSpinBox.adjustSize()

        self.computeButton = QPushButton("Compute")
        self.computeButton.adjustSize()
        self.computeButton.clicked.connect(self.compute)

        self.parameterGroupBoxLayout.addRow(testSizeLabel, self.testSizeSpinBox)
        self.parameterGroupBoxLayout.addRow(randomStateLabel, self.randomStateSpinBox)
        self.parameterGroupBoxLayout.addRow(cvLabel, self.cvSpinBox)
        self.parameterGroupBoxLayout.addRow(numberOfNeighborsLabel, self.numberOfNeighborsSpinBox)
        self.parameterGroupBoxLayout.addRow(self.computeButton)

        resultsGroupBox = QGroupBox("Results")
        resultsGroupBoxLayout = QGridLayout()
        resultsGroupBox.setLayout(resultsGroupBoxLayout)

        # Base columns are test_size, random_state, cross_validation, accuracy, and accuracy(cv)
        columns = ["test_size", "random_state", "cross_validation", "n_neighbors", "accuracy", "accuracy (cv)"]
        self.resultsTableView = TableView(None, columns)
        self.resultTableModel = self.resultsTableView.getModel()
        # For some reason, the model doesn't like an empty list such as [], so you have to make a matrix with [['', '', '', '' ,'', '']] and then remove it, in order to add new rows
        self.resultTableModel.removeRow(0)

        resultsGroupBoxLayout.addWidget(self.resultsTableView)

        layout.addWidget(parameterGroupBox)
        layout.addWidget(resultsGroupBox)

    def compute(self):
        # Progress bar
        progressDialog = ProgressDialog(None, None, 0, 100, self)
        progressDialog.setWindowTitle("Computing...")
        progressDialog.show()

        # Split the dataframe dataset.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=float(self.testSizeSpinBox.text()),
                                                            random_state=int(self.randomStateSpinBox.text()))

        # Create classifier object
        self.classifier = KNeighborsClassifier(n_neighbors=int(self.numberOfNeighborsSpinBox.text()))

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
        accuracy = accuracy_score(y_test, y_predict)

        progressDialog.setWindowTitle("Computing cross-validation...")
        progressDialog.setValue(75)
        QApplication.processEvents()

        accuracy_list = cross_val_score(self.classifier, self.X, self.y, cv=int(self.cvSpinBox.text()),
                                        scoring='accuracy')

        self.recordParametersRow = [self.testSizeSpinBox.text(), self.randomStateSpinBox.text(), self.cvSpinBox.text(), self.numberOfNeighborsSpinBox.text(), str(accuracy), str(accuracy_list.mean())]
        self.resultTableModel.insertNewRow(self.recordParametersRow)

        progressDialog.setWindowTitle("Done...")
        progressDialog.setValue(100)
        QApplication.processEvents()
        QApplication.beep()
        progressDialog.close()

        formattedAccuracy = "{0:.2f}".format(accuracy * 100)
        formattedCrossValidationAccuracy = "{0:.2f}".format(accuracy_list.mean() * 100)

        resultsMessageBox = QMessageBox()
        resultsMessageBox.setWindowTitle("Results of " + self.name + " classifier")
        resultsMessageBox.setText("The predictive model's accuracy is: " + formattedAccuracy + "%\n" +
                                   "The cross validation's accuracy is: " + formattedCrossValidationAccuracy + "%");
        resultsMessageBox.setStyleSheet("""
               QMessageBox {
                   background-color: rgb(255, 229, 204);
                   color: #fffff8;
                   font-size: 12px;
                   font-weight: bold;
               }
           """)
        resultsMessageBox.exec()

class DecisionTreeGroupBox(QGroupBox):
    def __init__(self, X, y):
        super().__init__()
        self.name = "Decision Tree"
        self.setTitle(self.name)
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
        testSizeLabel.setToolTip(
            "test_size : float, int or None, optional (default=None)\nIf float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.")

        self.testSizeSpinBox = QDoubleSpinBox()
        self.testSizeSpinBox.setDecimals(2)
        self.testSizeSpinBox.setSingleStep(0.05)
        self.testSizeSpinBox.setMinimum(0.05)
        self.testSizeSpinBox.setMaximum(0.95)
        self.testSizeSpinBox.setValue(0.3)
        self.testSizeSpinBox.adjustSize()

        randomStateLabel = QLabel("random_state")
        randomStateLabel.setToolTip(
            "random_state : int, RandomState instance or None, optional (default=None)\nIf int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.")

        self.randomStateSpinBox = QSpinBox()
        self.randomStateSpinBox.setMinimum(1)
        self.randomStateSpinBox.setMaximum(100)
        self.randomStateSpinBox.setValue(3)
        self.randomStateSpinBox.adjustSize()

        cvLabel = QLabel("cross_validaiton")
        cvLabel.setToolTip(
            "In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”: \nA model is trained using  of the folds as training data\nthe resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).")

        self.cvSpinBox = QSpinBox()
        self.cvSpinBox.setMinimum(2)
        self.cvSpinBox.setMaximum(20)
        self.cvSpinBox.setValue(10)
        self.cvSpinBox.adjustSize()

        self.computeButton = QPushButton("Compute")
        self.computeButton.adjustSize()
        self.computeButton.clicked.connect(self.compute)

        self.parameterGroupBoxLayout.addRow(testSizeLabel, self.testSizeSpinBox)
        self.parameterGroupBoxLayout.addRow(randomStateLabel, self.randomStateSpinBox)
        self.parameterGroupBoxLayout.addRow(cvLabel, self.cvSpinBox)
        self.parameterGroupBoxLayout.addRow(self.computeButton)

        resultsGroupBox = QGroupBox("Results")
        resultsGroupBoxLayout = QGridLayout()
        resultsGroupBox.setLayout(resultsGroupBoxLayout)

        # Base columns are test_size, random_state, cross_validation, accuracy, and accuracy(cv)
        columns = ["test_size", "random_state", "cross_validation", "accuracy", "accuracy (cv)"]
        self.resultsTableView = TableView(None, columns)
        self.resultTableModel = self.resultsTableView.getModel()
        # For some reason, the model doesn't like an empty list such as [], so you have to make a matrix with [['', '', '', '' ,'', '']] and then remove it, in order to add new rows
        self.resultTableModel.removeRow(0)

        resultsGroupBoxLayout.addWidget(self.resultsTableView)

        layout.addWidget(parameterGroupBox)
        layout.addWidget(resultsGroupBox)
        self.parameterGroupBoxLayout.addRow(self.computeButton)

    def compute(self):
        # Progress bar
        progressDialog = ProgressDialog(None, None, 0, 100, self)
        progressDialog.setWindowTitle("Computing...")
        progressDialog.show()

        # Split the dataframe dataset.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=float(self.testSizeSpinBox.text()),
                                                            random_state=int(self.randomStateSpinBox.text()))

        # Create classifier object
        self.classifier = DecisionTreeClassifier()
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

        accuracy = accuracy_score(y_test, y_predict)

        progressDialog.setWindowTitle("Computing cross-validation...")
        progressDialog.setValue(75)
        QApplication.processEvents()

        accuracy_list = cross_val_score(self.classifier, self.X, self.y, cv=int(self.cvSpinBox.text()),
                                        scoring='accuracy')

        self.recordParametersRow = [self.testSizeSpinBox.text(), self.randomStateSpinBox.text(), self.cvSpinBox.text(), str(accuracy), str(accuracy_list.mean())]
        self.resultTableModel.insertNewRow(self.recordParametersRow)

        progressDialog.setWindowTitle("Done...")
        progressDialog.setValue(100)
        QApplication.processEvents()
        QApplication.beep()
        progressDialog.close()

        formattedAccuracy = "{0:.2f}".format(accuracy * 100)
        formattedCrossValidationAccuracy = "{0:.2f}".format(accuracy_list.mean() * 100)

        resultsMessageBox = QMessageBox()
        resultsMessageBox.setWindowTitle("Results of " + self.name + " classifier")
        resultsMessageBox.setText("The predictive model's accuracy is: " + formattedAccuracy + "%\n" +
                                   "The cross validation's accuracy is: " + formattedCrossValidationAccuracy + "%");
        resultsMessageBox.setStyleSheet("""
               QMessageBox {
                   background-color: rgb(255, 229, 204);
                   color: #fffff8;
                   font-size: 12px;
                   font-weight: bold;
               }
           """)
        resultsMessageBox.exec()

# Compute GroupBox for Sklearn algorithms
class RandomForestGroupBox(QGroupBox):
    def __init__(self, X, y):
        super().__init__()
        self.name = "Random Forest"
        self.setTitle(self.name)
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
        self.testSizeSpinBox.adjustSize()

        randomStateLabel = QLabel("random_state")
        randomStateLabel.setToolTip("random_state : int, RandomState instance or None, optional (default=None)\nIf int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.")

        self.randomStateSpinBox = QSpinBox()
        self.randomStateSpinBox.setMinimum(1)
        self.randomStateSpinBox.setMaximum(100)
        self.randomStateSpinBox.setValue(3)
        self.randomStateSpinBox.adjustSize()

        cvLabel = QLabel("cross_validation")
        cvLabel.setToolTip("In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”: \nA model is trained using  of the folds as training data\nthe resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).")

        self.cvSpinBox = QSpinBox()
        self.cvSpinBox.setMinimum(2)
        self.cvSpinBox.setMaximum(20)
        self.cvSpinBox.setValue(10)
        self.cvSpinBox.adjustSize()

        numberOfEstimatorsLabel = QLabel("n_estimators")
        numberOfEstimatorsLabel.setToolTip("n_estimators : integer, optional (default=10) \nThe number of trees in the forest. Changed in version 0.20: The default value of n_estimators will change from 10 in version 0.20 to 100 in version 0.22.")

        self.numberOfEstimatorsSpinBox = QSpinBox()
        self.numberOfEstimatorsSpinBox.setMinimum(1)
        self.numberOfEstimatorsSpinBox.setMaximum(100)
        self.numberOfEstimatorsSpinBox.setValue(19)
        self.numberOfEstimatorsSpinBox.adjustSize()

        self.computeButton = QPushButton("Compute")
        self.computeButton.adjustSize()
        self.computeButton.clicked.connect(self.compute)

        self.parameterGroupBoxLayout.addRow(testSizeLabel, self.testSizeSpinBox)
        self.parameterGroupBoxLayout.addRow(randomStateLabel, self.randomStateSpinBox)
        self.parameterGroupBoxLayout.addRow(cvLabel, self.cvSpinBox)
        self.parameterGroupBoxLayout.addRow(numberOfEstimatorsLabel, self.numberOfEstimatorsSpinBox)
        self.parameterGroupBoxLayout.addRow(self.computeButton)

        resultsGroupBox = QGroupBox("Results")
        resultsGroupBoxLayout = QGridLayout()
        resultsGroupBox.setLayout(resultsGroupBoxLayout)

        # Base columns are test_size, random_state, cross_validation, accuracy, and accuracy(cv)
        columns = ["test_size", "random_state", "cross_validation", "n_estimators", "accuracy", "accuracy (cv)"]

        self.resultsTableView = TableView(None, columns)
        self.resultTableModel = self.resultsTableView.getModel()
        # For some reason, the model doesn't like an empty list such as [], so you have to make a matrix with [['', '', '', '' ,'', '']] and then remove it, in order to add new rows
        self.resultTableModel.removeRow(0)

        resultsGroupBoxLayout.addWidget(self.resultsTableView)

        layout.addWidget(parameterGroupBox)
        layout.addWidget(resultsGroupBox)


    def compute(self):
        #Progress bar
        progressDialog = ProgressDialog(None, None, 0, 100, self)
        progressDialog.setWindowTitle("Computing...")
        progressDialog.show()

        # Split the dataframe dataset.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=float(self.testSizeSpinBox.text()),
                                                            random_state=int(self.randomStateSpinBox.text()))

        # Create classifier object
        self.classifier = RandomForestClassifier(n_estimators=int(self.numberOfEstimatorsSpinBox.text()), bootstrap=True)

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

        accuracy = accuracy_score(y_test, y_predict)

        progressDialog.setWindowTitle("Computing cross-validation...")
        progressDialog.setValue(75)
        QApplication.processEvents()

        accuracy_list = cross_val_score(self.classifier, self.X, self.y, cv=int(self.cvSpinBox.text()), scoring='accuracy')

        self.recordParametersRow = [self.testSizeSpinBox.text(), self.randomStateSpinBox.text(), self.cvSpinBox.text(), self.numberOfEstimatorsSpinBox.text(), str(accuracy), str(accuracy_list.mean())]
        self.resultTableModel.insertNewRow(self.recordParametersRow)

        progressDialog.setWindowTitle("Done...")
        progressDialog.setValue(100)
        QApplication.processEvents()
        QApplication.beep()
        progressDialog.close()

        formattedAccuracy = "{0:.2f}".format(accuracy * 100)
        formattedCrossValidationAccuracy = "{0:.2f}".format(accuracy_list.mean() * 100)

        resultsMessageBox = QMessageBox()
        resultsMessageBox.setWindowTitle("Results of " + self.name + " classifier")
        resultsMessageBox.setText("The predictive model's accuracy is: " + formattedAccuracy + "%\n" +
                                   "The cross validation's accuracy is: " + formattedCrossValidationAccuracy + "%");
        resultsMessageBox.setStyleSheet("""
               QMessageBox {
                   background-color: rgb(255, 229, 204);
                   color: #fffff8;
                   font-size: 12px;
                   font-weight: bold;
               }
           """)
        resultsMessageBox.exec()

# Compute GroupBox for Sklearn algorithms
class LinearRegressionGroupBox(QGroupBox):
    def __init__(self, X, y):
        super().__init__()
        self.name = "Linear Regression"
        self.setTitle(self.name)
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
        self.testSizeSpinBox.adjustSize()

        randomStateLabel = QLabel("random_state")
        randomStateLabel.setToolTip("random_state : int, RandomState instance or None, optional (default=None)\nIf int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.")

        self.randomStateSpinBox = QSpinBox()
        self.randomStateSpinBox.setMinimum(1)
        self.randomStateSpinBox.setMaximum(100)
        self.randomStateSpinBox.setValue(3)
        self.randomStateSpinBox.adjustSize()

        cvLabel = QLabel("cross_validation")
        cvLabel.setToolTip("In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”: \nA model is trained using  of the folds as training data\nthe resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).")

        self.cvSpinBox = QSpinBox()
        self.cvSpinBox.setMinimum(2)
        self.cvSpinBox.setMaximum(20)
        self.cvSpinBox.setValue(10)
        self.cvSpinBox.adjustSize()

        self.computeButton = QPushButton("Compute")
        self.computeButton.adjustSize()
        self.computeButton.clicked.connect(self.compute)

        self.parameterGroupBoxLayout.addRow(testSizeLabel, self.testSizeSpinBox)
        self.parameterGroupBoxLayout.addRow(randomStateLabel, self.randomStateSpinBox)
        self.parameterGroupBoxLayout.addRow(self.computeButton)

        resultsGroupBox = QGroupBox("Results")
        resultsGroupBoxLayout = QGridLayout()
        resultsGroupBox.setLayout(resultsGroupBoxLayout)

        # Base columns
        columns = ["test_size", "random_state", "coefficients", "mean squared error", "variance score"]
        self.resultsTableView = TableView(None, columns)
        self.resultTableModel = self.resultsTableView.getModel()
        # For some reason, the model doesn't like an empty list such as [], so you have to make a matrix with [['', '', '', '' ,'', '']] and then remove it, in order to add new rows
        self.resultTableModel.removeRow(0)

        resultsGroupBoxLayout.addWidget(self.resultsTableView)

        layout.addWidget(parameterGroupBox)
        layout.addWidget(resultsGroupBox)

    def compute(self):
        #Progress bar
        progressDialog = ProgressDialog(None, None, 0, 100, self)
        progressDialog.setWindowTitle("Computing...")
        progressDialog.show()

        # Split the dataframe dataset.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=float(self.testSizeSpinBox.text()),
                                                            random_state=int(self.randomStateSpinBox.text()))

        # Create classifier object
        self.classifier = LinearRegression()

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

        # Accuracy of prediction
        arrayOfCoefficients = ["{0:.2f}".format(i) for i in self.classifier.coef_]
        stringOfCoefficients = ", ".join(arrayOfCoefficients)
        meanSquaredError = "{0:.2f}".format(mean_squared_error(y_test, y_predict))
        r2Score = "{0:.2f}".format(r2_score(y_test, y_predict))

        self.recordParametersRow = [self.testSizeSpinBox.text(), self.randomStateSpinBox.text(), stringOfCoefficients, meanSquaredError, r2Score]
        self.resultTableModel.insertNewRow(self.recordParametersRow)

        progressDialog.setWindowTitle("Done...")
        progressDialog.setValue(100)
        QApplication.processEvents()
        QApplication.beep()
        progressDialog.close()

        resultsMessageBox = QMessageBox()
        resultsMessageBox.setWindowTitle("Results of " + self.name + " classifier")
        resultsMessageBox.setText("The predictive model's coefficients are : " + stringOfCoefficients + "\n" +
                                   "The mean squared error is: " + meanSquaredError + "\n" +
                                   " The r-squared score is: " + r2Score);
        resultsMessageBox.setStyleSheet("""
               QMessageBox {
                   background-color: rgb(255, 229, 204);
                   color: #fffff8;
                   font-size: 12px;
                   font-weight: bold;
               }
           """)
        resultsMessageBox.exec()

# Compute GroupBox for Sklearn algorithms
class LogisticRegressionGroupBox(QGroupBox):
    def __init__(self, X, y):
        super().__init__()
        self.name = "Logistic Regression"
        self.setTitle(self.name)
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
        self.testSizeSpinBox.adjustSize()

        randomStateLabel = QLabel("random_state")
        randomStateLabel.setToolTip("random_state : int, RandomState instance or None, optional (default=None)\nIf int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.")

        self.randomStateSpinBox = QSpinBox()
        self.randomStateSpinBox.setMinimum(1)
        self.randomStateSpinBox.setMaximum(100)
        self.randomStateSpinBox.setValue(3)
        self.randomStateSpinBox.adjustSize()

        cvLabel = QLabel("cross_validation")
        cvLabel.setToolTip("In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”: \nA model is trained using  of the folds as training data\nthe resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).")

        self.cvSpinBox = QSpinBox()
        self.cvSpinBox.setMinimum(2)
        self.cvSpinBox.setMaximum(20)
        self.cvSpinBox.setValue(10)
        self.cvSpinBox.adjustSize()

        solverLabel = QLabel("solver")
        solverLabel.setToolTip("solver : str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional (default=’liblinear’). \nAlgorithm to use in the optimization problem. \nFor small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones. \nFor multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes. \n‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty \n‘liblinear’ and ‘saga’ also handle L1 penalty \n‘saga’ also supports ‘elasticnet’ penalty \n‘liblinear’ does not handle no penalty")

        self.solverComboBox = QComboBox()
        self.solverComboBox.addItems(["lbfgs", "newton-cg", "liblinear", "sag", "saga"])
        self.solverComboBox.setCurrentText("lbfgs")
        self.solverComboBox.adjustSize()

        self.computeButton = QPushButton("Compute")
        self.computeButton.adjustSize()
        self.computeButton.clicked.connect(self.compute)

        self.parameterGroupBoxLayout.addRow(testSizeLabel, self.testSizeSpinBox)
        self.parameterGroupBoxLayout.addRow(randomStateLabel, self.randomStateSpinBox)
        self.parameterGroupBoxLayout.addRow(cvLabel, self.cvSpinBox)
        self.parameterGroupBoxLayout.addRow(solverLabel, self.solverComboBox)
        self.parameterGroupBoxLayout.addRow(self.computeButton)

        resultsGroupBox = QGroupBox("Results")
        resultsGroupBoxLayout = QGridLayout()
        resultsGroupBox.setLayout(resultsGroupBoxLayout)

        # Base columns are test_size, random_state, cross_validation, accuracy, and accuracy(cv)
        columns = ["test_size", "random_state", "cross_validation", "solver", "accuracy", "accuracy (cv)"]
        self.resultsTableView = TableView(None, columns)
        self.resultTableModel = self.resultsTableView.getModel()
        # For some reason, the model doesn't like an empty list such as [], so you have to make a matrix with [['', '', '', '' ,'', '']] and then remove it, in order to add new rows
        self.resultTableModel.removeRow(0)

        resultsGroupBoxLayout.addWidget(self.resultsTableView)

        layout.addWidget(parameterGroupBox)
        layout.addWidget(resultsGroupBox)

    def compute(self):
        #Progress bar
        progressDialog = ProgressDialog(None, None, 0, 100, self)
        progressDialog.setWindowTitle("Computing...")
        progressDialog.show()

        # Split the dataframe dataset.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=float(self.testSizeSpinBox.text()),
                                                            random_state=int(self.randomStateSpinBox.text()))

        # Create classifier object
        self.classifier = LogisticRegression(solver=self.solverComboBox.currentText())

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
        accuracy = accuracy_score(y_test, y_predict)

        print("HELLO")

        progressDialog.setWindowTitle("Computing cross-validation...")
        progressDialog.setValue(75)
        QApplication.processEvents()

        accuracy_list = cross_val_score(self.classifier, self.X, self.y, cv=int(self.cvSpinBox.text()))

        self.recordParametersRow = [self.testSizeSpinBox.text(), self.randomStateSpinBox.text(), self.cvSpinBox.text(), self.solverComboBox.currentText(), str(accuracy), str(accuracy_list.mean())]
        self.resultTableModel.insertNewRow(self.recordParametersRow)

        progressDialog.setWindowTitle("Done...")
        progressDialog.setValue(100)
        QApplication.processEvents()
        QApplication.beep()
        progressDialog.close()

        formattedAccuracy = "{0:.2f}".format(accuracy * 100)
        formattedCrossValidationAccuracy = "{0:.2f}".format(accuracy_list.mean() * 100)

        resultsMessageBox = QMessageBox()
        resultsMessageBox.setWindowTitle("Results of " + self.name + " classifier")
        resultsMessageBox.setText("The predictive model's accuracy is: " + formattedAccuracy + "%\n" +
                                   "The cross validation's accuracy is: " + formattedCrossValidationAccuracy + "%");
        resultsMessageBox.setStyleSheet("""
               QMessageBox {
                   background-color: rgb(255, 229, 204);
                   color: #fffff8;
                   font-size: 12px;
                   font-weight: bold;
               }
           """)
        resultsMessageBox.exec()

# Compute GroupBox for Sklearn algorithms
class LinearSVMGroupBox(QGroupBox):
    def __init__(self, X, y):
        super().__init__()
        self.name = "Linear SVC"
        self.setTitle(self.name)
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
        self.testSizeSpinBox.adjustSize()

        randomStateLabel = QLabel("random_state")
        randomStateLabel.setToolTip("random_state : int, RandomState instance or None, optional (default=None)\nIf int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.")

        self.randomStateSpinBox = QSpinBox()
        self.randomStateSpinBox.setMinimum(1)
        self.randomStateSpinBox.setMaximum(100)
        self.randomStateSpinBox.setValue(3)
        self.randomStateSpinBox.adjustSize()

        cvLabel = QLabel("cross_validation")
        cvLabel.setToolTip("In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”: \nA model is trained using  of the folds as training data\nthe resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).")

        self.cvSpinBox = QSpinBox()
        self.cvSpinBox.setMinimum(2)
        self.cvSpinBox.setMaximum(20)
        self.cvSpinBox.setValue(10)
        self.cvSpinBox.adjustSize()

        self.computeButton = QPushButton("Compute")
        self.computeButton.adjustSize()
        self.computeButton.clicked.connect(self.compute)

        self.parameterGroupBoxLayout.addRow(testSizeLabel, self.testSizeSpinBox)
        self.parameterGroupBoxLayout.addRow(randomStateLabel, self.randomStateSpinBox)
        self.parameterGroupBoxLayout.addRow(cvLabel, self.cvSpinBox)
        self.parameterGroupBoxLayout.addRow(self.computeButton)

        resultsGroupBox = QGroupBox("Results")
        resultsGroupBoxLayout = QGridLayout()
        resultsGroupBox.setLayout(resultsGroupBoxLayout)

        # Base columns are test_size, random_state, cross_validation, accuracy, and accuracy(cv)
        columns = ["test_size", "random_state", "cross_validation", "accuracy", "acurracy (cv)"]
        self.resultsTableView = TableView(None, columns)
        self.resultTableModel = self.resultsTableView.getModel()
        # For some reason, the model doesn't like an empty list such as [], so you have to make a matrix with [['', '', '', '' ,'', '']] and then remove it, in order to add new rows
        self.resultTableModel.removeRow(0)

        resultsGroupBoxLayout.addWidget(self.resultsTableView)

        layout.addWidget(parameterGroupBox)
        layout.addWidget(resultsGroupBox)

    def compute(self):
        #Progress bar
        progressDialog = ProgressDialog(None, None, 0, 100, self)
        progressDialog.setWindowTitle("Computing...")
        progressDialog.show()

        # Split the dataframe dataset.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=float(self.testSizeSpinBox.text()),
                                                            random_state=int(self.randomStateSpinBox.text()))

        # Create classifier object
        self.classifier = LinearSVC()

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
        accuracy = accuracy_score(y_test, y_predict)

        progressDialog.setWindowTitle("Computing cross-validation...")
        progressDialog.setValue(75)
        QApplication.processEvents()

        accuracy_list = cross_val_score(self.classifier, self.X, self.y, cv=int(self.cvSpinBox.text()),
                                        scoring='accuracy')

        self.recordParametersRow = [self.testSizeSpinBox.text(), self.randomStateSpinBox.text(), self.cvSpinBox.text(), str(accuracy), str(accuracy_list.mean())]
        self.resultTableModel.insertNewRow(self.recordParametersRow)

        progressDialog.setWindowTitle("Done...")
        progressDialog.setValue(100)
        QApplication.processEvents()
        QApplication.beep()
        progressDialog.close()

        formattedAccuracy = "{0:.2f}".format(accuracy * 100)
        formattedCrossValidationAccuracy = "{0:.2f}".format(accuracy_list.mean() * 100)

        resultsMessageBox = QMessageBox()
        resultsMessageBox.setWindowTitle("Results of " + self.name + " classifier")
        resultsMessageBox.setText("The predictive model's accuracy is: " + formattedAccuracy + "%\n" +
                                   "The cross validation's accuracy is: " + formattedCrossValidationAccuracy + "%");
        resultsMessageBox.setStyleSheet("""
               QMessageBox {
                   background-color: rgb(255, 229, 204);
                   color: #fffff8;
                   font-size: 12px;
                   font-weight: bold;
               }
           """)
        resultsMessageBox.exec()

# Compute GroupBox for Sklearn algorithms
class MLPGroupBox(QGroupBox):
    def __init__(self, X, y):
        super().__init__()
        self.name = "Multilayer Perceptron"
        self.setTitle(self.name)
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
        self.testSizeSpinBox.adjustSize()

        randomStateLabel = QLabel("random_state")
        randomStateLabel.setToolTip("random_state : int, RandomState instance or None, optional (default=None)\nIf int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.")

        self.randomStateSpinBox = QSpinBox()
        self.randomStateSpinBox.setMinimum(1)
        self.randomStateSpinBox.setMaximum(100)
        self.randomStateSpinBox.setValue(3)
        self.randomStateSpinBox.adjustSize()

        cvLabel = QLabel("cv")
        cvLabel.setToolTip("In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”: \nA model is trained using  of the folds as training data\nthe resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).")

        self.cvSpinBox = QSpinBox()
        self.cvSpinBox.setMinimum(2)
        self.cvSpinBox.setMaximum(20)
        self.cvSpinBox.setValue(10)
        self.cvSpinBox.adjustSize()

        numberOfMaxIterationsLabel = QLabel("max_iter")
        numberOfMaxIterationsLabel.setToolTip("max_iter : int, optional, default 200 \nMaximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps")

        self.numberOfMaxIterationsSpinBox = QSpinBox()
        self.numberOfMaxIterationsSpinBox.setSingleStep(100)
        self.numberOfMaxIterationsSpinBox.setMinimum(1)
        self.numberOfMaxIterationsSpinBox.setMaximum(100000)
        self.numberOfMaxIterationsSpinBox.setValue(1000)
        self.numberOfMaxIterationsSpinBox.adjustSize()

        alphaLabel = QLabel("alpha")
        alphaLabel.setToolTip("alpha : float, optional, default 0.0001\nL2 penalty (regularization term) parameter.")

        self.alphaSpinBox = QDoubleSpinBox()
        self.alphaSpinBox.setDecimals(5)
        self.alphaSpinBox.setValue(0.005)
        self.alphaSpinBox.setSingleStep(0.005)
        self.alphaSpinBox.setMinimum(0.00001)
        self.alphaSpinBox.setMaximum(10000.000)
        self.alphaSpinBox.adjustSize()

        solverLabel = QLabel("solver")
        solverLabel.setToolTip("solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’ \nThe solver for weight optimization.\n‘lbfgs’ is an optimizer in the family of quasi-Newton methods.\n‘sgd’ refers to stochastic gradient descent.\b‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba\nNote: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, ‘lbfgs’ can converge faster and perform better.")

        self.solverMlpComboBox = QComboBox()
        self.solverMlpComboBox.addItems(["lbfgs", "sgd", "adam"])
        self.solverMlpComboBox.setCurrentText("adam")
        self.solverMlpComboBox.adjustSize()

        hiddenLayerSizesLabel = QLabel("number of hidden layers")
        hiddenLayerSizesLabel.setToolTip("hidden_layer_sizes : tuple, length = n_layers - 2, default (100,) \nThe ith element represents the number of neurons in the ith hidden layer.")

        self.numberOfHiddenLayerSizesSpinBox = QSpinBox()
        self.numberOfHiddenLayerSizesSpinBox.setMinimum(1)
        self.numberOfHiddenLayerSizesSpinBox.setMaximum(15)
        self.numberOfHiddenLayerSizesSpinBox.setValue(1)
        self.numberOfHiddenLayerSizesSpinBox.adjustSize()
        self.numberOfHiddenLayerSizesSpinBox.valueChanged.connect(self.hiddenLayerValueChange)

        self.nodeLabels = []
        self.nodeSpinBoxes = []

        self.numberOfNodesSpinBox = QSpinBox()
        self.numberOfNodesSpinBox.setMinimum(1)
        self.numberOfNodesSpinBox.setValue(1)
        self.numberOfNodesSpinBox.adjustSize()

        nodeLabel = QLabel("number of nodes (layer 1)")

        self.nodeLabels.append(nodeLabel)
        self.nodeSpinBoxes.append(self.numberOfNodesSpinBox)

        self.computeButton = QPushButton("Compute")
        self.computeButton.adjustSize()
        self.computeButton.clicked.connect(self.compute)

        self.parameterGroupBoxLayout.addRow(testSizeLabel, self.testSizeSpinBox)
        self.parameterGroupBoxLayout.addRow(randomStateLabel, self.randomStateSpinBox)
        self.parameterGroupBoxLayout.addRow(cvLabel, self.cvSpinBox)
        self.parameterGroupBoxLayout.addRow(numberOfMaxIterationsLabel, self.numberOfMaxIterationsSpinBox)
        self.parameterGroupBoxLayout.addRow(alphaLabel, self.alphaSpinBox)
        self.parameterGroupBoxLayout.addRow(solverLabel, self.solverMlpComboBox)
        self.parameterGroupBoxLayout.addRow(hiddenLayerSizesLabel, self.numberOfHiddenLayerSizesSpinBox)
        self.parameterGroupBoxLayout.addRow(nodeLabel, self.numberOfNodesSpinBox)
        self.parameterGroupBoxLayout.addRow(self.computeButton)

        resultsGroupBox = QGroupBox("Results")
        resultsGroupBoxLayout = QGridLayout()
        resultsGroupBox.setLayout(resultsGroupBoxLayout)

        columns = ["test_size", "random_state", "cross_validaiton", "max_iter", "alpha", "solver", "hidden_layer_sizes", "accuracy", "accuracy (cv)"]
        self.resultsTableView = TableView(None, columns)
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
                newNodeSpinBox.adjustSize()

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

        # Get the tuple of nodes and hidden layer sizes
        arrayOfNodesForEachHiddenLayer = list([int(spinbox.text()) for spinbox in self.nodeSpinBoxes])
        s = [str(i) for i in arrayOfNodesForEachHiddenLayer]
        hiddenLayersAndNodes = ", ".join(s)
        tupleOfNodes = tuple(arrayOfNodesForEachHiddenLayer)

        # Create classifier object
        self.classifier = MLPClassifier(activation='logistic', solver=self.solverMlpComboBox.currentText(),
                                        max_iter=int(self.numberOfMaxIterationsSpinBox.text()),
                                        learning_rate_init=0.002, alpha=float(self.alphaSpinBox.text()),
                                        hidden_layer_sizes=tupleOfNodes)

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
        accuracy = accuracy_score(y_test, y_predict)

        progressDialog.setWindowTitle("Computing cross-validation...")
        progressDialog.setValue(75)
        QApplication.processEvents()

        accuracy_list = cross_val_score(self.classifier, self.X, self.y, cv=int(self.cvSpinBox.text()),
                                        scoring='accuracy')

        self.recordParametersRow = [self.testSizeSpinBox.text(), self.randomStateSpinBox.text(), self.cvSpinBox.text(), self.numberOfMaxIterationsSpinBox.text(), str(float(self.alphaSpinBox.text())), self.solverMlpComboBox.currentText(), hiddenLayersAndNodes, str(accuracy), str(accuracy_list.mean())]
        self.resultTableModel.insertNewRow(self.recordParametersRow)

        progressDialog.setWindowTitle("Done...")
        progressDialog.setValue(100)
        QApplication.processEvents()
        QApplication.beep()
        progressDialog.close()

        formattedAccuracy = "{0:.2f}".format(accuracy * 100)
        formattedCrossValidationAccuracy = "{0:.2f}".format(accuracy_list.mean() * 100)

        resultsMessageBox = QMessageBox()
        resultsMessageBox.setWindowTitle("Results of " + self.name + " classifier")
        resultsMessageBox.setText("The predictive model's accuracy is: " + formattedAccuracy + "%\n" +
                                   "The cross validation's accuracy is: " + formattedCrossValidationAccuracy + "%");
        resultsMessageBox.setStyleSheet("""
               QMessageBox {
                   background-color: rgb(255, 229, 204);
                   color: #fffff8;
                   font-size: 12px;
                   font-weight: bold;
               }
           """)
        resultsMessageBox.exec()