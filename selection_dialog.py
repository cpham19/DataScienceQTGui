import pandas as pd
from functools import partial
from tableview import TableView
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QDialog, QCheckBox, QRadioButton, QDialogButtonBox
from PyQt5.QtCore import Qt

# Selection Feature Dialog
class SelectionFeaturesDialog(QDialog):
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path, sep=',')
        self.columns = list(self.df.columns.values)
        self.data = self.df.to_numpy().tolist()
        self.initUI()

    def initUI(self):
        # Dialog stuff

        self.setWindowTitle("Select features and label")
        self.setWindowFlags(Qt.WindowTitleHint | Qt.CustomizeWindowHint)

        self.setStyleSheet(""" 
            # QWidget {
            #     background: gray;
            #     color:white;
            # }
        """)

        self.resize(1280, 720)

        # Outer layout to contain the tableview and the groupbox for selections
        layout = QVBoxLayout()

        # Datatable
        self.tableView = TableView(self.data, self.columns)

        # Groupbox to split into two groupboxes for selections
        groupbox = QGroupBox("Selections")
        groupboxLayout = QHBoxLayout()
        groupbox.setLayout(groupboxLayout)

        # Feature groupbox
        featureGroupbox = QGroupBox("Select your features")
        featureGroupboxLayout = QVBoxLayout()
        featureGroupbox.setLayout(featureGroupboxLayout)

        # Label groupbox
        labelGroupbox = QGroupBox("Select your label")
        labelGroupboxLayout = QVBoxLayout()
        labelGroupbox.setLayout(labelGroupboxLayout)

        groupboxLayout.addWidget(featureGroupbox)
        groupboxLayout.addWidget(labelGroupbox)

        # Save the buttons in arrays for each group box
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

        # Ok and Cancel buttons
        confirmButtonBox = QDialogButtonBox(self)
        confirmButtonBox.setOrientation(Qt.Horizontal)
        confirmButtonBox.setStandardButtons(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        confirmButtonBox.accepted.connect(self.accept)
        confirmButtonBox.rejected.connect(self.reject)

        layout.addWidget(self.tableView)
        layout.addWidget(groupbox)
        layout.addWidget(confirmButtonBox)

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

    def getFeatures(self):
        array = []
        for feature in self.arrayOfFeatureCheckboxes:
            if (feature.isChecked()):
                array.append(feature.text())

        return array

    def getLabel(self):
        array = []
        for label in self.arrayOfLabelRadioButtons:
            if (label.isChecked()):
                array.append(label.text())

        return array