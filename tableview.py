from PyQt5.QtWidgets import QTableView, QAbstractItemView, QHeaderView
from PyQt5.QtCore import Qt, QUrl, QVariant, QAbstractItemModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem

class TableView(QTableView):
    def __init__(self, data = None, headers = None):
        super().__init__()
        self.headers = headers
        if (data == None):
            self.data = [[''] * len(self.headers)]
        else:
            self.data = data
        self.initUI()

    def initUI(self):
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.verticalHeader().setDefaultSectionSize(50)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tableModel = TableModel(self.data, self.headers)
        self.setModel(self.tableModel)

        self.setStyleSheet("""
            QHeaderView::section {
                background-color: #646464;
                padding: 4px;
                border: 1px solid #fffff8;
                font-size: 12px;
                color: white;
            }
        
            QTableView {
                gridline-color: gray;
            }
        
            QScrollBar:vertical {
                background: rgb(49, 49, 49);
            }
            
            QTableView QTableCornerButton::section {
                background-color: #646464;
                border: 1px solid #fffff8;
            }
        """)

    def getModel(self):
        return self.tableModel

class TableModel(QStandardItemModel):
    def __init__(self, data, headers):
        super().__init__()
        self.data = data
        self.headers = headers
        self.setupData()

    # def rowCount(self, parent):
    #     return len(self.data)

    def columnCount(self, parent):
        return len(self.headers)

    def headerData(self, col, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[col]
            elif orientation == Qt.Vertical:
                return col
        return None

    def setupData(self):
        rowLength = len(self.data)
        colLength = len(self.headers)

        for row in range(rowLength):
            for column in range(colLength):
                item = QStandardItem(str(self.data[row][column]))
                self.setItem(row, column, item)

    def insertNewRow(self, row):
        newRow = []
        for col in row:
            item = QStandardItem(str(col))
            newRow.append(item)

        self.appendRow(newRow)
