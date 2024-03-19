import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QSlider, QProgressBar
class IrisModule(nn.Module):
    def __init__(self):
        super(IrisModule, self).__init__()
        self.f1 = nn.Linear(4, 8)
        self.f2 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = self.f2(x)
        return x

class Irisclassifi(QWidget):
    def __init__(self):
        super().__init__()
        self.InitUserInterface()
        self.LoadIris()
        self.PreprocessIris()
        self.TrainedToBuildModel()

    def InitUserInterface(self):
        self.setWindowTitle('Iris Dataset Classifier')
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color:rgb(255, 229, 180);")

        self.label1 = QLabel('Sepal Length:')
        self.input1 = QLineEdit()
        self.label2 = QLabel('Sepal Width:')
        self.input2 = QLineEdit()
        self.label3 = QLabel('Petal Length:')
        self.input3 = QLineEdit()
        self.label4 = QLabel('Petal Width:')
        self.input4 = QLineEdit()

        self.predictbutton = QPushButton('Predict')
        self.predictbutton.clicked.connect(self.Predict)
        self.predictbutton.setStyleSheet("background-color : Red;color:white")

        self.label5= QLabel()
        self.problabel = QLabel()

        self.lrnglabel = QLabel('Learning Rate:')
        self.lrngslider = QSlider()
        self.lrngslider.setOrientation(1)
        self.lrngslider.setMinimum(1)
        self.lrngslider.setMaximum(10)
        self.lrngslider.setValue(5)

        self.epolabel = QLabel('Epochs:')
        self.eposlider = QSlider()
        self.eposlider.setOrientation(1)
        self.eposlider.setMinimum(1)
        self.eposlider.setMaximum(100)
        self.eposlider.setValue(50)

        self.tunebutton = QPushButton('Tune Hyperparameters')
        self.tunebutton.clicked.connect(self.TuneHyperparameters)
        self.tunebutton .setStyleSheet("background-color: rgb(32, 55, 157 );color:white")

        self.progressbar = QProgressBar()

        Mainbox = QVBoxLayout()
        box1 = QHBoxLayout()
        box1.addWidget(self.label1)
        box1.addWidget(self.input1)
        box1.addWidget(self.label2)
        box1.addWidget(self.input2)
        box2 = QHBoxLayout()
        box2.addWidget(self.label3)
        box2.addWidget(self.input3)
        box2.addWidget(self.label4)
        box2.addWidget(self.input4)
        box3 = QHBoxLayout()
        box3.addWidget(self.predictbutton)
        box4 = QHBoxLayout()
        box4.addWidget(self.label5)
        box5 = QHBoxLayout()
        box5.addWidget(self.problabel)
        box6 = QHBoxLayout()
        box6.addWidget(self.lrnglabel)
        box6.addWidget(self.lrngslider)
        box7 = QHBoxLayout()
        box7.addWidget(self.epolabel)
        box7.addWidget(self.eposlider)
        box8 = QHBoxLayout()
        box8.addWidget(self.tunebutton)
        box9 = QHBoxLayout()
        box9.addWidget(self.progressbar)

        Mainbox.addLayout(box1)
        Mainbox.addLayout(box2)
        Mainbox.addLayout(box3)
        Mainbox.addLayout(box4)
        Mainbox.addLayout(box5)
        Mainbox.addLayout(box6)
        Mainbox.addLayout(box7)
        Mainbox.addLayout(box8)
        Mainbox.addLayout(box9)

        self.setLayout(Mainbox)

    def LoadIris(self):
        iris_data = pd.read_csv('iris.csv')
        self.X = iris_data.iloc[:, :-1].values
        self.y = iris_data.iloc[:, -1].values

    def PreprocessIris(self):
        LabelEncoder1 = LabelEncoder()
        self.y = LabelEncoder1.fit_transform(self.y)

    def TrainedToBuildModel(self):
        self.model = IrisModule()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def TrainModel(self, epo):
        self.model.train()
        for i in range(epo):
            self.optimizer.zero_grad()
            outputs = self.model(torch.FloatTensor(self.X))
            loss = self.criterion(outputs, torch.LongTensor(self.y))
            loss.backward()
            self.optimizer.step()
            self.progressbar.setValue(i+1)

    def Predict(self):
        sepallength = float(self.input1.text())
        sepalwidth = float(self.input2.text())
        petallength = float(self.input3.text())
        petalwidth = float(self.input4.text())

        data = np.array([[sepallength, sepalwidth, petallength, petalwidth]])
        with torch.no_grad():
            output = self.model(torch.FloatTensor(data))
        predicted_class = torch.argmax(output).item()
        probability = torch.max(F.softmax(output, dim=1)).item()

        species = ['Setosa', 'Versicolor', 'Virginica']
        self.label5.setText(f'Predicted Class: {species[predicted_class]}')
        self.problabel.setText(f'Probability: {probability:.4f}')

    def TuneHyperparameters(self):
        learningrate = self.lrngslider.value() * 0.01
        epo= self.eposlider.value()

        self.progressbar.setValue(0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learningrate)
        self.TrainModel(epo)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    irisclassifier = Irisclassifi()
    irisclassifier.show()
    sys.exit(app.exec_())

