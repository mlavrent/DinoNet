from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import os
import csv
import sys


class MainWindow(QWidget):

    def __init__(self, imgSet: str, fps: float):
        super().__init__()

        self.imgSet: str = imgSet
        self.labelFile = "data/images/labels/" + imgSet + ".csv"

        self.curImg = 1
        self.inAir = False
        self.ducked = False

        self.data = {}
        self.interval = 1/fps

        self.initUI()

    def initUI(self):
        layout = QGridLayout()
        self.setLayout(layout)

        font = QFont()
        font.setBold(True)
        font.setPixelSize(20)

        self.imgHead = QLabel("Current: " + str(self.curImg), self)
        self.imgHead.setFont(font)
        self.imgLabel = QLabel(self)
        self.previewHead = QLabel("Next:")
        self.previewHead.setFont(font)
        self.previewLabel = QLabel(self)

        self.stateImg = QLabel(self)
        self.stateImg.setMinimumWidth(100)

        layout.addWidget(self.imgHead, 0, 0)
        layout.addWidget(self.imgLabel, 1, 0)
        layout.addWidget(self.previewHead, 2, 0)
        layout.addWidget(self.previewLabel, 3, 0)
        layout.addWidget(self.stateImg, 1, 1)
        self.setWindowTitle("Dino Labeler | " + self.imgSet)
        self.showImage()

        self.data["0001.jpeg"] = (0, 0, 0, 0)

        self.show()


    def imgExists(self, n: int):
        return os.path.isfile("data/images/" + self.imgSet + "/" + str(n).zfill(4) + ".jpeg")

    def showImage(self):
        imagePath = "data/images/" + self.imgSet + "/" + str(self.curImg).zfill(4) + ".jpeg"

        image = QPixmap(imagePath)
        image = image.scaledToWidth(800)
        self.imgHead.setText("Current: " + str(self.curImg))
        self.imgLabel.setPixmap(image)
        self.imgLabel.update()

        if self.imgExists(self.curImg + 1):
            previewImagePath = "data/images/" + self.imgSet + "/" + str(self.curImg + 1).zfill(4) + ".jpeg"

            previewImage = QPixmap(previewImagePath)
            previewImage = previewImage.scaledToWidth(800)
            self.previewLabel.setPixmap(previewImage)
            self.previewLabel.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q:
            with open(self.labelFile, 'w', newline='') as file:
                header = ['file', 'time', 'jump', 'inAir', 'duck', 'isDucked']
                writer = csv.DictWriter(file, fieldnames=header)

                writer.writeheader()
                i = 0
                for key in self.data.keys():
                    rowdict = {
                        'file': key,
                        'time': i,
                        'jump': self.data[key][0],
                        'inAir': self.data[key][1],
                        'duck': self.data[key][2],
                        'isDucked': self.data[key][3],
                    }
                    writer.writerow(rowdict)
                    i += self.interval

            self.close()

        elif event.key() == Qt.Key_W:
            file = str(self.curImg).zfill(4) + ".jpeg"
            jump = not self.inAir
            self.data[file] = int(jump), 1, 0, 0

            self.inAir = not self.inAir
            self.ducked = False

        elif event.key() == Qt.Key_S:
            file = str(self.curImg).zfill(4) + ".jpeg"
            duck = not self.ducked
            self.data[file] = 0, 0, int(duck), 1

            self.ducked = not self.ducked
            self.inAir = False

        elif event.key() == Qt.Key_Right:
            self.curImg = self.curImg + 1 if self.imgExists(self.curImg + 1) else self.curImg

            file = str(self.curImg).zfill(4) + ".jpeg"
            self.data[file] = 0, int(self.inAir), 0, int(self.ducked)

        if self.inAir:
            arrow = QPixmap("labeling/img/upArrow.svg")
            arrow = arrow.scaledToHeight(50)
            self.stateImg.setPixmap(arrow)
        elif self.ducked:
            downarrow = QPixmap("labeling/img/downArrow.svg")
            downarrow = downarrow.scaledToHeight(50)
            self.stateImg.setPixmap(downarrow)
        else:
            self.stateImg.clear()

        self.showImage()


def run(imgSet: str, fps: float = 10):
    app = QApplication([])
    window = MainWindow(imgSet, fps)
    app.exec_()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run(sys.argv[1], sys.argv[2])
    else:
        run(sys.argv[1])
