from PIL import Image
import numpy as np
import os
import csv
from typing import List, Generator


baseLabelDir: str = "data/images/labels/"
baseImgDir: str = "data/images/"


class Datum:
    def __init__(self, fileName: str, time: float, jump: bool, inAir: bool, duck: bool, isDucked: bool):
        self.fileName = fileName

        # convert to grayscale immediately
        pilImg = Image.open(fileName).convert('LA')
        self.imgArr = np.array(pilImg)
        self.normImgArr = self.imgArr / 255

        self.time = time
        self.jump = jump
        self.inAir = inAir
        self.duck = duck
        self.isDucked = isDucked

    def get_input(self):
        return self.normImgArr

    def get_target(self):
        return np.array([self.jump, self.inAir, self.duck, self.isDucked])

    def __str__(self) -> str:
        actionSymbol = "↑" if self.jump else ("↓" if self.duck else "•")
        stateSymbol = "✈" if self.inAir else ("🦆" if self.isDucked else "•")
        return "(t={} {} {})".format(round(self.time, 2), actionSymbol, stateSymbol)

    __repr__ = __str__


class DataFile:
    def __init__(self, datasetName: str):
        self.datasetName = datasetName
        self.filePath = baseLabelDir + datasetName + ".csv"

        self.file = open(self.filePath, 'r', newline='')
        self.reader = csv.DictReader(self.file, delimiter=',')

        self.rowNum = 0

    def next_row(self) -> Generator[Datum, None, None]:

        for row in self.reader:
            yield Datum(baseImgDir + self.datasetName + "/" + row['file'],
                        float(row['time']),
                        bool(int(row['jump'])), bool(int(row['inAir'])),
                        bool(int(row['duck'])), bool(int(row['isDucked'])))
            self.rowNum += 1

    def close(self):
        self.file.close()


class DataLoader:
    def __init__(self, datasetNames: List[str]):
        self.imgFiles = []
        for datasetName in datasetNames:
            self.imgFiles.append(DataFile(datasetName))

    def load_data(self) -> List[Datum]:
        data = []
        for file in self.imgFiles:
            for datum in file.next_row():
                data.append(datum)
        return data

    def close_data(self):
        for file in self.imgFiles:
            file.close()


if __name__ == "__main__":
    # Test dataloader
    loader = DataLoader(["game4"])
    print(loader.load_data())
