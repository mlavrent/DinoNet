from PIL import Image
from tensorflow.keras.utils import Sequence
import numpy as np
import random
import csv
import math
from typing import List, Tuple


baseLabelDir: str = "data/images/labels/"
baseImgDir: str = "data/images/"


class Datum:
    def __init__(self, fileName: str, time: float, jump: bool, inAir: bool, duck: bool, isDucked: bool):
        self.fileName = fileName

        self.time = time
        self.jump = jump
        self.inAir = inAir
        self.duck = duck
        self.isDucked = isDucked

    def get_input(self):
        # load image only when needed
        pilImg = Image.open(self.fileName).convert('LA')
        imgArr = np.array(pilImg)
        normImgArr = self.imgArr / 255

        return normImgArr

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


class DataLoader(Sequence):
    def __init__(self, datasetNames: List[str], batchSize: int):
        self.batchSize = batchSize

        self.dataList = []
        for datasetName in datasetNames:
            dataFile = DataFile(datasetName)

            for dataPoint in dataFile.next_row():
                self.dataList.append(dataPoint)

        random.shuffle(self.dataList)

        self.datasetSize = len(self.allData)

    def __getitem__(self, i: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        batch = self.dataList[(i % self.datasetSize) * self.batchSize:]
        return [(d.get_input(), d.get_target()) for d in batch]

    def __len__(self) -> int:
        return math.ceil(len(self.load_all_data())/self.batchSize)

    def close_data(self):
        for file in self.imgFiles:
            file.close()


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader(["game4"])
    print(loader.load_all_data())
