from PIL import Image
from tensorflow.keras.utils import Sequence
import numpy as np
import random
import csv
import math
from enum import Enum
from typing import List, Generator, Tuple


baseLabelDir: str = "data/images/labels/"
baseImgDir: str = "data/images/"


class DataType(Enum):
    TRAINING = "training/"
    VALIDATION = "validation/"
    TESTING = "testing/"
    GENERAL = ""


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
        normImgArr = imgArr / 255

        # 50% of the time, invert image colors
        return random.choice([normImgArr, 1 - normImgArr])

    def get_target(self):
        # return np.array([self.jump, self.inAir, self.duck, self.isDucked])
        doNothing = not (self.jump or self.isDucked)
        return np.array([self.jump, self.isDucked, doNothing])

    def __str__(self) -> str:
        actionSymbol = "↑" if self.jump else ("↓" if self.duck else "•")
        stateSymbol = "✈" if self.inAir else ("🦆" if self.isDucked else "•")
        return "(t={} {} {})".format(round(self.time, 2), actionSymbol, stateSymbol)

    __repr__ = __str__


class DataFile:
    def __init__(self, dataType: DataType, datasetName: str):
        self.datasetName = datasetName
        self.filePath = baseLabelDir + dataType.value + datasetName + ".csv"

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
    def __init__(self, datasetNames: List[str], dataType: DataType, batchSize: int):
        self.batchSize = batchSize

        self.dataList = []
        for datasetName in datasetNames:
            dataFile = DataFile(dataType, datasetName)

            for dataPoint in dataFile.next_row():
                self.dataList.append(dataPoint)

            dataFile.close()

        random.shuffle(self.dataList)

        self.datasetSize = len(self.dataList)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        batch = self.dataList[(i % self.datasetSize) * self.batchSize:]
        return np.array([d.get_input() for d in batch]), np.array([d.get_target() for d in batch])

    def __len__(self) -> int:
        return math.ceil(len(self.dataList)/self.batchSize)


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader(["game4"])
    print(loader.dataList)
