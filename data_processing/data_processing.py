from PIL import Image
from tensorflow.keras.utils import Sequence
import numpy as np
import random
import csv
import math
from enum import Enum
from typing import List, Generator, Tuple, Optional


baseLabelDir: str = "data/images/labels/"
baseImgDir: str = "data/images/"


class DataType(Enum):
    TRAINING = "training/"
    VALIDATION = "validation/"
    TESTING = "testing/"
    GENERAL = ""


def loadImage(fileName: str):
    pilImg = Image.open(fileName).convert("L")
    imgArr = np.array(pilImg).reshape((pilImg.size[1], pilImg.size[0], 1)) / 255

    return pilImg, imgArr


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
        _, imgArr = loadImage(self.fileName)

        # 50% of the time, invert image colors
        return random.choice([imgArr, 1 - imgArr])

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
    def __init__(self, datasetNames: List[str], dataType: DataType, batchSize: Optional[int], resampleCategories=False):
        self.dataList = []
        for datasetName in datasetNames:
            dataFile = DataFile(dataType, datasetName)

            for dataPoint in dataFile.next_row():
                self.dataList.append(dataPoint)

            dataFile.close()
        assert len(self.dataList) > 0

        random.shuffle(self.dataList)
        if resampleCategories:
            # Multiply each category by (# in max cat)/(# in this cat) and add that to datalist
            dataByCategory = [[] for _ in self.dataList[0].get_target()]

            for point in self.dataList:
                category = np.argmax(point.get_target())
                dataByCategory[category].append(point)

            maxCount = max([len(l) for l in dataByCategory])

            self.dataList = []
            for categoryData in dataByCategory:
                q, r = divmod(maxCount, len(categoryData))
                self.dataList.extend(categoryData * q + categoryData[:r])


        random.shuffle(self.dataList)
        self.datasetSize = len(self.dataList)
        self.batchSize = self.datasetSize if batchSize is None else batchSize

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        batch = self.dataList[i * self.batchSize:(i + 1) * self.batchSize]
        return np.array([d.get_input() for d in batch]), np.array([d.get_target() for d in batch])

    def __len__(self) -> int:
        return math.ceil(self.datasetSize/self.batchSize)


if __name__ == "__main__":
    # Calculations for class weights
    loader = DataLoader(["game1", "game2", "game3", "game4", "game5"], DataType.TRAINING, batchSize=10000)

    jumpCt = 0
    duckCt = 0
    doNothingCt = 0
    total = loader.datasetSize
    for inputs, targets in loader:
        sums = np.sum(targets, axis=0)
        jumpCt += sums[0]
        duckCt += sums[1]
        doNothingCt += sums[2]

    print("Jump: {:d}/{:d} ({:.1f}%)".format(jumpCt, total, 100 * (jumpCt/total)))
    print("Duck: {:d}/{:d} ({:.1f}%)".format(duckCt, total, 100 * (duckCt/total)))
    print("Nothing: {:d}/{:d} ({:.1f}%)".format(doNothingCt, total, 100 * (doNothingCt/total)))

