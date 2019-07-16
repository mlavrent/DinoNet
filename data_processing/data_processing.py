from PIL import Image
import numpy as np
import random
import csv
from typing import List, Generator


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


class DataLoader:
    def __init__(self, datasetNames: List[str]):
        self.imgFiles = []
        for datasetName in datasetNames:
            self.imgFiles.append(DataFile(datasetName))

    def load_all_data(self) -> List[Datum]:
        data = []
        for file in self.imgFiles:
            for datum in file.next_row():
                data.append(datum)
        return data

    def load_all_data_random_order(self) -> List[Datum]:
        data = self.load_all_data()
        random.shuffle(data)
        return data

    #TODO: switch over to using tf.keras.utils.Sequence for multiprocessing-safety

    def batch_generator(self, batchSize) -> List[Datum]:
        allData = self.load_all_data_random_order()
        datasetSize = len(allData)

        i = 0
        while True:
            # Get the batch
            if i + batchSize > datasetSize:
                # If needed, wrap around to start of data list
                batchData = allData[i:] + allData[:datasetSize - i]
            else:
                batchData = allData[i:i + batchSize]

            # Yield the batch out
            yield batchData

            # Increment the position
            i += batchSize
            if i >= len(allData):
                # If over, wrap around and re-shuffle the dataset
                i %= datasetSize
                random.shuffle(allData)

    def size(self) -> int:
        return len(self.load_all_data())

    def close_data(self):
        for file in self.imgFiles:
            file.close()


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader(["game4"])
    print(loader.load_all_data())
