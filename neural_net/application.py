from selenium import webdriver
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from neural_net.model import Model
import numpy as np
from PIL import Image
from enum import Enum
from io import BytesIO
import base64
import time


class Action(Enum):
    JUMP = 0
    DUCK = 1
    DO_NOTHING = 2


def logMessage(message: str):
    print("[{}] ".format(time.strftime("%D %H:%M:%S")) + message)


def startBrowser(conn: Connection):
    logMessage("Starting dino game")

    chromePath = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s"
    dinoURL = "chrome://dino"

    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get(dinoURL)

    body = driver.find_element_by_css_selector("body")

    time.sleep(3)
    body.send_keys(" ")
    time.sleep(3)  # Wait until it goes to full screen

    # Run browser until user closes it
    while True:
        # Take screenshot and send it
        sshotStr = driver.get_screenshot_as_base64()

        if sshotStr is not None:
            img = Image.open(BytesIO(base64.b64decode(sshotStr))).convert('LA')
            img = img.crop(box=(0, 285, 1920, 765))  # box from (0, 285) to (1920, 765) that's 1920x480
            img = img.resize((120, 30), resample=Image.BILINEAR)  # resize down to 120x30
            conn.send(img)
            logMessage("Screenshot sent")

            result = conn.recv()
            logMessage(result.name)
        else:
            break

    driver.quit()
    logMessage("Browser closed")


def runClassifier(conn: Connection, model_save_file: str):
    logMessage("Starting model")
    model = Model()
    model.load_weights(model_save_file).expect_partial()
    logMessage("Model loaded")

    while True:
        try:
            img = conn.recv()
            imgArr = np.array(img).reshape((1, img.size[1], img.size[0], 2)) / 255

            aResult = model.predict(imgArr)
            iResult = np.amax(aResult)
            if iResult == 0:
                conn.send(Action.JUMP)
            elif iResult == 1:
                conn.send(Action.DUCK)
            else:
                conn.send(Action.DO_NOTHING)

            logMessage("Result sent")
        except EOFError:
            break

    logMessage("Model stopped")



if __name__ == "__main__":
    # Get directory of save file to load model from
    # assert len(sys.argv) == 2
    # model_save_file = sys.argv[1]

    # Set up the two processes to run side-by-side
    browserConn, modelConn = Pipe(True)

    browserProc = Process(target=startBrowser, args=(browserConn,))
    modelProc = Process(target=runClassifier, args=(modelConn, "saved_models/adam-lr0.005/dinoModel"))
    modelProc.daemon = True

    modelProc.start()
    browserProc.start()

    # Wait for user to close browser before terminating
    browserProc.join()

    # Close the pipe connections
    browserConn.close()
    modelConn.close()
