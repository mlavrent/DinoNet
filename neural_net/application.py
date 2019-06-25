import webbrowser
import sys
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from neural_net.model import Model
import time


def logMessage(message: str):
    print("[{}] ".format(time.strftime("%D %H:%M:%S")) + message)


def startBrowser(conn: Connection):
    logMessage("Starting dino game")

    chromePath = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s"
    dinoURL = "chrome://dino"

    webbrowser.get(chromePath).open(dinoURL)

    conn.close()
    logMessage("Browser closed")


def runClassifier(conn: Connection, model_save_file: str):
    logMessage("Starting model")

    model = Model()
    # TODO: load model here

    logMessage("Model loaded")

    while True:
        break
    conn.close()
    logMessage("Model stopped")



if __name__ == "__main__":
    # Get directory of save file to load model from
    assert len(sys.argv) == 2
    model_save_file = sys.argv[1]

    # Set up the two processes to run side-by-side
    browserConn, modelConn = Pipe(True)

    browserProc = Process(target=startBrowser, args=(browserConn,))
    modelProc = Process(target=runClassifier, args=(modelConn, model_save_file))
    modelProc.daemon = True

    modelProc.start()
    browserProc.start()

    # Wait for user to close browser before terminating
    browserProc.join()
