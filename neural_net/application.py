import webbrowser
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


def runClassifier(conn: Connection):
    logMessage("Starting model")

    while True:
        break
    conn.close()
    logMessage("Model stopped")



if __name__ == "__main__":
    browserConn, modelConn = Pipe(True)

    browserProc = Process(target=startBrowser, args=(browserConn,))
    modelProc = Process(target=runClassifier, args=(modelConn,))
    modelProc.daemon = True

    modelProc.start()
    browserProc.start()

    # Wait for user to close browser before terminating
    browserProc.join()
