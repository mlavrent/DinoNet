from selenium import webdriver
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from neural_net.model import Model
from PIL import Image
from io import BytesIO
import base64
import time


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

    # Run browser until user closes it
    while True:
        # Take screenshot and send it
        sshot_str = driver.get_screenshot_as_base64()

        if sshot_str is not None:
            img = Image.open(BytesIO(base64.b64decode(sshot_str)))
        else:
            break

    driver.quit()
    logMessage("Browser closed")


def runClassifier(conn: Connection, model_save_file: str):
    logMessage("Starting model")
    model = Model()
    model.load_weights(model_save_file)
    logMessage("Model loaded")

    while True:
        try:
            input_img = conn.recv()
            result = model.predict(input_img)
            conn.send(result)
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
    modelProc = Process(target=runClassifier, args=(modelConn, "saved_models/dinoModel"))
    modelProc.daemon = True

    modelProc.start()
    browserProc.start()

    # Wait for user to close browser before terminating
    browserProc.join()

    # Close the pipe connections
    browserConn.close()
    modelConn.close()
