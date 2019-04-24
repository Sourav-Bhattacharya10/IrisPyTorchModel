import json
import keras
import keras.backend.tensorflow_backend
import logging
from datetime import datetime

with open("config.json") as g:
        gdata = json.load(g)

# Basic Logger
logging.basicConfig(filename = gdata["logfilepath"], level = logging.INFO)

# Save Keras Model
def saveModel(model, modelfilepath = None):
    """
    Save Keras model
    
    Params:\n
    `model` : Keras model object
    `modelfilepath` : If it is None, get it from the config.json Else pass the value
    """

    with open("config.json") as f:
        data = json.load(f)

    try:
        if modelfilepath == None:
            modelfilename = data["savedmodelfilename"]
            modelfilepath = str(data["modelfolder"]) + str(modelfilename)

        keras.models.save_model(model, modelfilepath)
        logging.info(str(datetime.today()) + ' : Saved keras model')

    except Exception as e:
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

# Load Keras Model
def loadModel(modelfilepath = None):
    """
    Load Keras model
    
    Params:\n
    `modelfilepath` : If it is None, get it from the config.json Else pass the value

    Returns object model
    """

    with open("config.json") as f:
        data = json.load(f)

    model = None

    try:
        if modelfilepath == None:
            modelfilename = data["savedmodelfilename"]
            modelfilepath = str(data["modelfolder"]) + str(modelfilename)

        # Unload Keras Model by reseting tf.Session
        if keras.backend.tensorflow_backend._SESSION:
            import tensorflow as tf
            tf.reset_default_graph() 
            keras.backend.tensorflow_backend._SESSION.close()
            keras.backend.tensorflow_backend._SESSION = None

        logging.info(str(datetime.today()) + ' : Cleared tf.Session')

        model = keras.models.load_model(modelfilepath)
        logging.info(str(datetime.today()) + ' : Loaded keras model')
    
    except Exception as e:
        model = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))
    
    return model