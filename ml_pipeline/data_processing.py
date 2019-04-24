import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

with open("config.json") as g:
        gdata = json.load(g)

# Basic Logger
logging.basicConfig(filename = gdata["logfilepath"], level = logging.INFO)

# Process the data such that it can be fitted in the model
def processData(inputDataFrame, columnsNamesToWorkWith = [], shuffleData = False, substituteEmptyFields = False):
    """
    Process the data from various sources as input for the machine learning model.
    
    Params:\n
    `inputDataFrame` : pandas DataFrame\n
    `columnsNamesToWorkWith` : list of the column names of the `inputDataFrame` to work with\n
    `shuffleData` : Should the data of the `inputDataFrame` be shuffled?
    `substituteEmptyFields` : Should the empty fields of the `inputDataFrame` be substituted with standard values i.e. for object or category type, fill "Not Available" and all numeric values with 0?

    Returns pandas.DataFrame resultingDataFrame
    """

    resultingDataFrame = None

    try:
        # Filter dataframe according to the columnsNamesToWorkWith list
        if len(columnsNamesToWorkWith) != 0:
            inputDataFrame = inputDataFrame[columnsNamesToWorkWith]
            logging.info(str(datetime.today()) + ' : Filtered dataframe according to the columnsNamesToWorkWith list')

        if shuffleData:
            # Shuffle the complete dataframe
            inputDataFrame = shuffleDataFrame(inputDataFrame)
            logging.info(str(datetime.today()) + ' : Shuffled data')

        if substituteEmptyFields:
            # Substitute the complete dataframe
            inputDataFrame = substituteDataFrame(inputDataFrame)
            logging.info(str(datetime.today()) + ' : Substituted data')

        resultingDataFrame = inputDataFrame
    
    except Exception as e:
        resultingDataFrame = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return resultingDataFrame

# Custom One Hot Encoding Method
def OneHotEncodingColumnsDataFrame(inputDataFrame, colname, actualvalues = [], distinctvalues = []):
    colname_df = pd.DataFrame({})
    
    try:
        for i in range(len(distinctvalues)):
            colname_df[str(colname) + '_' + str(distinctvalues[i])] = [0.0 for x in range(len(actualvalues))]
        
        for i in range(len(actualvalues)):
            av = actualvalues[i]
            
            for j in range(len(distinctvalues)):
                dv = distinctvalues[j]
                if av == dv:
                    colname_df[str(colname) + '_' + str(dv)][i] = 1.0
                    break
        
        for i in range(len(colname_df.columns)):
            inputDataFrame[str(colname_df.columns[i])] = colname_df[colname_df.columns[i]].values
            
        inputDataFrame = inputDataFrame.drop(colname, axis = 1)
        logging.info(str(datetime.today()) + ' : OneHotEncoding for column "' + colname + '" is done')

    except Exception as e:
        inputDataFrame = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))
        
    return inputDataFrame

# Transform the categorical data into one hot encoding
def dummifyData(inputDataFrame, dictObject):
    """
    Dummify the dataframe (one hot encoding)
    
    Params:\n
    `inputDataFrame` : pandas DataFrame\n
    `dictObject` : dictionary object for the object or categorical dtype where key is the column name and value is the list of unqiue categorical values\n

    Returns pandas.DataFrame resultingDataFrame
    """

    resultingDataFrame = None

    try:
        
        for key, value in dictObject.items():
            inputDataFrame = OneHotEncodingColumnsDataFrame(inputDataFrame, key, actualvalues = inputDataFrame[key].values, distinctvalues = value)
        
        logging.info(str(datetime.today()) + ' : Dummified data')

        resultingDataFrame = inputDataFrame

    except Exception as e:
        resultingDataFrame = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return resultingDataFrame

# Divide the data into features Xdata and labels Ydata
def divideDataInXY(inputDataFrame, listcolYNames):
    """
    Divide the dataframe in X and Y numpy arrays
    
    Params:\n
    `inputDataFrame` : pandas DataFrame\n
    `listcolYNames` : list of column names of the label or class

    Returns numpy.ndarray Xdata, Ydata
    """

    Xdata = None
    Ydata = None

    try:
        if (listcolYNames != None) & (len(listcolYNames) != 0):
            Xdata = inputDataFrame.loc[:, inputDataFrame.columns.str.contains('|'.join(listcolYNames)) != 1]
            Xdata = Xdata.values
            Xdata = Xdata.astype('float32')
            logging.info(str(datetime.today()) + ' : Got Xdata')
            Ydata = inputDataFrame[listcolYNames].values
            logging.info(str(datetime.today()) + ' : Got Ydata')
        else:
            raise Exception("listcolYNames not provided")

    except Exception as e:
        Xdata = None
        Ydata = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return Xdata, Ydata

# Normalize the Xdata using MinMax Scaler
def normalizeData(Xdata, fit_transform = True, scalerfilepath = None):
    """
    Normalize the numpy N-D array Xdata using Min-Max Scaler (except 1D array)
    
    Params:\n
    `Xdata` : numpy.ndarray\n
    `fit_transform` : Is the scaler object applied for the first time? If yes, then True. Else False
    `scalerfilepath` : If it is None, get it from the config.json Else pass the value

    Returns numpy.ndarray resultingXdata
    """

    with open("config.json") as f:
        data = json.load(f)

    resultingXdata = None

    try:
        if scalerfilepath == None:
            scalerfilename = data["scalerfilename"]
            scalerfilepath = str(data["modelfolder"]) + str(scalerfilename)


        if fit_transform:
            scaler = MinMaxScaler(feature_range=(0, 1))
            Xdata = scaler.fit_transform(Xdata)
            saveScalerObject(scaler, scalerfilepath)
            logging.info(str(datetime.today()) + ' : Scaled data using fit transform')
        else:
            scaler = loadScalerObject(scalerfilepath)
            Xdata = scaler.transform(Xdata)
            logging.info(str(datetime.today()) + ' : Scaled data using transform')

        resultingXdata = Xdata

    except Exception as e:
        resultingXdata = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return resultingXdata


# Denormalize the Xdata using MinMax Scaler
def denormalizeData(Xdata, scalerfilepath = None):
    """
    Normalize the numpy N-D array Xdata using Min-Max Scaler (except 1D array)
    
    Params:\n
    `Xdata` : numpy.ndarray\n
    `scalerfilepath` : If it is None, get it from the config.json Else pass the value

    Returns numpy.ndarray resultingXdata
    """

    with open("config.json") as f:
        data = json.load(f)

    resultingXdata = None

    try:
        if scalerfilepath == None:
            scalerfilename = data["scalerfilename"]
            scalerfilepath = str(data["modelfolder"]) + str(scalerfilename)

        scaler = loadScalerObject(scalerfilepath)
        Xdata = scaler.inverse_transform(Xdata)
        logging.info(str(datetime.today()) + ' : Unscaled data using inverse transform')

        resultingXdata = Xdata

    except Exception as e:
        resultingXdata = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return resultingXdata

# Normalize the 1-D array using MinMax Scaler (custom)
def normalize1DArray(Xdata, fit_transform = True, scalerfilepath = None):
    """
    Normalize the list or numpy 1D array Xdata using Min-Max Scaler (custom)
    
    Params:\n
    `Xdata` : numpy.ndarray or list\n
    `fit_transform` : Is the scaler object applied for the first time? If yes, then True. Else False\n
    `scalerfilepath` : If it is None, get it from the config.json Else pass the value

    Returns numpy.ndarray resultingXdata
    """

    with open("config.json") as f:
        data = json.load(f)

    resultingXdata = None

    try:
        if scalerfilepath == None:
            scalerfilename = data["scalerfilename"]
            scalerfilepath = str(data["modelfolder"]) + str(scalerfilename)

        if str(type(Xdata)) == "<class 'numpy.ndarray'>":
            Xdata = Xdata.tolist()

        if fit_transform:
            minX = min(Xdata)
            maxX = max(Xdata)

            if minX == maxX:
                for i in range(len(Xdata)):
                    Xdata[i] = 1
            else:
                for i in range(len(Xdata)):
                    Xdata[i] = ((Xdata[i] - minX)/(maxX - minX))

            dictScaler = {'minX' : minX, 'maxX' : maxX}

            saveScalerObject(dictScaler, scalerfilepath)

        else:
            dictScaler = loadScalerObject(scalerfilepath)

            minX = dictScaler['minX']
            maxX = dictScaler['maxX']

            if minX == maxX:
                for i in range(len(Xdata)):
                    Xdata[i] = 1
            else:
                for i in range(len(Xdata)):
                    Xdata[i] = ((Xdata[i] - minX)/(maxX - minX))

        resultingXdata = np.array(Xdata)

    except Exception as e:
        resultingXdata = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return resultingXdata


# Denormalize the 1-D array using MinMax Scaler (custom)
def denormalize1DArray(Xdata, scalerObject = None, scalerfilepath = None):
    """
    Denormalize the list or numpy 1D array Xdata using Min-Max Scaler (custom)
    
    Params:\n
    `Xdata` : numpy.ndarray or list\n
    `scalerfilepath` : If it is None, get it from the config.json Else pass the value

    Returns numpy.ndarray resultingXdata
    """

    with open("config.json") as f:
        data = json.load(f)

    resultingXdata = None

    try:
        if (scalerfilepath == None) & (scalerObject == None):
            scalerfilename = data["scalerfilename"]
            scalerfilepath = str(data["modelfolder"]) + str(scalerfilename)

        if str(type(Xdata)) == "<class 'numpy.ndarray'>":
            Xdata = Xdata.tolist()

        if scalerObject == None:
            dictScaler = loadScalerObject(scalerfilepath)
        else:
            dictScaler = scalerObject
        
        minX = dictScaler['minX']
        maxX = dictScaler['maxX']

        if minX == maxX:
            for i in range(len(Xdata)):
                Xdata[i] = minX
        else:
            for i in range(len(Xdata)):
                Xdata[i] = (Xdata[i] * (maxX - minX)) + minX

        resultingXdata = np.array(Xdata)

    except Exception as e:
        resultingXdata = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return resultingXdata


# Split the Xdata and Ydata in train and test sets
def generateTrainAndTestXY(Xdata, Ydata, splitfraction = 0.9):
    """
    Generate the numpy array trainX, trainY, testX, testY
    
    Params:\n
    `Xdata` : numpy.ndarray\n
    `Ydata` : numpy.ndarray\n
    `splitfraction` : float32. Default value 0.9 as 0.8 will used for training and 0.1 will be used in validation split\n

    Returns numpy.ndarray trainX, trainY, testX, testY
    """

    trainX = None
    trainY = None
    testX = None
    testY = None

    try:
        split = int(len(Xdata) * splitfraction)

        trainX = Xdata[0:split]
        trainY = Ydata[0:split]

        testX = Xdata[split:len(Xdata)]
        testY = Ydata[split:len(Xdata)]
        logging.info(str(datetime.today()) + ' : Got trainX, trainY, testX and testY')

    except Exception as e:
        trainX = None
        trainY = None
        testX = None
        testY = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return trainX, trainY, testX, testY


def saveScalerObject(scaler, scalerfilepath = None):
    """
    Save the scaler object for prediction part
    
    Params:\n
    `scaler` : scaler object\n
    `scalerfilepath` : If it is None, get it from the config.json Else pass the value
    """
    try:  
        joblib.dump(scaler, scalerfilepath)
        logging.info(str(datetime.today()) + ' : Saved scaler object')
    
    except Exception as e:
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

def loadScalerObject(scalerfilepath = None):
    """
    Load the scaler object for prediction part
    
    Params:\n
    `scalerfilepath` : If it is None, get it from the config.json Else pass the value

    Returns object scalerObject
    """

    scalerObject = None

    try:
        scalerObject = joblib.load(scalerfilepath)
        logging.info(str(datetime.today()) + ' : Loaded scaler object')

    except Exception as e:
        scalerObject = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return scalerObject

def shuffleDataFrame(inputDataFrame):

    resultingDataFrame = None

    try:
        resultingDataFrame = inputDataFrame.sample(frac = 1)
        logging.info(str(datetime.today()) + ' : Shuffled dataframe')
    
    except Exception as e:
        resultingDataFrame = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return resultingDataFrame

def substituteDataFrame(inputDataFrame):

    resultingDataFrame = None

    try:
        columnDataTypeDict = dict(inputDataFrame.dtypes)

        for key in columnDataTypeDict.keys():
            if str(columnDataTypeDict[key]) == "object" or str(columnDataTypeDict[key]) == "category":
                inputDataFrame[key].fillna("Not Available", inplace = True)
            else:
                inputDataFrame[key].fillna(0, inplace = True)
        logging.info(str(datetime.today()) + ' : Replaced null values')

        for key in columnDataTypeDict.keys():
            if str(columnDataTypeDict[key]) == "object"  or str(columnDataTypeDict[key]) == "category":
                inputDataFrame[key] = inputDataFrame[key].astype('category')
            else:
                inputDataFrame[key] = inputDataFrame[key].astype('float32')
        logging.info(str(datetime.today()) + ' : Transformed column datatype')

        resultingDataFrame = inputDataFrame

    except Exception as e:
        resultingDataFrame = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return resultingDataFrame

# Get the list of object type or categorical type column names
def getObjectTypeColumns(inputDataFrame):
    """
    Get the list of object type or categorical type column names
    
    Params:\n
    `inputDataFrame` : pandas DataFrame\n

    Returns list colslist
    """

    colslist = []

    try:
        columnDataTypeDict = dict(inputDataFrame.dtypes)

        for key in columnDataTypeDict.keys():
            if str(columnDataTypeDict[key]) == "object"  or str(columnDataTypeDict[key]) == "category":
                colslist.append(key)
        logging.info(str(datetime.today()) + ' : Got list of object type columns')
    
    except Exception as e:
        colslist = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return colslist

# Get the list of object type or categorical type column names
def getUniqueValuesDictionary(inputDataFrame, colslist = []):
    """
    Get a dictionary object for the object or categorical dtype where key is the column name and value is the list of unqiue categorical values
    
    Params:\n
    `inputDataFrame` : pandas DataFrame\n
    `colslist` : list of column names that are object or categorical dtype\n

    Returns dict dictUniqueValues
    """

    dictUniqueValues = {}

    try:
        
        if len(colslist) != 0:
            for i in range(len(colslist)):
                dictUniqueValues[colslist[i]] = inputDataFrame[colslist[i]].unique().tolist()
        logging.info(str(datetime.today()) + ' : Got list of object type columns')
    
    except Exception as e:
        dictUniqueValues = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return dictUniqueValues

# Save the UniqueValuesDictionary to a local path
def saveUniqueValuesDictionary(dictObj, filepath):
    """
    Save a dictionary object for the object or categorical dtype where key is the column name and value is the list of unqiue categorical values
    
    Params:\n
    `dictObj` : dictionary obejct\n
    `filepath` : filepath for saving the dictionary object\n
    """

    try:  
        joblib.dump(dictObj, filepath)
        logging.info(str(datetime.today()) + ' : Saved unique values dictionary object')
    
    except Exception as e:
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

# Load the UniqueValuesDictionary from a local path
def loadUniqueValuesDictionary(filepath):
    """
    Load a dictionary object for the object or categorical dtype where key is the column name and value is the list of unqiue categorical values
    
    Params:\n
    `filepath` : filepath for loading the dictionary object\n

    Returns dict dictObject
    """

    dictObject = None

    try:
        dictObject = joblib.load(filepath)
        logging.info(str(datetime.today()) + ' : Loaded unique values dictionary object')

    except Exception as e:
        dictObject = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return dictObject



# Save the object to a local path
def saveObject(obj, filepath):
    """
    Save an object for the object or categorical dtype where key is the column name and value is the list of unqiue categorical values
    
    Params:\n
    `obj` : object\n
    `filepath` : filepath for saving the dictionary object\n
    """

    try:  
        joblib.dump(obj, filepath)
        logging.info(str(datetime.today()) + ' : Saved object')
    
    except Exception as e:
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

# Load the object from a local path
def loadObject(filepath):
    """
    Load an object for the object or categorical dtype where key is the column name and value is the list of unqiue categorical values
    
    Params:\n
    `filepath` : filepath for loading the dictionary object\n

    Returns object obj
    """

    obj = None

    try:
        obj = joblib.load(filepath)
        logging.info(str(datetime.today()) + ' : Loaded unique values dictionary object')

    except Exception as e:
        obj = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return obj