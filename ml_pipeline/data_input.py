# 'pip install xlrd' for excel files
import json
import logging
from datetime import datetime
import pandas as pd
import pymssql
import pymongo
import requests

with open("config.json") as g:
        gdata = json.load(g)

# Basic Logger
logging.basicConfig(filename = gdata["logfilepath"], level = logging.INFO)

def inputData(sourceType, fileIndex = 0, connIndex = 0, strSelectQuery = None, dictFindQuery = None, dictProjectQuery = None, dictSortQuery = None, numLimit = None, queryIndex = 0, endpointIndex = 0, headerslist = []):
    """
    Get the data from various sources as input for the machine learning model.

    Params:\n
    `sourceType` : "csv" or "tsv" or "excel" or "json" or "sqldatabase" or "nosqldatabase" or "restapi" based on the config.json file
    `fileIndex` : file index in the config.json file if the sourceType is "csv", "tsv", "excel" or "json"
    `connIndex` : connection index in the config.json file if the sourceType is "sqldatabase" or "nosqldatabase"
    `strSelectQuery` : custom select query if the sourceType is "sqldatabase"
    `dictFindQuery` : custom find query if the sourceType is "nosqldatabase"
    `dictProjectQuery` : custom project query if the sourceType is "nosqldatabase"
    `dictSortQuery` : custom sort query if the sourceType is "nosqldatabase"
    `numLimit` : number of rows if the sourceType is "nosqldatabase"
    `queryIndex` : query index in the config.json file if the sourceType is "sqldatabase" or "nosqldatabase"
    `endpointIndex` : endpoint index in the config.json file if the sourceType is "restapi"
    `headerslist` : REST API header list in the config.json file if the sourceType is "restapi"

    Returns pandas.DataFrame
    """

    resultingDataFrame = None

    try:
        if sourceType == "csv":
            resultingDataFrame = csvInput(fileIndex)
        elif sourceType == "tsv":
            resultingDataFrame = tsvInput(fileIndex)
        elif sourceType == "excel":
            resultingDataFrame = excelInput(fileIndex)
        elif sourceType == "json":
            resultingDataFrame = jsonInput(fileIndex)
        elif sourceType == "sqldatabase":
            resultingDataFrame = sqldatabaseInput(connIndex, strSelectQuery, queryIndex)
        elif sourceType == "nosqldatabase":
            resultingDataFrame = nosqldatabaseInput(connIndex,dictFindQuery,dictProjectQuery, dictSortQuery, numLimit, queryIndex)
        elif sourceType == "restapi":
            resultingDataFrame = restapiInput(endpointIndex, headerslist)
        else:
            raise Exception("sourceType invalid")

    except Exception as e:
        resultingDataFrame = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))
    
    return resultingDataFrame


def csvInput(fileIndex = 0):

    with open("config.json") as f:
        data = json.load(f)

    csvfilepath = data["csv"]["files"][fileIndex]["file"]
    logging.info(str(datetime.today()) + ' : CSV File')
    
    rs = None

    try:
        rs = pd.read_csv(csvfilepath)
        logging.info(str(datetime.today()) + ' : Got data')

    except Exception as e:
        rs = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return rs


def tsvInput(fileIndex = 0):
    
    with open("config.json") as f:
        data = json.load(f)

    tsvfilepath = data["tsv"]["files"][fileIndex]["file"]
    logging.info(str(datetime.today()) + ' : TSV File')
    
    rs = None

    try:
        rs = pd.read_csv(tsvfilepath, sep = '\t')
        logging.info(str(datetime.today()) + ' : Got data')

    except Exception as e:
        rs = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))
        
    return rs


def excelInput(fileIndex = 0):

    with open("config.json") as f:
        data = json.load(f)

    excelfilepath = data["excel"]["files"][fileIndex]["file"]
    logging.info(str(datetime.today()) + ' : Excel File')
    
    rs = None

    try:
        rs = pd.read_excel(excelfilepath)
        logging.info(str(datetime.today()) + ' : Got data')

    except Exception as e:
        rs = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return rs


def jsonInput(fileIndex = 0):

    with open("config.json") as f:
        data = json.load(f)

    jsonfilepath = data["json"]["files"][fileIndex]["file"]
    logging.info(str(datetime.today()) + ' : JSON File')
    
    rs = None

    try:
        rs = pd.read_json(jsonfilepath)
        logging.info(str(datetime.today()) + ' : Got data')

    except Exception as e:
        rs = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return rs


def getColumnNames(cursorobject):
    colNameList = []

    for i in range(len(cursorobject.description)):
        desc = cursorobject.description[i]
        colNameList.append(desc[0])

    return colNameList

def getColumnValues(datalist, index):
    ls = []
    for i in range(len(datalist)):
        ls.append(datalist[i][index])
        
    return ls

def sqldatabaseInput(connIndex = 0, strSelectQuery = None, queryIndex = 0):

    with open("config.json") as f:
        data = json.load(f)

    connection1 = data["sqldatabase"]["connections"][connIndex]
    servername = connection1["servername"]
    username = connection1["username"]
    userpassword = connection1["userpassword"]
    databasename = connection1["databasename"]
    logging.info(str(datetime.today()) + ' : SQL Database')
    
    rs = None

    try:
        con = pymssql.connect(server = servername, user = username, password = userpassword, database = databasename)
        cur = con.cursor()
        logging.info(str(datetime.today()) + ' : Database Connected')

        # Retrieve all data using query
        if strSelectQuery == None:
            querystring = connection1["queries"][queryIndex]["query"]
        else:
            querystring = strSelectQuery
            
        cur.execute(querystring)

        columns = getColumnNames(cur)
        rows = list(cur)

        cur.close()
        del cur
        con.close()

        rs = pd.DataFrame()

        for i in range(len(columns)):
            rs[columns[i]] = getColumnValues(rows, i)
        logging.info(str(datetime.today()) + ' : Got data')

    except Exception as e:
        rs = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return rs


def nosqldatabaseInput(connIndex = 0, dictFindQuery = None, dictProjectQuery = None, dictSortQuery = None, numLimit = None, queryIndex = 0):

    with open("config.json") as f:
        data = json.load(f)

    connection1 = data["nosqldatabase"]["connections"][connIndex]
    connectionurl = connection1["connectionurl"]
    logging.info(str(datetime.today()) + ' : NoSQL Database')
    
    rs = None

    try:
        client = pymongo.MongoClient(connectionurl)
        logging.info(str(datetime.today()) + ' : Database Connected')

        # Retrieve all data using query
        databasename = connection1["databasename"]
        db = client[databasename]

        collectionname = connection1["collectionname"]
        collection = db[collectionname]

        if (dictFindQuery == None) & (dictProjectQuery ==  None):
            findQuery = connection1["queries"][queryIndex]["find"]
            projectQuery = connection1["queries"][queryIndex]["project"]
            dictionaryFind = json.loads(findQuery)
            dictionaryProject = json.loads(projectQuery)
        elif (dictFindQuery != None) & (dictProjectQuery ==  None):
            projectQuery = connection1["queries"][queryIndex]["project"]
            dictionaryFind = dictFindQuery
            dictionaryProject = json.loads(projectQuery)
        elif (dictFindQuery == None) & (dictProjectQuery !=  None):
            findQuery = connection1["queries"][queryIndex]["find"]
            dictionaryFind = json.loads(findQuery)
            dictionaryProject = dictProjectQuery
        else:
            dictionaryFind = dictFindQuery
            dictionaryProject = dictProjectQuery
        
        
        if (dictSortQuery == None) & (numLimit == None):
            sortQuery = connection1["queries"][queryIndex]["sort"]
            sortQuery = json.loads(sortQuery)

            limitQuery = connection1["queries"][queryIndex]["limit"]

            if (sortQuery != {}) & (limitQuery != 0):
                sortList = []

                for key,value in sortQuery.items():
                    if sortQuery[key] == -1:
                        sortList.append((key,pymongo.DESCENDING))
                    else:
                        sortList.append((key,pymongo.ASCENDING))

                res = collection.find(dictionaryFind, dictionaryProject).sort(sortList).limit(limitQuery)

            elif (sortQuery != {}) & (limitQuery == 0):

                sortList = []

                for key,value in sortQuery.items():
                    if sortQuery[key] == -1:
                        sortList.append((key,pymongo.DESCENDING))
                    else:
                        sortList.append((key,pymongo.ASCENDING))

                res = collection.find(dictionaryFind, dictionaryProject).sort(sortList)

            elif (sortQuery == {}) & (limitQuery != 0):
                res = collection.find(dictionaryFind, dictionaryProject).limit(limitQuery)

            else:
                res = collection.find(dictionaryFind, dictionaryProject)

        elif (dictSortQuery != None) & (numLimit == None):
            sortQuery = dictSortQuery
            limitQuery = connection1["queries"][queryIndex]["limit"]

            sortList = []

            for key,value in sortQuery.items():
                if sortQuery[key] == -1:
                    sortList.append((key,pymongo.DESCENDING))
                else:
                    sortList.append((key,pymongo.ASCENDING))

            if (limitQuery != 0):
                res = collection.find(dictionaryFind, dictionaryProject).sort(sortList).limit(limitQuery)
            else:
                res = collection.find(dictionaryFind, dictionaryProject).sort(sortList)
            
        elif (dictSortQuery == None) & (numLimit != None):
            sortQuery = connection1["queries"][queryIndex]["sort"]
            sortQuery = json.loads(sortQuery)
            
            if (sortQuery != {}):

                sortList = []

                for key,value in sortQuery.items():
                    if sortQuery[key] == -1:
                        sortList.append((key,pymongo.DESCENDING))
                    else:
                        sortList.append((key,pymongo.ASCENDING))

                res = collection.find(dictionaryFind, dictionaryProject).sort(sortList).limit(numLimit)
            else:
                res = collection.find(dictionaryFind, dictionaryProject).limit(numLimit)

        else:
            sortQuery = dictSortQuery
            sortList = []

            for key,value in sortQuery.items():
                if sortQuery[key] == -1:
                    sortList.append((key,pymongo.DESCENDING))
                else:
                    sortList.append((key,pymongo.ASCENDING))

            res = collection.find(dictionaryFind, dictionaryProject).sort(sortList).limit(numLimit)

        reslist = list(res)
        
        rs = pd.DataFrame(reslist)
        logging.info(str(datetime.today()) + ' : Got data')

    except Exception as e:
        rs = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return rs


def restapiInput(endpointIndex = 0, headerslist = []):
    
    with open("config.json") as f:
        data = json.load(f)

    endpoint1 = data["restapi"]["endpoints"][endpointIndex]
    # api_token = 'your_api_token'
    endpointUrl = endpoint1["url"]
    requestMethod = endpoint1["method"]
    
    logging.info(str(datetime.today()) + ' : REST API')
    
    rs = None

    try:
        if requestMethod == "GET":
            response = requests.get(endpointUrl) # , headers=headers

            res = None
            if response.status_code == 200:
                res = json.loads(response.content.decode('utf-8'))
            else:
                raise Exception("Response Status Code is not 200")
        
            reslist = res
            if type(res) == type({}):
                tmplist = []
                tmplist.append(res)
                reslist = tmplist

            rs = pd.DataFrame(reslist)

        elif requestMethod == "POST":
            headers = {}
            for i in range(len(headerslist)):
                if headerslist[i] == 'Authorization':
                    headers[headerslist[i]] = 'Bearer ' + str(endpoint1["headers"][i][headerslist[i]])
                else:
                    headers[headerslist[i]] = endpoint1["headers"][i][headerslist[i]]

            response = requests.post(endpointUrl, data = endpoint1["body"], headers = headers) # , headers=headers

            res = None
            if response.status_code == 200:
                res = json.loads(response.content.decode('utf-8'))
            else:
                raise Exception("Response Status Code is not 200")
        
            reslist = res
            if type(res) == type({}):
                tmplist = []
                tmplist.append(res)
                reslist = tmplist

            rs = pd.DataFrame(reslist)

        else:
            raise Exception("Request Method is invalid")
        logging.info(str(datetime.today()) + ' : Got data')

    except Exception as e:
        rs = None
        logging.exception(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

    return rs