# -*- coding: utf-8 -*-
"""

Predicts Electricity Demand

"""
import numpy as np
import pandas as pd
from sklearn import linear_model

# prepareData: preprocess and split the data into training, validation, and test sets
def prepareData(data):
    data = defineData(data)
    summerdata = preprocess(extractSummer(data))
    splits = splitData(summerdata)
    return splits

# getCombo: get all combinations of variables
def getCombo(vars):
    if len(vars) == 0:
        return [[]]
    combo = []
    for c in getCombo(vars[1:]):
        combo += [c, c + [vars[0]]]

    return combo
        
# createModel: runs several different model versions and determine the one that gives the best result.
# Returns the best model.
def createModel(trainingData, validationData, drivers):
    vars = getCombo(drivers)
    vars.remove([])
    scores = np.zeros(len(vars))
    model = {}
    for i in range(0, len(vars)):
        model[i] = train(trainingData, vars[i], "Ontario_Demand")
        validation_predicted = validate(model[i], validationData, vars[i])
        scores[i] = evaluate(validation_predicted, validationData["Ontario_Demand"])
    
    bestScoreIndex = np.argmax(scores)
    
    model[bestScoreIndex]["vars"] = vars[bestScoreIndex]
    return model[bestScoreIndex]
        
# defineData: Specify and define the type(s) of the column(s) of the data
def defineData(data):
    data["Date"] = pd.to_datetime(data["Date"])
    return data

# getHolidays: returns the dates of the holidays and weekends
def getHolidays():
    summer_holidays = ["2016-07-01", "2017-07-03", "2018-07-02", "2019-07-01", "2020-07-01",
                      "2016-08-01", "2017-08-07", "2018-08-06", "2019-08-05", "2020-08-03"]
    return summer_holidays
    
# extractSummer: extract the summer data
def extractSummer(data):
    months = [7,8]
    extract = data[data["Date"].map(lambda x: x.month).between(months[0], months[1])]
    return extract
    
# preprocess: preprocess the data
def preprocess(data):
    holidays = getHolidays()
    
    def mapWeekdayType(weekday, date):
        if weekday in ["Saturday", "Sunday"] or date in holidays:
            return 0
        return 1
    
    data["WeekdayType"] = list(map(lambda x,y: mapWeekdayType(x, y), data["Weekday"], data["Date"]))
    data["CDH"] = list(map(lambda x: max(x - 18, 0.0), data["Temperature"]))
    return data

# splitData: splits the data into training, validation, and test sets
def splitData(data):
    train = data[data["Date"].map(lambda x: x.year).between(2016, 2018)].copy().reset_index()
    validate = data[data["Date"].map(lambda x: x.year) == 2019].copy().reset_index()
    test = data[data["Date"].map(lambda x: x.year) == 2020].copy().reset_index()
    return {"train": train, "validate": validate, "test": test}
    
# train: train the model with the training data
def train(data, x, y):
    
    # fit data with linear model
    model = linear_model.LinearRegression()
    model.fit(data[x], data[y])
    
    # get the residuals
    residuals = pd.DataFrame()
    residuals["Hour"] = data["Hour"]
    residuals["WeekdayType"] = data["WeekdayType"]
    residuals["Residuals"] = data[y] - model.predict(data[x])
    
    # get baseline drift
    baseComponent = residuals.groupby(["Hour", "WeekdayType"])["Residuals"].mean()
    baseComponent.columns = ["Residuals"]
    baseComponent = baseComponent.reset_index()
    
    return {"linreg": model, "component": baseComponent}
    
# validate: validate the model with the validation set
def validate(model, data, x):
    return test(model, data, x)
    
# test: test the model with the test set
def test(model, data, x):
    # create baseline values from the data
    base_data = model["component"]
    
    def getBaseValue(k):
        return base_data[base_data["Hour"] == data["Hour"][k]][base_data["WeekdayType"] == data["WeekdayType"][k]]["Residuals"]
        
    base_predict = np.fromiter((map(getBaseValue, range(0, data.shape[0]))), dtype = 'float64')
    
    predicted = model["linreg"].predict(data[x]) + base_predict
    return predicted

# evaluate: determine how close the predicted data are to the expected data
def evaluate(predicted, expected):
    diff = expected - predicted
    return sum(abs(diff) < 500)/len(diff)
   
# run: runs the entire modelling process including training, validation, and test
def run():
    drivers = ["CDH", "WeekdayType", "Hour", "Relative_Humidity", "Dew_Point", "Wind_Speed", "Humidex"]
    # load data
    data = pd.read_csv("Sample Dataset.csv")
    splits = prepareData(data)
    model = createModel(splits["train"], splits["validate"], drivers)
    test_predicted = test(model, splits["test"], model["vars"])
    test_score = evaluate(test_predicted, splits["test"]["Ontario_Demand"])
    return model, splits, test_predicted, test_score


    
    
    


