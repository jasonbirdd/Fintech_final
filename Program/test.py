# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 22:04:35 2022

@author: JasonJhan
"""
import joblib
import numpy as np
loaded_rf = joblib.load("./result/random_forest.joblib")

age = int(input("Age:"))
Diabetes = int(input("Diabetes:"))
BloodPressureProblems = int(input("BloodPressureProblems:"))
AnyTransplants = int(input("AnyTransplants:"))
AnyChronicDiseases = int(input("AnyChronicDiseases:"))
Height = int(input("Height:"))
Weight = int(input("Weight:"))
KnownAllergies = int(input("KnownAllergies:"))
HistoryOfCancerInFamily = int(input("HistoryOfCancerInFamily:"))
NumberOfMajorSurgeries = int(input("NumberOfMajorSurgeries:"))
bmi = Weight/(Height/100)**2

if bmi<24:
    bmi=0
elif 24<=bmi<27:
    bmi=1
elif 27<=bmi<30:
    bmi=2
elif 30<=bmi<35:
    bmi=3
else:
    bmi=4

age_new = 0
if age<20:
    age_new=0
elif 20<=bmi<40:
    age_new=1
elif 40<=bmi<65:
    age_new=2
else:
    age_new=3

price = [15000,16000,17000,18000, 19000,
         20000, 21000, 22000, 23000, 24000,
         25000, 26000, 27000, 28000, 29000,
         30000, 31000, 32000, 34000, 35000,
         36000, 38000, 39000, 40000]


pred = loaded_rf.predict(np.array([age,
                                Diabetes,
                                BloodPressureProblems,
                                AnyTransplants,
                                AnyChronicDiseases,
                                Height,
                                Weight,
                                KnownAllergies,
                                HistoryOfCancerInFamily,
                                NumberOfMajorSurgeries,
                                bmi,
                                age_new]).reshape(1,-1))
print(pred)
print(f"your predict price is {price[pred[0]]}")
