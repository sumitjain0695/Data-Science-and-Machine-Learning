from tkinter import *
import numpy as np
import pandas as pd
import json
import csv
import sqlite3
from itertools import chain
from prettytable import PrettyTable
import os
import matplotlib.pyplot as plt

l1 = [
    "back_pain",
    "constipation",
    "abdominal_pain",
    "diarrhoea",
    "mild_fever",
    "yellow_urine",
    "yellowing_of_eyes",
    "acute_liver_failure",
    "fluid_overload",
    "swelling_of_stomach",
    "swelled_lymph_nodes",
    "malaise",
    "blurred_and_distorted_vision",
    "phlegm",
    "throat_irritation",
    "redness_of_eyes",
    "sinus_pressure",
    "runny_nose",
    "congestion",
    "chest_pain",
    "weakness_in_limbs",
    "fast_heart_rate",
    "pain_during_bowel_movements",
    "pain_in_anal_region",
    "bloody_stool",
    "irritation_in_anus",
    "neck_pain",
    "dizziness",
    "cramps",
    "bruising",
    "obesity",
    "swollen_legs",
    "swollen_blood_vessels",
    "puffy_face_and_eyes",
    "enlarged_thyroid",
    "brittle_nails",
    "swollen_extremeties",
    "excessive_hunger",
    "extra_marital_contacts",
    "drying_and_tingling_lips",
    "slurred_speech",
    "knee_pain",
    "hip_joint_pain",
    "muscle_weakness",
    "stiff_neck",
    "swelling_joints",
    "movement_stiffness",
    "spinning_movements",
    "loss_of_balance",
    "unsteadiness",
    "weakness_of_one_body_side",
    "loss_of_smell",
    "bladder_discomfort",
    "foul_smell_of urine",
    "continuous_feel_of_urine",
    "passage_of_gases",
    "internal_itching",
    "toxic_look_(typhos)",
    "depression",
    "irritability",
    "muscle_pain",
    "altered_sensorium",
    "red_spots_over_body",
    "belly_pain",
    "abnormal_menstruation",
    "dischromic _patches",
    "watering_from_eyes",
    "increased_appetite",
    "polyuria",
    "family_history",
    "mucoid_sputum",
    "rusty_sputum",
    "lack_of_concentration",
    "visual_disturbances",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
    "coma",
    "stomach_bleeding",
    "distention_of_abdomen",
    "history_of_alcohol_consumption",
    "fluid_overload",
    "blood_in_sputum",
    "prominent_veins_on_calf",
    "palpitations",
    "painful_walking",
    "pus_filled_pimples",
    "blackheads",
    "scurring",
    "skin_peeling",
    "silver_like_dusting",
    "small_dents_in_nails",
    "inflammatory_nails",
    "blister",
    "red_sore_around_nose",
    "yellow_crust_ooze"
]
disease = [
    "Fungal infection",
    "Allergy",
    "GERD",
    "Chronic cholestasis",
    "Drug Reaction",
    "Peptic ulcer diseae",
    "AIDS",
    "Diabetes",
    "Gastroenteritis",
    "Bronchial Asthma",
    "Hypertension",
    " Migraine",
    "Cervical spondylosis",
    "Paralysis (brain hemorrhage)",
    "Jaundice",
    "Malaria",
    "Chicken pox",
    "Dengue",
    "Typhoid",
    "hepatitis A",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Alcoholic hepatitis",
    "Tuberculosis",
    "Common Cold",
    "Pneumonia",
    "Dimorphic hemmorhoids(piles)",
    "Heartattack",
    "Varicoseveins",
    "Hypothyroidism",
    "Hyperthyroidism",
    "Hypoglycemia",
    "Osteoarthristis",
    "Arthritis",
    "(vertigo) Paroymsal  Positional Vertigo",
    "Acne",
    "Urinary tract infection",
    "Psoriasis",
    "Impetigo"
]


data = ['Name of Patient',
        'Symptom_1',
        'Symptom_2',
        'Symptom_3',
        'Symptom_4',
        'Symptom_5',
        'Disease(predicted by Decision Tree)',
        'Disease(predicted by Random Forest)',
        'Disease(predicted by Naive Bayes)',
        ]
temp = []
temp1 = []


db_data = []
dis_DT = None
dis_RF = None
dis_NB = None

l2 = []
for x in range(0, len(l1)):
    l2.append(0)

# TESTING DATA df --------------------------------------------------------
df = pd.read_csv("Training.csv")

# print(df.head())

df.replace({"prognosis": {"Fungal infection": 0,
                          "Allergy": 1,
                          "GERD": 2,
                          "Chronic cholestasis": 3,
                          "Drug Reaction": 4,
                          "Peptic ulcer diseae": 5,
                          "AIDS": 6,
                          "Diabetes ": 7,
                          "Gastroenteritis": 8,
                          "Bronchial Asthma": 9,
                          "Hypertension ": 10,
                          "Migraine": 11,
                          "Cervical spondylosis": 12,
                          "Paralysis (brain hemorrhage)": 13,
                          "Jaundice": 14,
                          "Malaria": 15,
                          "Chicken pox": 16,
                          "Dengue": 17,
                          "Typhoid": 18,
                          "hepatitis A": 19,
                          "Hepatitis B": 20,
                          "Hepatitis C": 21,
                          "Hepatitis D": 22,
                          "Hepatitis E": 23,
                          "Alcoholic hepatitis": 24,
                          "Tuberculosis": 25,
                          "Common Cold": 26,
                          "Pneumonia": 27,
                          "Dimorphic hemmorhoids(piles)": 28,
                          "Heart attack": 29,
                          "Varicose veins": 30,
                          "Hypothyroidism": 31,
                          "Hyperthyroidism": 32,
                          "Hypoglycemia": 33,
                          "Osteoarthristis": 34,
                          "Arthritis": 35,
                          "(vertigo) Paroymsal  Positional Vertigo": 36,
                          "Acne": 37,
                          "Urinary tract infection": 38,
                          "Psoriasis": 39,
                          "Impetigo": 40}},
           inplace=True)

# print(df.head())

X = df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr -------------------------------------------------------
tr = pd.read_csv("Testing.csv")
tr.replace({"prognosis": {"Fungal infection": 0,
                          "Allergy": 1,
                          "GERD": 2,
                          "Chronic cholestasis": 3,
                          "Drug Reaction": 4,
                          "Peptic ulcer diseae": 5,
                          "AIDS": 6,
                          "Diabetes ": 7,
                          "Gastroenteritis": 8,
                          "Bronchial Asthma": 9,
                          "Hypertension ": 10,
                          "Migraine": 11,
                          "Cervical spondylosis": 12,
                          "Paralysis (brain hemorrhage)": 13,
                          "Jaundice": 14,
                          "Malaria": 15,
                          "Chicken pox": 16,
                          "Dengue": 17,
                          "Typhoid": 18,
                          "hepatitis A": 19,
                          "Hepatitis B": 20,
                          "Hepatitis C": 21,
                          "Hepatitis D": 22,
                          "Hepatitis E": 23,
                          "Alcoholic hepatitis": 24,
                          "Tuberculosis": 25,
                          "Common Cold": 26,
                          "Pneumonia": 27,
                          "Dimorphic hemmorhoids(piles)": 28,
                          "Heart attack": 29,
                          "Varicose veins": 30,
                          "Hypothyroidism": 31,
                          "Hyperthyroidism": 32,
                          "Hypoglycemia": 33,
                          "Osteoarthristis": 34,
                          "Arthritis": 35,
                          "(vertigo) Paroymsal  Positional Vertigo": 36,
                          "Acne": 37,
                          "Urinary tract infection": 38,
                          "Psoriasis": 39,
                          "Impetigo": 40}},
           inplace=True)

X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------


def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()
    clf3 = clf3.fit(X, y)

    # accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred = clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))
    # -----------------------------------------------------

    psymptoms = [
        Symptom1.get(),
        Symptom2.get(),
        Symptom3.get(),
        Symptom4.get(),
        Symptom5.get()]

    for k in range(0, len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted = predict[0]

    h = "no"
    for a in range(0, len(disease)):
        if(predicted == a):
            h = "yes"
            break

    if (h == "yes"):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")


def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X, np.ravel(y))

    # accuracy---------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred = clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))
    # -----------------------------------------------------

    psymptoms = [
        Symptom1.get(),
        Symptom2.get(),
        Symptom3.get(),
        Symptom4.get(),
        Symptom5.get()]

    for k in range(0, len(l1)):
        for z in psymptoms:
            if(z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted = predict[0]

    h = "no"
    for a in range(0, len(disease)):
        if(predicted == a):
            h = "yes"
            break

    if (h == "yes"):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")
    dis_RF = t2.get('1.0', END)
    print(dis_RF)


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb = gnb.fit(X, np.ravel(y))

    # accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred = gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))
    # -----------------------------------------------------

    psymptoms = [
        Symptom1.get(),
        Symptom2.get(),
        Symptom3.get(),
        Symptom4.get(),
        Symptom5.get()]
    for k in range(0, len(l1)):
        for z in psymptoms:
            if(z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted = predict[0]

    h = "no"
    for a in range(0, len(disease)):
        if(predicted == a):
            h = "yes"
            break

    if (h == "yes"):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")
    dis_NB = disease[a]

# gui=============================================================


root = Tk()
root.configure()

# entry var
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Name = StringVar()

# labels
NameLb = Label(root, text="Name of the Patient :")
NameLb.grid(row=6, column=0, pady=20, sticky=W)


S1Lb = Label(root, text="Symptom 1 :")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2 :")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3 :")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4 :")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5 :")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)


lrLb = Label(root, text="DecisionTree")
lrLb.grid(row=15, column=0, pady=10, sticky=W)

destreeLb = Label(root, text="RandomForest")
destreeLb.grid(row=17, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="NaiveBayes")
ranfLb.grid(row=19, column=0, pady=10, sticky=W)

OPTIONS = sorted(l1)

NameEn = Text(root, height=1, width=30, fg="black")
NameEn.grid(row=6, column=1)


S1En = OptionMenu(root, Symptom1, *OPTIONS)
S1En.grid(row=7, column=1)

try:
    OPTIONS2 = OPTIONS.remove(Symptom1.get())
except BaseException:
    OPTIONS2 = OPTIONS
S2En = OptionMenu(root, Symptom2, *OPTIONS2)
S2En.grid(row=8, column=1)

try:
    OPTIONS3 = OPTIONS2.remove(Symptom2.get())
except BaseException:
    OPTIONS3 = OPTIONS
S3En = OptionMenu(root, Symptom3, *OPTIONS3)
S3En.grid(row=9, column=1)

try:
    OPTIONS4 = OPTIONS3.remove(Symptom3.get())
except BaseException:
    OPTIONS4 = OPTIONS

S4En = OptionMenu(root, Symptom4, *OPTIONS4)
S4En.grid(row=10, column=1)

try:
    OPTIONS5 = OPTIONS4.remove(Symptom4.get())
except BaseException:
    OPTIONS5 = OPTIONS

S5En = OptionMenu(root, Symptom5, *OPTIONS5)
S5En.grid(row=11, column=1)

dst = Button(
    root,
    text="DecisionTree",
    command=DecisionTree,
    bg="green",
    fg="yellow")
dst.grid(row=8, column=3, padx=10)

rnf = Button(
    root,
    text="Randomforest",
    command=randomforest,
    bg="green",
    fg="yellow")
rnf.grid(row=9, column=3, padx=10)

lr = Button(
    root,
    text="NaiveBayes",
    command=NaiveBayes,
    bg="green",
    fg="yellow")
lr.grid(row=10, column=3, padx=10)

t1 = Text(root, height=1, width=40, fg="black")
t1.grid(row=15, column=1, padx=10)

t2 = Text(root, height=1, width=40, fg="black")
t2.grid(row=17, column=1, padx=10)

t3 = Text(root, height=1, width=40, fg="black")
t3.grid(row=19, column=1, padx=40)

# to display the contents of .db (database)


def show_db():
    crsr.execute("SELECT *FROM `disease`")
    result = crsr.fetchall()
    
    x = PrettyTable()
    x.field_names = ["P_name", "Symptom_1", "Symptom_2",
                     "Symptom_3", "Symptom_4", "Symptom_5",
                     "Disease_1","Disease_2","Disease_3"]
    
    for i in result:
        x.add_row(i)
    print(x)

# saves data into a csv file to be processed in pandas for further
# model training and visualisation
flag=0
try:
    da=pd.read_csv('new_file.csv')
    if da.empty:
        flag=1
except:
    flag=1
    
if flag == 1:
    with open("new_file.csv", "a", newline='') as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerow(data)

def onclick_pandas():

    temp = [NameEn.get('1.0', "end-1c"), Symptom1.get(), Symptom2.get(),
            Symptom3.get(), Symptom4.get(), Symptom5.get()]

    temp.append(t1.get('1.0', "end-1c"))
    temp.append(t2.get('1.0', "end-1c"))
    temp.append(t3.get('1.0', "end-1c"))
    with open("new_file.csv", "a", newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerow(temp)
    temp = list()


# database connection using SQLite3, for storing data into a .db file
connection = sqlite3.connect("dp.db")
crsr = connection.cursor()
crsr.execute("CREATE TABLE IF NOT EXISTS disease (pname VARCHAR(20), symp_1 VARCHAR(35),symp_2 VARCHAR(35),symp_3 VARCHAR(35),symp_4 VARCHAR(35),symp_5 VARCHAR(35),dis_1 VARCHAR(30),dis_2 VARCHAR(30),dis_3 VARCHAR(30))")

##click event for database entry
def onclick_sqlite():

    temp1 = [NameEn.get('1.0', "end-1c"), Symptom1.get(), Symptom2.get(),
             Symptom3.get(), Symptom4.get(), Symptom5.get()]

    temp1.append(t1.get('1.0', "end-1c"))
    temp1.append(t2.get('1.0', "end-1c"))
    temp1.append(t3.get('1.0', "end-1c"))

    crsr.execute(
        "INSERT INTO disease (pname,symp_1,symp_2,symp_3,symp_4,symp_5,dis_1,dis_2,dis_3)"
        "VALUES(?,?,?,?,?,?,?,?,?)", temp1)

    connection.commit()
    temp1 = list()


def onclick_reset():

    NameEn.delete('1.0', END)
    t1.delete('1.0', END)
    t2.delete('1.0', END)
    t3.delete('1.0', END)

    Symptom1.set(None)
    Symptom2.set(None)
    Symptom3.set(None)
    Symptom4.set(None)
    Symptom5.set(None)


connection = sqlite3.connect("dp.db")
crsr = connection.cursor()


# opens the csv file

def open_csv():
    os.startfile('new_file.csv')

##example matplotlib visualisation from the recorded data
def onclick_chart():
    df = pd.read_sql_query("SELECT dis_2,count(dis_2) FROM disease WHERE (symp_1='chest_pain') OR (symp_2='chest_pain') OR (symp_3='chest_pain') OR (symp_4='chest_pain') OR (symp_5='chest_pain') group by dis_2 HAVING count(dis_2) > 1",connection)
    #print(df)
    disease_data = df["dis_2"]
    count = df["count(dis_2)"]
    plt.pie(count, labels=disease_data,autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Chest Pain Diseases.")
    plt.show()

btn1 = Button(
    root,
    text="SAVE(pandas)",
    command=onclick_pandas,
    bg="green",
    fg="yellow")
btn1.grid(row=28, column=0, padx=40, pady=10)

btn2 = Button(
    root,
    text="SAVE(SQLite3)",
    command=onclick_sqlite,
    bg="green",
    fg="yellow")
btn2.grid(row=28, column=1, padx=40, pady=10)


btn3 = Button(
    root,
    text="open(csv)",
    command=open_csv,
    bg="green",
    fg="yellow")
btn3.grid(row=29, column=0, padx=40, pady=10)

btn5 = Button(
    root,
    text="REFRESH",
    command=onclick_reset,
    bg="green",
    fg="yellow")
btn5.grid(row=30, column=0, padx=40, pady=10)

btn4 = Button(root, text="show DB", command=show_db, bg="green", fg="yellow")
btn4.grid(row=29, column=1, padx=40, pady=10)

btn6 = Button(root, text="CHEST-PAIN CHART", command=onclick_chart, bg="blue", fg="yellow")
btn6.grid(row=30, column=1, padx=40,pady=10)

root.mainloop()
crsr.close()
connection.close()

