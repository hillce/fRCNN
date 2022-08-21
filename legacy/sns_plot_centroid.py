from operator import sub
import pandas
import json, sys, copy
import numpy as np
from pandas.core.algorithms import quantile
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import ttest_ind

sns.set_theme("talk")
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = [6.4*3,4.8*3]
sns.set(font_scale=2.5)

with open("centroid_org_sL_norm.json","r") as f:
    dfDict = json.load(f)

df = pandas.DataFrame(dfDict)
df = df.astype({"X":float,"Y":float,"Organ":str,"Slice Location":float})
print(df)
keysOI = ["Liver","Lung","Heart","Kidney","Spleen","IVC","Aorta"]

bodyDf = df[df["Organ"] == "Body"]
for i,j in bodyDf.iterrows():
    df = df[(df["X"] == -j["X"]) and (df["Eid"] == j["Eid"])]
    df = df[(df["Y"] == -j["Y"]) and (df["Eid"] == j["Eid"])]

print(df)
for kOI in keysOI:
    subDf = df[df["Organ"] == kOI]
    # subCopy = copy.deepcopy(subDf)
    # for i,j in subCopy.iterrows():
    #     tempDf = bodyDf[bodyDf["Eid"] == j["Eid"]]

    #     x = list(tempDf["X"])[0]
    #     y = list(tempDf["Y"])[0]

    #     if (j["X"] == -x) and (j["Y"] == -y):
    #         subDf.drop(i)
    #         subDf.remo
    
    # print(subDf)
    sns.histplot(data=subDf,x="X",y="Y",hue="Wrong Location")
    plt.title(kOI)
    plt.xlim([-100,100])
    plt.ylim([-100,100])
    plt.show()