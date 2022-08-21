import pandas
import json, sys
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

# with open("vol_org_sL.json","r") as f:
#     dfDict = json.load(f)

# for i in dfDict["Volume"].keys():
#     if dfDict["Volume"][i] == "nan":
#         dfDict["Volume"][i] = 0.0

# df = pandas.DataFrame(dfDict)
# df = df.astype({"Volume":float,"Organ":str,"Slice Location":float})
# print(df)
# keysOI = ["Liver","Body","Lung","Heart","Kidney","Spleen","IVC","Aorta"]


# for subOrg in keysOI:
#     subDf = df[df["Organ"] == subOrg]

#     cat1 = subDf[subDf["Wrong Location"] == "True"]
#     cat2 = subDf[subDf["Wrong Location"] == "False"]

#     sig = ttest_ind(cat1["Volume"],cat2["Volume"])

#     # fig = plt.figure()
#     # sns.set_theme("talk")
#     # sns.set_style("darkgrid")
#     plt.title(subOrg)
#     sns.boxplot(data=subDf,x="Wrong Location",y="Volume",showfliers = False)
#     print("{} {}".format(subOrg,sig.pvalue))

#     # axes = fig.get_axes()

#     # x1, x2 = 0, 1   
#     # y = np.max([cat1.quantile(0.75) + 1.5*(cat1.quantile(0.75)-cat1.quantile(0.25)),cat2.quantile(0.75) + 1.5*(cat2.quantile(0.75)-cat2.quantile(0.25))])
#     # y, h, col = y + 1000, axes.get_ylim()[1]-0.1*axes.get_ylim()[1], 'k'
#     # plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#     # if sig.pvalue < 0.005:
#     #     plt.text((x1+x2)*.5, y+h//10, "*", ha='center', va='bottom', color=col)
#     # else:
#     #     plt.text((x1+x2)*.5, y+h//10, "ns", ha='center', va='bottom', color=col)

#     plt.savefig("./Figures/{}.png".format(subOrg))
#     plt.close("all")

####################################################################

####################################################################

with open("vol_org_sL_norm.json","r") as f:
    dfDict = json.load(f)

for i in dfDict["Norm. Volume"].keys():
    if dfDict["Norm. Volume"][i] == "nan":
        dfDict["Norm. Volume"][i] = 0.0

df = pandas.DataFrame(dfDict)
df = df.astype({"Norm. Volume":float,"Organ":str,"Slice Location":float})
print(df)
keysOI = ["Liver","Lung","Heart","Kidney","Spleen","IVC","Aorta"]

# for subOrg in keysOI:
#     subDf = df[df["Organ"] == subOrg]

#     cat1 = subDf[subDf["Wrong Location"] == "True"]
#     cat2 = subDf[subDf["Wrong Location"] == "False"]

#     sig = ttest_ind(cat1["Norm. Volume"],cat2["Norm. Volume"])

#     # fig = plt.figure()
#     # sns.set_theme("talk")
#     # sns.set_style("darkgrid")
#     plt.title(subOrg)
#     sns.boxplot(data=subDf,x="Wrong Location",y="Norm. Volume",showfliers = False)
#     print("{} {}".format(subOrg,sig.pvalue))

#     # axes = fig.get_axes()

#     # x1, x2 = 0, 1   
#     # y = np.max([cat1.quantile(0.75) + 1.5*(cat1.quantile(0.75)-cat1.quantile(0.25)),cat2.quantile(0.75) + 1.5*(cat2.quantile(0.75)-cat2.quantile(0.25))])
#     # y, h, col = y + 1000, axes.get_ylim()[1]-0.1*axes.get_ylim()[1], 'k'
#     # plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#     # if sig.pvalue < 0.005:
#     #     plt.text((x1+x2)*.5, y+h//10, "*", ha='center', va='bottom', color=col)
#     # else:
#     #     plt.text((x1+x2)*.5, y+h//10, "ns", ha='center', va='bottom', color=col)

#     plt.savefig("./Figures/{}_norm.png".format(subOrg))
#     plt.close("all")
# # sns.scatterplot(data=df,x="Slice Location",y="Volume",hue="Organ")
# # plt.show()

newDf = df[df["Organ"] != "Body"]
sns.boxplot(data=newDf,y="Norm. Volume",x="Organ",hue="Wrong Location",showfliers=False,linewidth=2.5)
plt.ylim([-0.05,0.6])
plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6])
plt.savefig("./Figures/Bplot_norm.png")
plt.close("all")

# df = pandas.read_json("vol_ratio.json")
# df = df.fillna(0.0)

# cat1 = df[df["Wrong Location"] == True]
# cat2 = df[df["Wrong Location"] == False]

# print(cat1)
# print(ttest_ind(cat1["Liver to Body ratio"],cat2["Liver to Body ratio"]))

# # sns.set_theme("talk")
# # sns.set_style("darkgrid")
# sns.boxplot(data=df,y="Liver to Body ratio",x="Wrong Location",showfliers = False)
# plt.ylim([0,1])
# plt.show()
