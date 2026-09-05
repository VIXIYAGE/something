import json
import pandas as pd

# 1. Load the raw JSON file
# with open("data/extract2_.json", "r") as f:
#     json_data = json.load(f)

# 2. Convert ONLY the 'DETAIL' list into your DataFrame
df = pd.read_json("backend/data/ytr2.json")

# 3. Print the recommended encoding column cleanly
#print(df["recommended_encoding"])

print(df.shape)
# #coloumn = name,age,gender,sleep_hours,screen_time_hours,exercise_hours,stress_level,social_media_hours,mood_score,behaviour_type

# df = pd.read_json("data/ytr2.json")
# print(len(dfi.columns))
# print(len(df))



# print(df.shape)


# mapping = {
#     "Healthy": 5,0-
#     "Balanced": 4,
#     "Moderate": 3,
#     "Stressed": 1,
#     "Anxious": 2
# }
# df=pd.read_csv("YTR.csv")

# print(df["liked"].unique())
# df = df.dropna(subset=["liked"])

# df["liked"] = df["liked"].replace({
#     "yes": 1,
#     "no": 0,
#     "1": 1,
#     "0": 0
# })

# df["liked"] = pd.to_numeric(
#     df["liked"],
#     errors="coerce"
# )

# df = df[df["liked"].isin([0,1])]
# dc = pd.DataFrame(
#     dr["behaviour_type"]
# )
# dr["encoded"]=dr["behaviour_type"].map(mapping)
# print(dr[["name","behaviour_type","encoded"]])

# encoded = pd.get_dummies(dc)
# print(dc.shape)
# print(encoded)
# print(dr.groupby('behaviour_type')
# ['stress_level'].agg(['mean','std','min','max']))

# a=dr["behaviour_type","name"].astype("category").cat.codes
# print(a)
# print(dr.columns)
#print(dr.describe)
#print(dr["age"]>23)
# a= np.array([[2,4,5],[3,6,7],[6,34,5]])
# b= np.array([[0,4,5],[3,9,2],[4,5,6]])
# c=a.flatten()
# d=b.reshape(9,1)
# print(a.shape)
# print(b.shape)
# print(np.dot(c,d))

# print(np.random.rand(2,3))
# print(np.random.choice(a.flatten(),4))
# np.random.shuffle(a)
# print(a)
# np.random.seed(56)
# print(np.random.rand(4))


# #a=a.reshape(1,9)
# print(a.shape)
# print(a)
# print(a.ndim)


# d  = np.array([[[1, 2], [3, 4]],
#               [[5, 6], [7, 8]]],dtype=float)


# shape: (2, 2, 2)
# print(d.shape)

# print(d.ndim)
# print(d[1][1][0])
# print(dtype(d))
# print(d[0][0])
# print(d[0][0][0])
# print(d[1,1,0])



