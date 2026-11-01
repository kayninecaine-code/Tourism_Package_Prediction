import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")
# CustomerID is an identifier and "Unnamed: 0" is a leftover row index; neither is predictive
df.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore", inplace=True)

# Data cleaning
df["Gender"] = df["Gender"].replace("Fe Male", "Female")  # fix the typo that splits one category in two
before = len(df)
df = df.drop_duplicates()                                  # exact duplicate customer records
print(f"Removed {before - len(df)} duplicate rows.")

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

target = "ProdTaken"  # 1 if the customer purchased the package, else 0
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
