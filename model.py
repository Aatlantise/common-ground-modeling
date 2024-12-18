import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
import random


def create_negative_samples(data, num_negatives=2):
    negative_samples = []
    for _, row in data.iterrows():
        for _ in range(num_negatives):
            # Randomly choose another row for unrelated referent/context
            random_row = data.sample(1).iloc[0]
            if row['referent'] != random_row['referent']:  # Ensure it's unrelated
                negative_samples.append({
                    "referent": row['referent'],
                    "anaphora": "",
                    "case": row['case'],
                    "humanness": row['humanness'],
                    "pronominality": row['pronominality'],
                    "word_d": random.randint(5, 50),  # Random distance
                    "clause_d": random.randint(1, 5),
                    **{k: random.randint(0, 5) for k in ["nomp", "accp", "oblp", "genp", "othp",
                                                        "nomn", "accn", "obln", "genn", "othn"]}
                })
    return pd.DataFrame(negative_samples)


# Step 1: Load the dataset
df = pd.read_csv("anaphoric_pair_distribution.tsv", sep="\t")  # Adjust separator as needed

# Generate synthetic negatives
negative_data = create_negative_samples(df)


# Step 2: Feature Engineering
# Select relevant columns
categorical_features = ["case", "humanness", "pronominality"]
numerical_features = ["word_d", "clause_d", "nomp", "accp", "oblp", "genp",
                      "othp", "nomn", "accn", "obln", "genn", "othn"]

# Encode categorical features using OneHotEncoder
ohe = OneHotEncoder()
encoded_categorical = pd.DataFrame(
    ohe.fit_transform(df[categorical_features]).toarray(),
    columns=ohe.get_feature_names_out(categorical_features)
)
negative_categorical = pd.DataFrame(
    ohe.fit_transform(negative_data[categorical_features]).toarray(),
    columns=ohe.get_feature_names_out(categorical_features)
)

# Combine numerical and encoded categorical features
X = pd.concat([encoded_categorical, df[numerical_features]], axis=1)
Y = pd.concat([negative_categorical, negative_data[numerical_features]], axis=1)

# Step 3: Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)

# Step 4: Train/Test Split
# Since this is unsupervised, we split to simulate "normal" vs "unseen" data
X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=42)

# Step 5: Train One-Class SVM
# nu: proportion of outliers expected, gamma: kernel parameter
svm_model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.1)  # Adjust nu for sensitivity
svm_model.fit(X_train)

# Step 6: Predictions
# Predict whether points are "inliers" (+1) or "outliers" (-1)
train_preds = svm_model.predict(X_train)
test_preds = svm_model.predict(X_test)
negative_preds = svm_model.predict(Y_scaled)

# Map predictions for clarity
def map_prediction(pred):
    return ["Inlier" if p == 1 else "Outlier" for p in pred]

print(f"Inlier prediction rate: {len([p for p in map_prediction(test_preds) if p == 'Inlier'])/len(test_preds)}")
print(f"Outlier prediction rate: {len([p for p in map_prediction(negative_preds) if p == 'Outlier'])/len(negative_preds)}")



