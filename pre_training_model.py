import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,classification_report,precision_score,f1_score,confusion_matrix


#1. Load data
df = pd.read_csv(r"D:\ai_cybersecurity\ai_soc_synthetic_dataset_master.csv")

df["protocol_enc"]  = (df["protocol"]== "TCP").astype(int)

feature_cols = [
    "dst_port", "protocol_enc", "duration_sec", "packet_count", "byte_count",
    "packet_rate", "byte_rate", "avg_packet_size",
    "connection_count_60s", "unique_dst_ports_60s",
]




#2. Seperate X and y features and use train_test_split
X = df[["connection_count_60s"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#3. Train the model
model = XGBClassifier(n_estimators = 200,max_depth = 5,learning_rate=0.1,eval_metric="logloss",random_state=42)
model.fit(X_train,y_train)


#4 Evaluate
y_pred = model.predict(X_test)
print("================================================================")
print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 score :", f1_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, target_names=["normal", "attack"]))
print("================================================================")


df.groupby("label")["packet_rate"].plot(
    kind="hist",
    alpha=0.5,
    bins=30
)

plt.xlabel("Packet Rate")
plt.ylabel("Frequency")
plt.title("Packet Rate: Normal vs Attack")
plt.show()