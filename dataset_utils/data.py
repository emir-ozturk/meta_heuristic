import pandas as pd

def load_diabetes_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv('./datasets/diabetes.csv')
    attributes = df.drop("Outcome", axis=1)
    target = df["Outcome"]
    
    return attributes, target

def load_titanic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv('./datasets/titanic.csv')
    # Sonuca etki etmeyen değişkenler çıkarılır
    df.drop(columns=["Name", "Ticket", "Cabin", "Embarked", "PassengerId"], inplace=True)
    # Değişkenlerin veri tipi dönüştürülür
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    # Null değerleri silinir
    df.dropna(inplace=True)

    attributes = df.drop("Survived", axis=1)
    target = df["Survived"]

    return attributes, target

def load_cinnamon_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv('./datasets/balanced_cinnamon_quality.csv')
    df.drop(columns=["Sample_ID"], inplace=True)
    df["Quality_Label"] = df["Quality_Label"].map({"Low": 0, "Medium": 1, "High": 2})

    attributes = df.drop("Quality_Label", axis=1)
    target = df["Quality_Label"]

    return attributes, target
