'''
小登对学长耐心批改、给出建议表示万分感谢。

本小登无任何Python基础，本题绝大部分由 AI 辅助学习，另外还观看了部分题目中给出的B站教学视频合集、查阅了相关资料。

源代码、学习笔记所有内容均为古法手搓，不存在任何复制粘贴情况.
'''
# 请从项目根目录 lingrui_ml_05 运行本代码。

import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class TitanicModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def predict_passenger(model, total_data):
    print("Please input passenger data")

    passenger_id = int(input("PassengerId:"))

    pclass = int(input("Pclass:"))
    if pclass not in [1, 2, 3]:
        raise ValueError("Pclass must be 1,2,3")
    
    name = input("Name")

    sex = input("Sex:")
    if sex not in ["male", "female"]:
        raise ValueError("Sex must be male/female")
    
    print("If Age is missing,please press Enter in the next step")
    age_input = input("Age:")
    if age_input == "":
        age = None
    else:
        age = float(age_input)
    if not pd.isna(age) and age < 0:
        raise ValueError("Age must be non-negative")

    sibsp = int(input("SibSp:"))
    if sibsp< 0:
        raise ValueError("SibSp is negative")

    parch = int(input("Parch:"))
    if parch < 0:
        raise ValueError("Parch is negative")

    ticket = input("Ticket:")

    fare = float(input("Fare:"))
    if  fare < 0:
        raise ValueError("Fare is negative")

    print("If Cabin is missing,please press Enter in the next step")
    cabin_input = input("Cabin:")
    if cabin_input == "":
        cabin = None
    else:
        cabin = cabin_input

    print("If Embarked is missing,please press Enter in the next step")
    embarked_input = input("Embarked:")
    if embarked_input == "":
        embarked = None
    else:
        embarked = embarked_input
    if (
        not pd.isna(embarked)
        and embarked not in ["S", "Q", "C"]
        ):
        raise ValueError("Embarked must be S, Q, C or None")

    print("\n")

    passenger_data = {
    "PassengerId": passenger_id,
    "Pclass": pclass,
    "Name": name,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Ticket": ticket,
    "Fare": fare,
    "Cabin": cabin,
    "Embarked": embarked
    }
    new_data = pd.DataFrame([passenger_data])

    new_data["Age"] = new_data["Age"].fillna(total_data["age_median"])
    new_data["Age"] = new_data["Age"].astype(float)

    new_data["Embarked"] = new_data["Embarked"].fillna(total_data["embarked_mode"])

    new_data["Has_Cabin"] = new_data["Cabin"].notna().astype(int)

    new_data["Sex"] = (new_data["Sex"] == "female").astype(int)

    new_data["Embarked_S"] = (new_data["Embarked"] == "S").astype(int)
    new_data["Embarked_C"] = (new_data["Embarked"] == "C").astype(int)
    new_data["Embarked_Q"] = (new_data["Embarked"] == "Q").astype(int)

    new_data = new_data.drop(columns=["Cabin", "PassengerId", "Name", "Ticket", "Embarked"])

    for column in ["Age", "Parch", "Fare", "SibSp"]:
        column_mean = total_data["standardization_data"][column]["mean"]
        column_std = total_data["standardization_data"][column]["std"]

        new_data[column] = (new_data[column] - column_mean) / column_std

    new_data = new_data[total_data["feature_order"]]
    new_data = new_data.astype(float)

    new_tensor = torch.tensor(new_data.to_numpy(), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output_new = model(new_tensor)
        probability = torch.sigmoid(output_new)
        prediction = (probability >= 0.5).float()
        prediction = int(prediction.item())

    return probability.item(),prediction

def main():

    # Read data

    data = pd.read_csv("data/train.csv")

    data.info()
    print(data.shape, "\n")
    
    print(data["Sex"].value_counts(dropna=False), data["Embarked"].value_counts(dropna=False), data["Pclass"].value_counts(), "\n", sep = "\n\n")

    seed = 88
    torch.manual_seed(seed)
    
    # Split training and test data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed, stratify=data["Survived"])

    # Observe missing values
    age = train_data["Age"]

    print("Age mean:", age.mean(), "\t" 
          "Age median:", age.median(), "\t"
          "Age min:", age.min(), "\t"
          "Age max", age.max(),"\n")

    age_median = age.median()
    embarked_mode = train_data["Embarked"].mode().iloc[0]

    for _data in [train_data, test_data]:
        #Handle missing values
        _data["Age"] = _data["Age"].fillna(age_median).astype(float)
        _data["Embarked"] = _data["Embarked"].fillna(embarked_mode)
    
        # Feature engineering

        _data["Has_Cabin"] = _data["Cabin"].notna().astype(int)

        _data["Sex"] = (_data["Sex"] == "female").astype(int)
        _data["Embarked_S"] = (_data["Embarked"] == "S").astype(int)
        _data["Embarked_C"] = (_data["Embarked"] == "C").astype(int)
        _data["Embarked_Q"] = (_data["Embarked"] == "Q").astype(int)

        _data.drop(columns=["Cabin","PassengerId", "Name", "Ticket", "Embarked"], inplace=True)
    

    # Standardization
    X_train = train_data.drop(columns=["Survived"])
    y_train = train_data["Survived"]
    X_test = test_data.drop(columns=["Survived"])
    y_test = test_data["Survived"]

    print(train_data[["Pclass", "Age", "Parch", "Fare", "SibSp"]].describe())
    print("\n")

    standardization_data = {}

    for column in ["Age", "Parch", "Fare", "SibSp"]:
        column_mean = X_train[column].mean()
        column_std = X_train[column].std()

        standardization_data[column] = {"mean": float(column_mean), "std":float(column_std)}

        X_train[column] = (X_train[column] - column_mean) / column_std
        X_test[column] = (X_test[column] - column_mean) / column_std

    feature_order = list(X_train.columns)

    # Convert data to tensors
    X_train = X_train.astype(float)
    y_train = y_train.astype(float)
    X_test = X_test.astype(float)
    y_test = y_test.astype(float)

    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)

    train_dataset =  TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train model

    model = TitanicModel()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 790

    train_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    for epoch in range(1,num_epochs + 1):
        model.train()

        running_loss = 0.0
        for batch_X,batch_y in train_loader:
            optimizer.zero_grad()

            output = model(batch_X)
            loss = criterion(output, batch_y)
            running_loss += loss.item() * len(batch_X)
            loss.backward()
            optimizer.step()
        train_loss_history.append(running_loss / len(train_dataset))

    # 注意：这里的 running_loss 记录的是每批次模型更新前的损失，所以记录的每轮的损失平均值,反应了不同参数下不同批次的损失的总体情况。

    # Evaluate model

        model.eval()
        with torch.no_grad():

            train_correct = 0

            for batch_X, batch_y in train_eval_loader:

                output = model(batch_X)

                probability = torch.sigmoid(output)
                predictions = (probability >= 0.5).float()
                train_correct += (predictions == batch_y).sum().item()

            test_correct = 0
            for batch_X, batch_y in test_loader:

                output = model(batch_X)

                probability = torch.sigmoid(output)
                predictions = (probability >= 0.5).float()
                test_correct += (predictions == batch_y).sum().item()

        train_accuracy = train_correct / len(train_dataset)
        train_accuracy_history.append(train_accuracy)

        test_accuracy = test_correct / len(test_dataset)
        test_accuracy_history.append(test_accuracy)

    print("Final train accuracy =",train_accuracy_history[-1] )
    print("Final test accuracy =", test_accuracy_history[-1])
    print("\n")

    # Save and reload model

    save_data = {
        "model_state_dict": model.state_dict(),
        "age_median": float(age_median),
        "embarked_mode": embarked_mode,
        "standardization_data": standardization_data,
        "feature_order": feature_order
    }
    torch.save(save_data, "src/titanic_model.pth")

    loaded_data = torch.load("src/titanic_model.pth")
    loaded_model = TitanicModel()
    loaded_model.load_state_dict(loaded_data["model_state_dict"])

    loaded_model.eval()
    model.eval()
    with torch.no_grad():
        output_01 = model(X_test_tensor)
        output_02 = loaded_model(X_test_tensor)

    if torch.allclose(output_01, output_02):
        print("Model parameters saved and loaded successfully")

    epochs = range(1, num_epochs + 1)

    # Plot training results

    plt.plot(epochs, train_loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("docs/Loss")
    plt.show()

    plt.plot(epochs, train_accuracy_history, label="Train Accuracy")
    plt.plot(epochs, test_accuracy_history, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train and Test Accuracy")
    plt.legend()
    plt.savefig("docs/Accuracy")
    plt.show()

    # Predict a new passenger

    (survival_probability, prediction_outcome) = predict_passenger(loaded_model, loaded_data)
    print("Survival probability =", survival_probability,
        "\nPrediction outcome =", prediction_outcome)

if __name__ == "__main__":
    main()
