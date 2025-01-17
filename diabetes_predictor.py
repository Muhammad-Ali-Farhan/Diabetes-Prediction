
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("diabetes.csv")


X = data.drop("Outcome", axis=1)  
y = data["Outcome"]  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(random_state=42)  
model.fit(X_train_scaled, y_train)  


def get_user_input():
    print("\nEnter your medical data below:")
    gender = input("Are you male or female? (M/F): ").strip().upper()
    
    if gender not in ["M", "F"]:
        print("Invalid input. Assuming 'M' (male).")
        gender = "M"
    
    pregnancies = 0
    if gender == "F":
        pregnancies = float(input("Pregnancies: "))
    
    glucose = float(input("Glucose level: "))
    blood_pressure = float(input("Blood Pressure: "))
    
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin Level: "))
    bmi = float(input("BMI: "))
    diabetes_pedigree = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))
    
    
    return np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])


def predict_diabetes(input_data):
    
    scaled_data = scaler.transform(input_data)
    
    
    probabilities = model.predict_proba(scaled_data)
    diabetic_chance = probabilities[0][1] * 100  
    non_diabetic_chance = probabilities[0][0] * 100  
    
    
    print(f"\nPrediction Results:")
    print(f"Chance of being diabetic: {diabetic_chance:.2f}%")
    
    
    if diabetic_chance >= 80:
        print("⚠️ High likelihood of diabetes! Please consult a doctor.")
    elif diabetic_chance >= 50:
        print("⚠️ Moderate chance of diabetes. Consider a medical checkup.")
    else:
        print("✔️ Low likelihood of diabetes. Stay healthy!")


if __name__ == "__main__":
    print("Diabetes Prediction Program")
    user_data = get_user_input()  
    predict_diabetes(user_data)   
