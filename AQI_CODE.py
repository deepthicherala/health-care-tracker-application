import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Label, Entry, Button
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

main = tk.Tk()
main.title("Indian Air Quality and Health Impact")
main.geometry('1200x800')
main.config(bg='skyblue3')

font = ('times', 15, 'bold')
tk.Label(main, text="Indian Air Quality and Health Impact Output", font=('times', 20, 'bold'), bg='skyblue3').place(x=330, y=60)
text = tk.Text(main, height=25, width=120)
text.place(x=330, y=100)
text.config(font=('times', 12, 'bold'))

# Global variables
df_train = None
df_test = None
model = None
label_encoder = None

# Upload train dataset
def upload_train():
    global df_train
    file_path = filedialog.askopenfilename()
    df_train = pd.read_csv(file_path)
    text.delete('1.0', tk.END)
    text.insert(tk.END, "✅ Train dataset loaded successfully!\n\n")
    text.insert(tk.END, df_train.head())

# Upload test dataset
def upload_test():
    global df_test
    file_path = filedialog.askopenfilename()
    df_test = pd.read_csv(file_path)
    text.delete('1.0', tk.END)
    text.insert(tk.END, "✅ Test dataset loaded successfully!\n\n")
    text.insert(tk.END, df_test.head())

# Preprocessing
def preprocess():
    global df_train, df_test, label_encoder

    if df_train is None or df_test is None:
        messagebox.showerror("Error", "Please upload both train and test datasets.")
        return

    text.delete('1.0', tk.END)

    def bin_aqi(val):
        if val <= 50:
            return 'Good'
        elif val <= 100:
            return 'Moderate'
        elif val <= 150:
            return 'Unhealthy for Sensitive'
        elif val <= 200:
            return 'Unhealthy'
        elif val <= 300:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'

    # Binning
    df_train['AQI_Category'] = df_train['AQI'].apply(bin_aqi)
    df_test['AQI_Category'] = df_test['AQI'].apply(bin_aqi)

    # Encode AQI Category
    label_encoder = LabelEncoder()
    df_train['AQI_Label'] = label_encoder.fit_transform(df_train['AQI_Category'])
    df_test['AQI_Label'] = label_encoder.transform(df_test['AQI_Category'])

    text.insert(tk.END, "✅ Preprocessing completed successfully!\n\n")
    text.insert(tk.END, df_train[['AQI', 'AQI_Category', 'AQI_Label']].head())

# EDA Plots
def eda():
    global df_train
    if df_train is None:
        messagebox.showerror("Error", "Please upload and preprocess the train dataset first.")
        return

    plt.figure(figsize=(8, 6))
    sns.countplot(x='AQI_Category', data=df_train)
    plt.title('AQI Category Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(df_train['PM2.5'], kde=True)
    plt.title('PM2.5 Distribution')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='AQI_Category', y='Temperature (°C)', data=df_train)
    plt.title('Temperature by AQI Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Train RFC model
def train_rfc():
    global model, df_train
    if df_train is None:
        messagebox.showerror("Error", "Please upload and preprocess the train dataset first.")
        return

    X = df_train.drop(columns=['AQI', 'AQI_Category', 'AQI_Label', 'Health Impact Score'])
    y = df_train['AQI_Label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/rfc_aqi_model.pkl')
    text.insert(tk.END, "\n✅ Random Forest Classifier trained and saved successfully!\n")

# Train SVC model
def train_svc():
    global model, df_train
    if df_train is None:
        messagebox.showerror("Error", "Please upload and preprocess the train dataset first.")
        return

    X = df_train.drop(columns=['AQI', 'AQI_Category', 'AQI_Label', 'Health Impact Score'])
    y = df_train['AQI_Label']
    model = SVC(kernel='rbf')
    model.fit(X, y)

    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/svc_aqi_model.pkl')
    text.insert(tk.END, "\n✅ Support Vector Classifier trained and saved successfully!\n")

# Predict with individual entries
def predict():
    global model, label_encoder
    if model is None:
        messagebox.showerror("Error", "Please train the model first.")
        return

    text.delete('1.0', tk.END)

    feature_names = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3',
                     'Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)',
                     'Rainfall (mm)', 'Pressure (hPa)', 'Vehicle Count', 'Industrial Activity Index']

    popup = Toplevel(main)
    popup.title("Enter Feature Values")
    entries = {}

    for i, feature in enumerate(feature_names):
        Label(popup, text=feature).grid(row=i, column=0, padx=10, pady=5, sticky='w')
        entry = Entry(popup, width=30)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entries[feature] = entry

    def submit():
        try:
            input_data = [float(entries[feat].get()) for feat in feature_names]
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numerical values.")
            return

        popup.destroy()
        input_df = pd.DataFrame([input_data], columns=feature_names)
        prediction = model.predict(input_df)[0]
        category = label_encoder.inverse_transform([prediction])[0]

        text.insert(tk.END, "\n🔍 Input Features:\n")
        for name, val in zip(feature_names, input_data):
            text.insert(tk.END, f"{name}: {val}\n")

        text.insert(tk.END, f"\n✅ Predicted AQI Category: {category}\n")

    Button(popup, text="Submit", command=submit, bg='pale green').grid(row=len(feature_names), columnspan=2, pady=10)

# Buttons
tk.Button(main, text="Upload Train Dataset", command=upload_train, bg='pale green', width=20, font=font).place(x=50, y=100)
tk.Button(main, text="Upload Test Dataset", command=upload_test, bg='pale green', width=20, font=font).place(x=50, y=150)
tk.Button(main, text="Preprocess", command=preprocess, bg='pale green', width=20, font=font).place(x=50, y=200)
tk.Button(main, text="Show EDA", command=eda, bg='pale green', width=20, font=font).place(x=50, y=250)
tk.Button(main, text="Train RFC", command=train_rfc, bg='pale green', width=20, font=font).place(x=50, y=300)
tk.Button(main, text="Train SVC", command=train_svc, bg='pale green', width=20, font=font).place(x=50, y=350)
tk.Button(main, text="Predict", command=predict, bg='pale green', width=20, font=font).place(x=50, y=400)

main.mainloop()
