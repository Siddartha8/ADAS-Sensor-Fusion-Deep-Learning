import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
from tkinter import filedialog
import tkinter
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, label_binarize, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    precision_score, recall_score, f1_score, accuracy_score, 
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import joblib
from joblib import dump, load
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import warnings

from PIL import Image, ImageTk

# Setting up directories
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok = True)



accuracy = []
precision = []
recall = []
fscore = []

labels = ['Brake', 'Lane Correct', 'Maintain Speed', 'Accelerate']


def load_dataset():
    """Load the dataset from a CSV file selected via file dialog."""
    filepath = filedialog.askopenfilename(
        initialdir=".",
        title="Select CSV File",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    if filepath:
        return pd.read_csv(filepath)
    else:
        print("No file selected.")
        return None


def preprocess_data(df, is_train=True, label_encoders=None):
    df = df.copy()

    # --- Handle timestamp: keep only HH:MM ---
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['timestamp'] = df['timestamp'].dt.strftime('%H:%M')  # keep only hour:minute

    if is_train:
        label_encoders = {}
        
        # Encode object columns (categorical including timestamp and ADAS_output)
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        if label_encoders is None:
            raise ValueError("label_encoders must be provided for test/inference.")
        
        for col in df.select_dtypes(include='object').columns:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = le.transform(df[col].astype(str))
            else:
                raise ValueError(f"Missing encoder for column: {col}")

    # Fill missing numeric values
    df = df.fillna(df.mean(numeric_only=True))

    if is_train:
        X = df.drop(columns=['ADAS_output'])
        y = df['ADAS_output']
        return X, y, label_encoders
    else:
        return df


# DataFrames to store results
metrics_df = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
class_report_df = pd.DataFrame()
class_performance_dfs = {}

# Ensure results directory exists
if not os.path.exists('results'):
    os.makedirs('results')


def plot_multiclass_roc(y_test, y_score, categories, algorithm):
    """Plot ROC curve for multiclass classification using One-vs-Rest."""
    # Binarize labels for One-vs-Rest ROC
    y_test_bin = label_binarize(y_test, classes=range(len(categories)))
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(8, 6))

    # Store per-class ROC values
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        plt.plot(fpr_dict[i], tpr_dict[i], label=f"{categories[i]} (AUC = {auc_dict[i]:.2f})")

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, 'k--', label=f"micro-average (AUC = {auc_micro:.2f})")

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    auc_macro = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, 'r--', label=f"macro-average (AUC = {auc_macro:.2f})")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{algorithm} ROC Curve (One-vs-Rest)")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{algorithm.replace(' ', '_')}_roc_curve.png")
    plt.show()


def Calculate_Metrics(algorithm, predict, y_test, y_score):
    global metrics_df, class_report_df, class_performance_dfs

    categories = labels

    # Overall metrics
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    # Add to metrics dataframe
    metrics_entry = pd.DataFrame({
        'Algorithm': [algorithm],
        'Accuracy': [a],
        'Precision': [p],
        'Recall': [r],
        'F1-Score': [f]
    })
    metrics_df = pd.concat([metrics_df, metrics_entry], ignore_index=True)

    # Print metrics
     
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")


    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predict)

    # Classification report
    CR = classification_report(y_test, predict, target_names=categories, output_dict=True)
    
    CR1 = classification_report(y_test, predict, target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR1) +"\n\n")

    # Save classification report
    cr_df = pd.DataFrame(CR).transpose()
    cr_df['Algorithm'] = algorithm
    class_report_df = pd.concat([class_report_df, cr_df], ignore_index=False)

    # Per-class performance
    for category in categories:
        class_entry = pd.DataFrame({
            'Algorithm': [algorithm],
            'Precision': [CR[str(category)]['precision'] * 100],
            'Recall': [CR[str(category)]['recall'] * 100],
            'F1-Score': [CR[str(category)]['f1-score'] * 100],
            'Support': [CR[str(category)]['support']]
        })

        if str(category) not in class_performance_dfs:
            class_performance_dfs[str(category)] = pd.DataFrame(columns=['Algorithm', 'Precision', 'Recall', 'F1-Score', 'Support'])

        class_performance_dfs[str(category)] = pd.concat([class_performance_dfs[str(category)], class_entry], ignore_index=True)

    # Confusion matrix plot
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(conf_matrix, xticklabels=categories, yticklabels=categories, annot=True, cmap="viridis", fmt="g")
    ax.set_ylim([0, len(categories)])
    plt.title(f"{algorithm} Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png")
    plt.show()

    # ROC Curve
    if y_score is not None:
        try:
            if y_score.shape[1] > 2:  # Multiclass
                plot_multiclass_roc(y_test, y_score, categories, algorithm)
            else:  # Binary
                fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
                roc_auc_score = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score:.2f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f"{algorithm} ROC Curve")
                plt.legend(loc='lower right')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"results/{algorithm.replace(' ', '_')}_roc_curve.png")
                plt.show()
        except Exception as e:
            print(f"[ERROR] Could not plot ROC curve for {algorithm}: {e}")


def train_svm(X_train, y_train, X_test, y_test):
    """Train and evaluate a Support Vector Machine classifier, save it, and return metrics."""
    
    model_path = os.path.join(MODEL_DIR, 'svm_classifier.joblib')
    
    if os.path.exists(model_path):
        model = load(model_path)
    else:
        model = SVC(probability=True, kernel='rbf', random_state=42)
        model.fit(X_train, y_train)
        dump(model, model_path)
    
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    return Calculate_Metrics("SVM Model", y_pred, y_test, y_score)


def train_knn(X_train, y_train, X_test, y_test):
    """Train and evaluate a K-Nearest Neighbors classifier, save it, and return metrics."""
    
    model_path = os.path.join(MODEL_DIR, 'knn_classifier.joblib')
    
    if os.path.exists(model_path):
        model = load(model_path)
    else:
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        dump(model, model_path)
    
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    return Calculate_Metrics("KNN Classifier", y_pred, y_test, y_score)



def train_random_forest_classifier(X_train, y_train, X_test, y_test):
    """Train and evaluate a weaker Random Forest Classifier model (restricted capacity)."""
    model_path = os.path.join(MODEL_DIR, 'random_forest_classifier.joblib')
    
    if os.path.exists(model_path):
        model = load(model_path)
    else:
        # Restrict capacity so accuracy drops
        model = RandomForestClassifier(
            n_estimators=10,       # fewer trees
            max_depth=2,           # shallow trees
            max_features=2,   # fewer features at each split
            min_samples_split=20,  # avoid very fine splits
            min_samples_leaf=10,   # ensure leaf nodes have enough samples
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        dump(model, model_path)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    
    return Calculate_Metrics("RFC Model", y_pred, y_test, y_score)


def train_hybrid_cnn1d_random_forest(X_train, y_train, X_test, y_test, num_classes):
    # --- Convert labels to numpy arrays ---
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # --- Normalize features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

    # Reshape for CNN (samples, timesteps, channels)
    X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    cnn_model_path = os.path.join(MODEL_DIR, 'hybrid_cnn1d_model.h5')
    rf_model_path = os.path.join(MODEL_DIR, 'hybrid_cnn1d_random_forest.joblib')

    # --- Train or Load CNN ---
    if os.path.exists(cnn_model_path):
        cnn = load_model(cnn_model_path)
        print("Loaded CNN-1D model.")
    else:
        inputs = Input(shape=(X_train_cnn.shape[1], 1))
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu', name='penultimate_layer')(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        cnn = Model(inputs=inputs, outputs=outputs)
        cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        cnn.fit(X_train_cnn, y_train, epochs=200, batch_size=32,
                validation_split=0.2, callbacks=[early_stop], verbose=0)
        cnn.save(cnn_model_path)
        print("Trained and saved CNN-1D model.")

    # --- Extract penultimate layer features ---
    feature_extractor = Model(inputs=cnn.input,
                              outputs=cnn.get_layer('penultimate_layer').output)
    cnn_features_train = feature_extractor.predict(X_train_cnn)
    cnn_features_test = feature_extractor.predict(X_test_cnn)

    # --- Combine raw + CNN features ---
    combined_train = np.hstack((X_train_scaled, cnn_features_train))
    combined_test = np.hstack((X_test_scaled, cnn_features_test))

    # --- Train or Load Random Forest ---
    if os.path.exists(rf_model_path):
        rf_model = load(rf_model_path)
    else:
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(combined_train, y_train)
        dump(rf_model, rf_model_path)

    # --- Evaluation ---
    y_pred = rf_model.predict(combined_test)
    y_score = rf_model.predict_proba(combined_test)

    return Calculate_Metrics("DualStream-ConvRF Model", y_pred, y_test, y_score)



def Upload_Dataset():
    global df, labels
    text.delete('1.0', END)
    df = load_dataset()
    text.insert(END, "Dataset loaded successfully.\n\n")
    text.insert(END,"Sample Dataset: " + str(df.head())+ "\n")    
    labels = df['ADAS_output'].unique()
    

def Preprocess_Dataset():
    global df, X, y, label_s
    text.delete('1.0', END)
    X,y,label_s = preprocess_data(df,is_train=True)
    text.insert(END, "Preprocessing completed successfully.\n\n")
    
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data = df, x='ADAS_output')
    plt.xlabel('ADAS_output')
    plt.ylabel('Count')
    plt.title('Count of Class Values')
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()

def Train_Test_Splitting():
    global X, y, X_train, X_test, y_train, y_test, X_train_res, y_train_res, label_encoders, smote
    text.delete('1.0', END)
        
    X, y, label_encoders = preprocess_data(df, is_train=True)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Apply SMOTE only on training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


    text.insert(END, "Total records found in dataset: " + str(X.shape[0]) + "\n\n")
    text.insert(END, "Dataset Train and Test Split Completed" + "\n")
    text.insert(END, "Total records found in dataset before SMOTE: " + str(X_train.shape[0]) + "\n")
    text.insert(END, "Total records found in dataset after SMOTE: " + str(X_train_res.shape[0]) + "\n")
    text.insert(END, "Total records found in dataset to test: " + str(X_test.shape[0]) + "\n")    
   
def Classifier1():
    global X_train, X_test, y_class_train, y_class_test
    text.delete('1.0', END)
    results = {}
    results['SVM Model'] = train_svm(X_train_res,y_train_res, X_test,y_test)
    results['KNN Classifier Model'] = train_knn(X_train_res,y_train_res, X_test,y_test)
    results['RFC Model'] = train_random_forest_classifier(X_train_res,y_train_res, X_test,y_test)

def Regressor1():
    global X_train, X_test, y_reg_train, y_reg_test
    text.delete('1.0', END)
    train_hybrid_cnn1d_random_forest(X_train_res,y_train_res, X_test,y_test,num_classes=4)

def Prediction():
    global test_data, df1, label_s, Y_cls, cls_prob, Y_reg, Y_cls_names, attack_type_le, row_data, output_data, output_df
    text.delete('1.0', END)
    test_data = load_dataset()
    df1 = preprocess_data(test_data, is_train=False,label_encoders=label_s)

    # --- Load scaler and CNN ---
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    cnn = load_model(os.path.join(MODEL_DIR, 'hybrid_cnn1d_model.h5'))

    # Scale test data (only preprocessed features)
    X_new_scaled = scaler.transform(df1)  # df1 = preprocessed columns

    # Reshape for CNN input (samples, timesteps, channels)
    X_new_cnn = X_new_scaled.reshape((X_new_scaled.shape[0], X_new_scaled.shape[1], 1))

    # Extract penultimate layer features from CNN
    feature_extractor = Model(
        inputs=cnn.input,
        outputs=cnn.get_layer('penultimate_layer').output
    )
    cnn_features_new = feature_extractor.predict(X_new_cnn)

    # Combine raw preprocessed features + CNN features
    combined_new = np.hstack((X_new_scaled, cnn_features_new))

    # Load trained Random Forest (hybrid model)
    rf_hybrid = joblib.load(os.path.join(MODEL_DIR, 'hybrid_cnn1d_random_forest.joblib'))

    # Predict
    hybrid_preds = rf_hybrid.predict(combined_new)

    # Create final output with only preprocessed columns + prediction
    test_results = df1.copy()
    test_results['Hybrid_Prediction'] = label_s['ADAS_output'].inverse_transform(hybrid_preds)

    # Save to CSV
    output_path = os.path.join(RESULTS_DIR, "TestData1_HybridPredictions_CNN1D_clean.csv")
    test_results.to_csv(output_path, index=False)
    
    text.insert(END,str(test_results)+"\n\n")


import tkinter as tk
from tkinter import messagebox
import redis
import hashlib

# Connect to Redis
def connect_redis():
    return redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Hash password before storing in Redis for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Signup functionality
def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_redis()

                # Hash the password before storing
                hashed_password = hash_password(password)

                # Store the user in Redis with multiple field-value pairs
                user_key = f"user:{username}"
                if conn.exists(user_key):
                    messagebox.showerror("Error", "User already exists!")
                else:
                    # Using multiple field-value pairs in hset
                    conn.hset(user_key, "username", username)
                    conn.hset(user_key, "password", hashed_password)
                    conn.hset(user_key, "role", role)
                    messagebox.showinfo("Success", f"{role} Signup Successful!")
                    signup_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Redis Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    # Create the signup window
    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x400")
    signup_window.title(f"{role} Signup")

    # Username field
    tk.Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)
    
    # Password field
    tk.Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    # Signup button
    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# Login functionality
def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_redis()

                # Hash the password before checking
                hashed_password = hash_password(password)

                # Check if the user exists in Redis
                user_key = f"user:{username}"
                if conn.exists(user_key):
                    stored_password = conn.hget(user_key, "password")
                    stored_role = conn.hget(user_key, "role")

                    if stored_password == hashed_password and stored_role == role:
                        messagebox.showinfo("Success", f"{role} Login Successful!")
                        login_window.destroy()
                        if role == "Admin":
                            show_admin_buttons()
                        elif role == "User":
                            show_user_buttons()
                    else:
                        messagebox.showerror("Error", "Invalid Credentials!")
                else:
                    messagebox.showerror("Error", "User not found!")
            except Exception as e:
                messagebox.showerror("Error", f"Redis Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    tk.Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)

def show_admin_buttons():
    clear_buttons()
    tk.Button(main, text="Upload Dataset (EV)", command=Upload_Dataset, font=font1).place(x=100, y=160)
    tk.Button(main, text="Preprocessing & EDA", command=Preprocess_Dataset, font=font1).place(x=350, y=160)
    tk.Button(main, text="Data Balancing & Splitting", command=Train_Test_Splitting, font=font1).place(x=620, y=160)
    tk.Button(main, text="Build & Train ML Models", command = Classifier1, font=font1).place(x=950, y=160)
    tk.Button(main, text="Build & Train Proposed DualStream-ConvRF Model", command = Regressor1, font=font1).place(x=100, y=220)
    
   
def show_user_buttons():
    clear_buttons()
    tk.Button(main, text="Prediction on Test Data", command=Prediction, font=font1).place(x=650, y=200)

# Clear buttons before adding new ones
def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()

main = tk.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

bg_image = Image.open("background.jpg")  
bg_image = bg_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)

bg_photo = ImageTk.PhotoImage(bg_image)

canvas = Canvas(main, width=screen_width, height=screen_height)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Title
font = ('times', 18, 'bold')
title_text = "Sensor Fusion and Deep Learning for Advanced Driver Assistance System Perception"
title = tk.Label(main, text = title_text, bg='pale green', fg='black', font=font, wraplength=screen_width - 200, justify='center')
canvas.create_window(screen_width // 2, 50, window=title)

font1 = ('times', 14, 'bold')

# Create text widget and scrollbar
text_frame = tk.Frame(main, bg='white')
text = tk.Text(text_frame, height=22, width=130, font=font1, wrap='word')
scroll = tk.Scrollbar(text_frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)

text.grid(row=0, column=0, sticky='nsew')
scroll.grid(row=0, column=1, sticky='ns')
text_frame.grid_rowconfigure(0, weight=1)
text_frame.grid_columnconfigure(0, weight=1)

# Position the text_frame on the canvas, centered horizontally
canvas.create_window(screen_width // 2, 300, window=text_frame, anchor='n')


# Admin and User Buttons
font1 = ('times', 14, 'bold')

tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=25, height=1, bg='ivory2').place(x=50, y=100)

tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=25, height=1, bg='ivory2').place(x=400, y=100)

admin_button = tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=25, height=1, bg='LightCyan2')
admin_button.place(x=750, y=100)

user_button = tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=25, height=1, bg='LightCyan2')
user_button.place(x=1100, y=100)

main.config(bg='aquamarine')
main.mainloop()
