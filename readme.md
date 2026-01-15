# Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview
This project focuses on predicting **customer churn** (whether a customer will stay with or leave a company) using an **Artificial Neural Network (ANN)**. The model is trained on a churn dataset and deployed using **Streamlit** to provide an interactive web-based prediction interface.

The application is divided into three major components:
1. **Model Training**
2. **Model Prediction**
3. **Deployment using Streamlit**

---

## 🧠 1. Model Training
- The churn dataset is preprocessed by removing irrelevant columns and encoding categorical features.
- Numerical features are scaled to improve model performance.
- An **Artificial Neural Network (ANN)** is built using a deep learning framework.
- The model is trained on historical customer data to learn patterns associated with churn behavior.
- After training, the model is saved for later use during prediction and deployment.

---

## 🔍 2. Model Prediction
- The trained ANN model is loaded from the saved file.
- New customer data is passed to the model after applying the same preprocessing steps used during training.
- The model outputs a probability score indicating the likelihood of churn.
- Based on a defined threshold, the customer is classified as:
  - **Customer will stay**
  - **Customer will leave (churn)**

---

## 🌐 3. Deployment using Streamlit
- **Streamlit** is used to build a user-friendly web application.
- Users can input customer details through the UI.
- The application processes the input data, feeds it to the ANN model, and displays the prediction instantly.
- This makes the model accessible to non-technical users and suitable for real-world use.

---

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Libraries & Frameworks:**
  - NumPy
  - Pandas
  - Scikit-learn
  - TensorFlow / Keras
  - Streamlit
  - Matplotlib / Seaborn (for analysis & visualization)
- **Model:** Artificial Neural Network (ANN)
- **Deployment:** Streamlit Web Application

---

## 📂 Project Structure
- `model_training.py` – Data preprocessing and ANN model training
- `model_prediction.py` – Loading the trained model and making predictions
- `app.py` – Streamlit application for deployment
- `saved_model/` – Trained ANN model files
- `dataset/` – Churn dataset
- `README.md` – Project documentation

---

## ✨ Key Features
- End-to-end machine learning pipeline
- Accurate churn prediction using deep learning
- Interactive and easy-to-use web interface
- Scalable and reusable model

---

## 👩‍💻 Author
**Prakriti Dheeraj**  
Aspiring Machine Learning Engineer  
📧 Email: prakritidheeraj620@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/prakriti-dheeraj-122464297  
🐙 GitHub: https://github.com/Prakriti-dheeraj

---

## 📄 License
This project is for educational and learning purposes.
