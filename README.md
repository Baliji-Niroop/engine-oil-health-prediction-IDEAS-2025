# 🚗 Engine Oil Health Prediction

### IDEAS Internship – Autumn 2025 | Indian Statistical Institute (ISI), Kolkata

A machine learning and deep learning–based system designed to **predict engine oil degradation and health status** using sensor-derived automotive parameters such as engine RPM, oil temperature, pressure, and degradation indicators.

This project was developed as part of the **IDEAS Internship Program – Autumn 2025** at the **Indian Statistical Institute (ISI), Kolkata**, under the mentorship of **Dr. Sandip Bhattacharyya**.

---

# 🎯 Problem Statement

Engine oil plays a critical role in maintaining engine efficiency, lubrication, and thermal stability. Conventional oil replacement strategies rely on **fixed time or mileage intervals**, which may lead to:

* 🔸 **Premature oil replacement** → unnecessary maintenance cost
* 🔸 **Delayed oil replacement** → increased engine wear and damage risk

This project introduces a **data-driven predictive maintenance approach** that classifies engine oil health using real-time sensor parameters, enabling **intelligent and optimized maintenance decisions**.

---

# 👥 Project Team

* **Baliji Niroop**
* **Sanjeev Manvith Vellala**
* **Kottu Shyam Sailesh**
* **Vamsi Raghav Tallapaka**

---

# 🧠 Methods & Models

## Machine Learning Models

The following models were trained and evaluated for engine oil health classification:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Gradient Boosting
* XGBoost
* Stacking Ensemble Model

## Deep Learning Model

* **Keras Sequential Neural Network**

---

# ⚙️ Data Preprocessing & Techniques

The pipeline includes several preprocessing and modeling techniques to improve performance and robustness:

* Missing value handling
* Feature scaling using **StandardScaler**
* Handling class imbalance using **SMOTE**
* **5-Fold Cross Validation** for model reliability
* **SHAP (SHapley Additive exPlanations)** for model interpretability and explainability

---

# 📊 Model Evaluation

Multiple evaluation metrics were used to ensure model performance and reliability:

* Accuracy
* F1 Score
* ROC-AUC Score
* Cross-Validation Accuracy
* Confusion Matrix
* ROC Curve

A **multi-model comparison framework** was implemented to ensure transparency and reproducibility of results.

---

# 📂 Repository Structure

```
Data/            → Raw and processed datasets
Notebook/        → Exploratory analysis, modeling, SHAP analysis
Models/          → Saved trained models
Results/         → Model evaluation plots and visualizations
Reports/         → Internship report and presentation slides

evaluation_report.py
model_comparison.py
```

---

# 🚘 Automotive Significance

From an **automobile engineering perspective**, this system can support:

* Predictive maintenance strategies
* Intelligent oil change scheduling
* Reduced engine wear and friction losses
* Improved fuel efficiency and engine longevity
* Potential integration with **ECU-based diagnostic systems**

### Future Extension

* Real-time **IoT-based oil health monitoring system**
* Integration with vehicle onboard diagnostics (OBD)
* Edge-based predictive maintenance systems for smart vehicles

---

# 🔗 Project Repository

GitHub Repository:

https://github.com/sanjeevmanvithvellala/engine-oil-health-prediction-IDEAS-2025

---

# 🔒 Intellectual Property & License

This work was developed during the **IDEAS Internship 2025 at the Indian Statistical Institute (ISI), Kolkata**, under the supervision of **Dr. Sandip Bhattacharyya**.

All materials in this repository are protected under **Intellectual Property Rights (IPR)**.

Unauthorized reuse or redistribution without proper citation or permission is prohibited.

© 2025 The Authors. Released under the **MIT License** (see `LICENSE` file).
