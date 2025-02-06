# **Storm Gust Prediction with Machine Learning and WRF Model Variables**  

This repository contains selected scripts used in our research on **storm gust prediction** using **machine learning models** and **WRF model variables** for the **Northeast United States**. The study evaluates different machine learning algorithms, performs hyperparameter tuning, and analyzes model performance spatially and statistically.  

📄 **Publication:** This research is published in *Artificial Intelligence for the Earth Systems*  
[DOI: 10.1175/AIES-D-23-0047.1](https://doi.org/10.1175/AIES-D-23-0047.1)  

---

## **📂 Repository Structure**
The repository includes the following scripts:

### **1️⃣ Data**
- 📁 **Directory:** `Data/`  
- **Description:** Contains the datasets and station information used in this study.

### **2️⃣ Training and Inference**
- **Scripts:**
  - `RF_train_inference.py`  
  - `XGB_train_inference.py`  
  - `GLM_identity_train_inference.py`  
  - `GLM_log_train_inference.py`  
- **Description:** These scripts train and perform inference using:
  - **Random Forest (RF)**
  - **XGBoost (XGB)**
  - **Generalized Linear Model (GLM) with Identity link function**
  - **Generalized Linear Model (GLM) with Log link function**  
- **Related to:** *Figure 3 in the paper.*

### **3️⃣ Hyperparameter Tuning**
- **Script:** `Hyperopt_XGB.py`  
- **Description:** Performs hyperparameter tuning using the **Hyperopt** Python package. The script is designed for **XGBoost**, but it can be modified for **Random Forest**.

### **4️⃣ Bootstrapped Confidence Intervals**
- **Script:** `Bootstrapped_CI.py`  
- **Description:** Computes **bootstrapped 95% confidence intervals** for model performance.  
- **Related to:** *Figure 4 in the paper.*

### **5️⃣ Feature Importance Analysis**
- **Script:** `Feature_importance_plot.py`  
- **Description:** Plots **permutation feature importance** for **Random Forest** and **XGBoost** models.  
- **Related to:** *Figure 5 in the paper.*

### **6️⃣ Spatial Error Analysis**
- **Scripts:**
  - `Spatial_error_calculate.py`
  - `Spatial_error_plot.py`
  - `Spatial_error_plot_percent_decrease.py`  
- **Description:**  
  - Computes **prediction errors** (e.g., **MAE, RMSE, CRMSE**) at each station.  
  - Analyzes **percentage decrease in error** due to ML models compared to the physics-based **WRF model**.  
-  **Related to:** *Figures 6 and 7 in the paper.*

### **7️⃣ Training Dataset Size and Learning Curve Analysis**
- **Scripts:**
  - `Error_vs_training_size.py`
  - `Learning_curve_with_CI.py`  
- **Description:**  
  - Analyzes how **prediction error changes** with increasing training dataset size.  
  - Plots **learning curves** with **bootstrapped 95% confidence intervals**.  
- **Related to:** *Figure 8 in the paper.*


### **Usage**
These scripts can be adapted to work with other datasets by modifying the data preprocessing steps. Ensure that your dataset follows a similar structure to the one used in this research [DOI: 10.1175/AIES-D-23-0047.1](https://doi.org/10.1175/AIES-D-23-0047.1) 


### **Citation**
If you find this repository useful, please cite our paper [DOI: 10.1175/AIES-D-23-0047.1](https://doi.org/10.1175/AIES-D-23-0047.1)

### **Contact**
For questions or collaborations, feel free to reach out via GitHub Issues or email:israt.jahan@uconn.edu

