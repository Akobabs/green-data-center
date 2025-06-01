# 🌿 Green Data Center: Electricity Consumption Optimization with ML

This project builds an intelligent system to **optimize electricity consumption** in green data centers using **machine learning** and **interactive visualizations**. It leverages the [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/energy+efficiency) to:

* 📉 Predict **Cooling Load** (regression)
* ⚡ Classify **Power Usage Effectiveness (PUE)** (classification)
* 📊 Visualize results via a **Streamlit dashboard**

The system demonstrates strong performance:

* **Regression**: MSE = 3.06 | R² = 0.97
* **Classification**: Precision = 1.00 | Recall = 0.98 | F1 = 0.99 | AUC = 1.00

---

## 🗂 Project Structure

```
green-data-center/
├── data/
│   ├── raw/                    ← Original dataset (ENB2012_data.xlsx)
│   └── processed/              ← Cleaned dataset
├── figures/                    ← Visualizations and dashboard screenshots
├── models/                     ← Saved ML models (.pkl)
├── notebooks/                 ← Jupyter notebook for EDA & preprocessing
├── src/                        ← Python scripts
│   ├── data_preprocessor.py
│   ├── model_trainer.py
│   └── test_preprocessor.py   ← Unit tests
├── dashboard.py                ← Streamlit app
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## ⚙️ Prerequisites

* Python 3.9+
* Docker (optional, for deployment)
* Git (optional, for version control)
* UCI Dataset: [`ENB2012_data.xlsx`](https://archive.ics.uci.edu/ml/machine-learning-databases/00242/) placed in `data/raw/`

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Akobabs/green-data-center.git
cd green-data-center
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure the following packages are present:

```
pandas, numpy, scikit-learn, seaborn, matplotlib,
streamlit, plotly, jupyter, joblib, pytest, openpyxl
```

---

## 📊 Usage Workflow

### 🔍 1. Exploratory Data Analysis

Launch the notebook:

```bash
jupyter notebook notebooks/eda_and_preprocessing.ipynb
```

Outputs:

* `data/processed/preprocessed_energy_data.csv`
* `figures/cooling_load_histogram.png`
* `figures/correlation_matrix.png`

---

### 🧹 2. Data Preprocessing

Run:

```bash
python src/data_preprocessor.py
```

Output:

* `data/processed/preprocessed_energy_data.csv`

---

### 🧠 3. Model Training

Train both regression and classification models:

```bash
python src/model_trainer.py
```

Outputs:

* `models/random_forest_regressor.pkl`
* `models/random_forest_classifier.pkl`
* `figures/regression_performance.jpg`
* `figures/roc_curve.png`

---

### 🖥 4. Launch Dashboard

```bash
streamlit run dashboard.py
```

Access at: [http://localhost:8501](http://localhost:8501)

Dashboard features:

* 📈 Visualizations (Cooling Load trend, PUE distribution)
* 🎚 Interactive sliders for feature input
* 🧾 Model performance metrics
* 📸 Screenshot: `figures/dashboard_screenshot.png`

---

### 🐳 5. Deploy with Docker (Optional)

Build and run the app containerized:

```bash
docker build -t green-data-center .
docker run -p 8501:8501 green-data-center
```

---

### ✅ 6. Run Tests

```bash
pytest src/test_preprocessor.py
```

---

## 📁 Key Outputs

| Type          | Files                                                                         |
| ------------- | ----------------------------------------------------------------------------- |
| **Data**      | `data/raw/ENB2012_data.xlsx`<br>`data/processed/preprocessed_energy_data.csv` |
| **Models**    | `models/random_forest_regressor.pkl`<br>`models/random_forest_classifier.pkl` |
| **Figures**   | `figures/cooling_load_histogram.png`, `regression_performance.jpg`, etc.      |
| **Dashboard** | `dashboard.py`, `figures/dashboard_screenshot.png`                            |

---

## 📊 Performance Summary

| Task           | Metric    | Score |
| -------------- | --------- | ----- |
| Regression     | MSE       | 3.06  |
|                | R²        | 0.97  |
| Classification | Precision | 1.00  |
|                | Recall    | 0.98  |
|                | F1 Score  | 0.99  |
|                | AUC       | 1.00  |

---

## 📝 Notes

* The PUE feature is synthetically engineered from existing variables.
* If Streamlit fails to load, check that `preprocessed_energy_data.csv` exists.
* Use figures and metrics in your report (Ch. 1–5, Appendices A & C).
* If Docker fails to start, check if port 8501 is already in use.

---

## 🙏 Acknowledgments

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/energy+efficiency) for the dataset
* [Streamlit](https://streamlit.io/) and [Plotly](https://plotly.com/) for interactive visualizations

---

**© 2025 Green Data Center Optimization Project**
*Developed by [Akobabs](https://github.com/Akobabs)*

---
