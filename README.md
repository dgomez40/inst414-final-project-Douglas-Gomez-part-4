# Weather & Air Pollution Predictive Analytics

This project analyzes the relationship between weather conditions and air pollution (PM2.5) levels in Montgomery County, MD, and includes predictive analytics using machine learning.

---

## **Project Structure**
- `extract/` – Scripts to extract raw data  
- `transform/` – Scripts to clean and merge data (`transform.py`)  
- `load/` – Scripts to save processed data (`load.py`)  
- `analysis/` – Predictive analytics & model scripts  
  - `model.py` – Trains machine learning models  
  - `evaluate.py` – Evaluates model performance  
- `main.py` – Runs the entire workflow (ETL + modeling + evaluation)  

---

## **Requirements**
Install all dependencies:
pip install -r requirements.txt