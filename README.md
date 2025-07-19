# 🔥 Classification of Fire Types in India Using MODIS Satellite Data

This project focuses on using **Machine Learning** techniques to classify different fire types in India using data obtained from the **MODIS satellite**. Forest fires, agricultural fires, and other fire events pose a major threat to ecosystems, air quality, and human life. With the right classification model, we can detect and monitor these fires efficiently using satellite data.

---

## 📌 Project Objectives

- 📊 Analyze and preprocess MODIS satellite fire data.
- 🧹 Handle missing values, feature engineering, and data encoding.
- ⚙️ Build machine learning models (e.g., Random Forest, Decision Tree, SVM) to classify fire types.
- 📈 Evaluate models using accuracy, classification report, and confusion matrix.
- 🌍 Visualize the geographic spread and frequency of fire types in India.

---

## 🗂️ Dataset

- **Source**: MODIS Active Fire Detections (NASA FIRMS)
- **Format**: CSV
- **Features**:
  - Latitude, Longitude
  - Brightness
  - Scan & Track
  - Confidence
  - FRP (Fire Radiative Power)
  - Type of Fire (e.g., Forest, Agricultural)

> The dataset must be placed in the same directory as the script or modified in the code path.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas, NumPy** – Data Manipulation
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn** – Machine Learning

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fire-classification-india.git
   cd fire-classification-india
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python refactored_fire_classification.py
   ```

---

## 📊 Results

- Achieved over **X% accuracy** using [Best ML Model].
- Visualized spatial distribution of fire types.
- Identified key contributing features to fire classification.

---

## 🧠 Future Enhancements

- Integrate **deep learning models (e.g., CNNs)** for image-based fire detection.
- Create a **web dashboard** for real-time monitoring.
- Automate daily fire detection with satellite API integration.

---

## 📌 License

This project is licensed under the MIT License.

---

## 🤝 Acknowledgements

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) for satellite data
- Scikit-learn and open-source Python community
