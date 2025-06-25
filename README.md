
# Emotional Classification Speech Processing

This project aims to build an end-to-end **audio emotion recognition system** that accurately classifies human emotions from speech signals. Leveraging machine learning techniques and audio signal processing (like MFCC feature extraction), the model identifies emotional states such as happiness, anger, sadness, etc., from input speech.

The system is trained on a labelled audio dataset and evaluated using performance metrics like confusion matrix, F1-score, and class-wise accuracy. A trained model is exported for real-time testing through a Python script and a deployed **Streamlit-based web application**, where users can upload audio files and receive instant emotion predictions.

---

##  Objective

To design and implement a robust machine learning pipeline that classifies **emotional states from audio data**, leveraging feature extraction techniques and trained model.
---

##  Dataset

The project uses audio files provided through two folders:
- `Audio_Speech_Actors_01-24`
- `Audio_Song_Actors_01-24`

Each `.wav` file is labelled with emotion codes ranging from `01` to `08`, mapped as:

| Code | Emotion     |
|------|-------------|
| 01   | Neutral     |
| 02   | Calm        |
| 03   | Happy       |
| 04   | Sad         |
| 05   | Angry       |
| 06   | Fearful     |
| 07   | Disgust     |
| 08   | Surprised   |

---

##  Pre-processing & Feature Extraction

I have extracted features from each `.wav` file using `librosa`:

- **MFCC** (Mel-Frequency Cepstral Coefficients)
Represents the short-term power spectrum of sound. MFCCs model how humans perceive audio and are widely used for speech and emotion recognition.
- **Chroma STFT**(Short-Time Fourier Transform)
Captures the energy distribution across 12 distinct pitch classes (like musical notes) — useful because emotional speech often varies in pitch and tone.
- **Spectral Contrast**
Measures the difference between peaks and valleys in the frequency spectrum. Helps distinguish between different timbres (e.g., harsh vs. smooth voices)
- **Spectral Centroid**
Represents the "centre of mass" of the spectrum. It indicates whether the sound is bright (higher centroid) or dark (lower centroid), which can vary with emotion.


---

##  Class Distribution Analysis

After combining both datasets:



## Model Pipeline <br>

1)Label Encoding

2)Train-Test Split (80-20 with stratification)

3)Feature Scaling using StandardScaler

4)SMOTE Oversampling

5)Model Training using:<br>

*RandomForestClassifier (GridSearchCV tuned)<br>
*XGBoost Classifier (RandomizedSearchCV tuned)<br>
*Multilayer Perceptron Classifier(MLP Classifier)<br>
The best Model with good metrics will be chosen for final training. 
6)Model Evaluation and confusion matrix after every model training

## Random Forest Classifier 
To classify the extracted emotion features, I have trained a Random Forest Classifier with hyperparameter tuning using GridSearchCV. The model was trained on a balanced dataset using SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.
### Metric Description
1) **Accuracy**- The ratio of correct predictions to the total number of predictions.<br>
2)**F1 score**-The harmonic mean of precision and recall.<br>
3)**Macro F1 Score**- Compute F1 score for each class individually and then take the average, treating all classes equally.<br>
4)**Weighted F1 Score**- Compute F1 score for each class and then take the average, but weigh by the number of true instances in each class. I have used this in my project as our model has class imbalance, it reflects overall performance better.<br><br>


**Metric	Scores**-
Accuracy	69.4% <br>
Weighted F1	70% <br>
Macro F1	68% <br>
**F1 Score Based Analysis (Emotion-Wise)**

Calm	0.84	🔹 Best-performing emotion. High precision & recall. Likely due to clear, consistent acoustic patterns.
Angry	0.76	🔹 Strong performance. Model identifies it confidently—probably due to its intense, sharp vocal tone.
Sad	0.67	🔸 Moderately high. Slight confusion with similar low-energy emotions like fearful.
Fearful	0.65	🔸 Mid-level score. Often confused with sad or angry, reflecting overlap in vocal tone features.
Happy	0.68	🔸 Moderate. Good precision but slightly low recall — model misses some happy samples.
Neutral	0.69	🔸 Performs well despite fewer samples. Likely due to its distinguishable flat tone.
Disgust	0.55	🔻 Lower score. Caused by small sample size and subtle vocal features. SMOTE helps but not enough.
Surprised	0.61	🔻 Low F1 but better than disgust. Still limited by data imbalance and overlapping features.
**Classification Report Insights:** <br>
Calm, Angry, and Sad emotions are classified well (F1 > 0.65).Disgust and Surprised emotions had lower support in data but still achieved ~0.55–0.61 F1-score.Neutral was relatively balanced, scoring ~0.69 F1
