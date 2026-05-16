# SMS Spam Detection
**Group 4** — CS 450

A spam detection system that classifies SMS messages as spam or ham using Naive Bayes and Logistic Regression with TF-IDF feature extraction.

---

## Requirements

Python 3.7 or higher is required. Install all dependencies with:

```bash
python3 -m pip install pandas scikit-learn matplotlib seaborn nltk
```

---

## Dataset

**SMS Spam Collection** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) and place the file named `SMSSpamCollection` in the same folder as `spam_detection.py`.

The file must be named exactly `SMSSpamCollection`.

---

## How to Run

```bash
python3 spam_detection.py
```

The script runs automatically top to bottom and will:

1. Load and explore the dataset
2. Clean and preprocess the text
3. Split data into training and test sets (80/20)
4. Extract TF-IDF features
5. Train Naive Bayes and Logistic Regression models
6. Print evaluation reports for all model configurations
7. Run ablation studies on n-gram range, decision threshold, and TF-IDF filtering
8. Print a model comparison table
9. Save two chart images to the current folder:
   - `confusion_matrices.png`
   - `model_comparison.png`
10. Launch an interactive spam detector in the terminal

---

## Interactive Spam Detector

After the evaluation output, the script enters an interactive loop where you can type any message and get an instant spam/ham prediction:

```
─── Spam Detector (type 'quit' to exit) ───

Enter a message: Congratulations! You have won a free prize, click here now
  Message  : Congratulations! You have won a free prize, click here now
  Result   : SPAM
  Spam prob: 99.87%

Enter a message: Hey, are we still on for dinner tonight?
  Message  : Hey, are we still on for dinner tonight?
  Result   : HAM
  Spam prob: 0.31%

Enter a message: quit
Goodbye!
```

Type `quit` to exit.

---

## Configuration

The decision threshold can be adjusted at the top of section 6 in the code:

```python
CUSTOM_THRESHOLD = 0.4
```

---

## Project Structure

```
spam_detection.py       # Main script
SMSSpamCollection       # Dataset file (download separately)
confusion_matrices.png  # Generated after running
model_comparison.png    # Generated after running
README.md               # This file
```
