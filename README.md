# IF3070 Artificial Intelligence - Project 2  
**Machine Learning Algorithm Implementation**  

---

## Objective  
This project aims to provide hands-on experience in implementing machine learning algorithms to solve real-world problems.  

---

## Specifications  
You will implement the following machine learning models on the **PhiUSIIL Phishing URL Dataset**:  
1. **K-Nearest Neighbor (KNN)**  
   - Implemented from scratch  
   - Includes support for parameters: number of neighbors, and distance metrics (Euclidean, Manhattan, Minkowski)  
2. **Gaussian Naive-Bayes**  
   - Implemented from scratch  
3. **Comparison with scikit-learn**  
   - Evaluate and compare results from scratch implementations with scikit-learn.  
4. **Model Saving and Loading**  
   - Save and load models using any method (e.g., `.txt`, `.pkl`).  

---

## Dataset  
**PhiUSIIL Phishing URL Dataset**  
A dataset containing URL descriptions and features with labels for legitimate (1) and phishing (0) URLs.  
[Access the dataset here](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).  

---

## Setup  

### Prerequisites  
1. Install Python  
   ```bash  
   # in Linux
    sudo apt install python3

    # Or other OS
    https://www.python.org/

2. Install Jupyter Notebook
Run the following command to install Jupyter notebook:
   ```bash 
   pip install jupyter notebook

Other Python libraries (install using pip): sklearn, scipy, matplotlib, numpy, pandas, joblib, pickle

### Installation
1. Clone the repository:
   ```bash 
    git clone https://github.com/abelaprl/KNN-NaiveBayes-DasarAI.git  
2. Navigate to the project folder:
    ```bash 
    cd KNN-NaiveBayes-DasarAI  
3. Run the Notebook Program:
   ```bash 
    python3 src/Notebook.py  

## Task Distribution  

| NIM      | Name                          | Tasks                                                                 |
|----------|-------------------------------|-----------------------------------------------------------------------|
| 18222008 | Abel Apriliani                | Notebook initialization, EDA, Data Cleaning, Data Validation, Submission, Report |
| 18222036 | Olivia Christy Lismanto       | Data Preprocessing, Submission, Error Analysis, Report               |
| 18222044 | Khansa Adilla Reva            | KNN Implementation, Submission, Report                                |
| 18222062 | Nafisha Virgin                | Naive Bayes Implementation, Report                                    |
                
## References

1. [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
2. [Dataset Article](https://www.sciencedirect.com/science/article/abs/pii/S0167404823004558?via%3Dihub)
3. [KNN Documentation](https://scikit-learn.org/1.5/modules/neighbors.html)
4. [Naive-Bayes Documentation](https://scikit-learn.org/1.5/modules/naive_bayes.html)

