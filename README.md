# Mass Yield Sampling
Gaussian Algorithms for synthetic data generation of fission mass yields

### Στα requirements.txt βρίσκονται οι βιβλιοθήκες που χρειάζεται: 
```pip install -r requirements.txt```

## Για το Gaussian Process Regression

### Sampling

Χρησιμοποιήθηκε το **Gaussian Mixture** από το scikit-learn για να γίνουν extract 10,000 samples -> **test_data**

### Δομή του GPR:
1. Εισαγωγή data και χωρισμός σε input-output
2. Train_data = Expiremental Data (JENDL 5.0) (U-235, U-238, Th-232, Cm-246 (Induced))
3. Test_data = GaussianMixture samples
>Με το **linspace** της numpy κάνω το X -> 10.000 data αντί για 107
5. Δοκιμή των kernels:
   - **ContantKernel**
   - **RBF**
   - **ExpSineSquared**
6. Με **n_restarts_optimizer=300** για τα epochs
7. **fit** του model και **predict** των errors
8. Γραφική με το **matplotlib** με Prediction, Observation, Confidence Interval

## Για το Gaussian Mixture Regression

1. Εισαγωγή data και χωρισμός σε input-output (αλλαγή του element variable για συγκεκριμένο στοιχείο που θέλουμε να προβλέψουμε)
2. Δημιουργία Gaussian Mixture Model με for loop στην χρήση aic criterion για τον βέλτιστο αριθμό gaussians
3. Δημιουργία Gaussian Mixture Regression model με:
    - Αριθμό gaussians = αριθμό gaussians gmm
    - priors = gmm weights
    - means = gmm means
    - covariances = gmm covariances κανονικοποιημένα
4. Plot GMR και data extraction

## Script Usage

### Automation Script
```python pass_arg_GPR.py "CSV_File" "Plot Title"```

- Για το CSV_File μπορούν να επιλεχθούν από το Data Folder όπως **./Data/Th-232**
- Για το Plot Title μπορεί ως εξής : **Τh-232 (Induced)**



