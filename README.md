# Mass Yield Sampling
Gaussian Algorithms for synthetic data generation of fission mass yields

### In requirements.txt are the libraries that are needed: 
```pip install -r requirements.txt```

## For Gaussian Process Regression

### Sampling

**Gaussian Mixture** was used from scikit-learn library in order to extract 10,000 samples -> **test_data**

### GPR Structure:
1. Insert data και divide input-output
2. Train_data = Expiremental Data (JENDL 5.0) (U-235, U-238, Th-232, Cm-246 (Induced))
3. Test_data = GaussianMixture samples
>With **linspace** of numpy the X becomes 10.000 data instead 107
5. Kernels trials:
   - **ContantKernel**
   - **RBF**
   - **ExpSineSquared**
6. With **n_restarts_optimizer=300** for epochs
7. **fit** the model and **prediction** of errors
8. Plots with **matplotlib** for Prediction, Observation, Confidence Interval

## For Gaussian Mixture Regression

1. Data insertion, division for inputs-outputs (change of element variable for specific element we want to predict)
2. Creation Gaussian Mixture Model with for loop for aic criterion in order to get the best number of gaussians
3. Creation Gaussian Mixture Regression model with:
    - Number of gaussians = number of gaussians gmm
    - priors = gmm weights
    - means = gmm means
    - covariances = gmm covariances κανονικοποιημένα
4. Plot GMR and data extraction

## Script Usage

### Automation Script
```python pass_arg_GPR.py "CSV_File" "Plot Title"```

- The elements can be picked from the CSV_File that is located Data Folder like this: **./Data/Th-232**
- For the Plot Title it can be written : **Τh-232 (Induced)**

# Citing
The main publication (and documentation) to cite is:

- Vasilis Tsioulos and Vaia Prassa, "Machine learning analysis of fission product yields", EPJ Web Conf., 304 (2024) 01015, DOI: [10.1051/epjconf/202430401015](https://doi.org/10.1051/epjconf/202430401015)

