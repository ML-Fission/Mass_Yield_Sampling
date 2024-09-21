import os
import glob
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

components_range = range(1, 11)

# Initialize variables to store AIC values and GMM models
aic_values = []
gmm_models = []
bic_values = []

dir = ".\Data\Mass_Yields"
os.chdir(dir)
datasets = []
file_name = []
for csv_file in glob.glob('*.csv'):
    print(f'Processing file: {csv_file}')
    df = pd.read_csv(csv_file)
    datasets.append(df)
    file_name.append(csv_file.replace('.csv', ''))

sns.set(style="darkgrid")

# Iterate over different numbers of components
for data in datasets:
    aic = []
    bic = []
    for n_components in components_range:
        # Create and fit GMM
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(data)
        
        # Store AIC value and model
        aic.append(gmm.aic(data))
        bic.append(gmm.bic(data))
    aic_values.append(aic)
    bic_values.append(bic)

plt.figure(figsize=(12, 6))

for i, dataset in enumerate(datasets):
    plt.plot(components_range, aic_values[i], label=f'{file_name[i]}  AIC', marker='o')
    plt.plot(components_range, bic_values[i], label=f'{file_name[i]}  BIC', marker='x')

plt.xlabel('Number of Components')
plt.ylabel('AIC / BIC Values')
plt.title('AIC and BIC Values for All Elements in Mass Yields')
plt.legend()
plt.grid(True)
plt.show()