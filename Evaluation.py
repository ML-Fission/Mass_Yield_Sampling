import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
def compare_datasets(main_dataset_path, GPR_path, GMM_path, title, fig_filename):
    main_df = pd.read_csv(main_dataset_path)
    GPR_df = pd.read_csv(GPR_path)
    GMM_df = pd.read_csv(GMM_path)


    column = 'Fission Yield'

    gmm_color = '#FF5733'  
    gpr_color = '#9B59B6'
    exp_color = '#3366FF'  

    # Kernel Density Estimation
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })

    #sns.set_style('whitegrid')


    sns.kdeplot(main_df[column], label='JENDL', fill=True, ax=axes[0],color=exp_color)
    sns.kdeplot(GMM_df[column], label='GMM', fill=True, ax=axes[0], color=gmm_color)
    axes[0].set_title('Gaussian Mixture Model')
    axes[0].legend()
    sns.kdeplot(main_df[column], label='JENDL', fill=True, ax=axes[1],color=exp_color)
    sns.kdeplot(GPR_df[column], label='GPR', fill=True, ax=axes[1], color=gpr_color)
    axes[1].set_title('Gaussian Process Regression')
    axes[1].legend()
    #fig.suptitle(title, fontsize=22)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("Fission Yield", fontsize=14)
    axes[0].set_ylabel("Density Distribution", fontsize=14)
    axes[1].set_ylabel("Density Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    plt.close()

files_A = ["U-235_14","Pu-240_500","Cm-245_0.0253"]
titles_A = [r"^{235}\text{U}",r"^{240}\text{Pu}",r"^{245}\text{Cm}"]
# files_A = ["Cf-252", "Cm-244", "Cm-246_Ind", "Cm-246", "Th-232", "U-235", "U-238"]
# titles_A = [r"^{252}\text{Cf}",r"^{244}\text{Cm}",r"^{246}\text{Cm} Induced",r"^{246}\text{Cm} Spontaneous",r"^{232}\text{Th}",r"^{235}\text{U}",r"^{238}\text{U}"]
# files_Z = ["Cm-244_0,5", "Fm-255_Thermal", "Pu-239_14", "U-235_14", "U-233_14","Fm-256"]
# titles_Z = [r"^{244}\text{Cm}", r"^{255}\text{Fm}", r"^{239}\text{Pu}", r"^{235}\text{U}", r"^{233}\text{U}",r"^{256}\text{Fm}"]

for i in range(len(files_A)):
    main_dataset_path = fr".\Data\Mass_Yields\{files_A[i]}.csv"
    GPR_path = fr".\Results\Mass_Yields\Samples_GPR\{files_A[i]}_GPR_A.csv"
    GMM_path = fr".\Results\Mass_Yields\Samples_GMM\{files_A[i]}_GMM_A.csv"
    title = fr'Validating the Sampling for ${titles_A[i]}$'
    fig_filename = fr'.\Results\Validation\Mass_Yields\{files_A[i]}_A.png'

    compare_datasets(main_dataset_path, GPR_path, GMM_path, title, fig_filename)

# for i in range(len(files_Z)):
#     main_dataset_path = f".\Data\Charge_Yields\{files_Z[i]}.csv"
#     GPR_path = f".\Results\Charge_Yields\Samples_GPR\{files_Z[i]}_GPR_Z.csv"
#     GMM_path = f".\Results\Charge_Yields\Samples_GMM\{files_Z[i]}_GMM_Z.csv"
#     title = f'Validating the Sampling for ${titles_Z[i]}$'
#     fig_filename = f'.\Results\Validation\Charge_Yields\{files_Z[i]}_Z.png'

#     compare_datasets(main_dataset_path, GPR_path, GMM_path, title, fig_filename)