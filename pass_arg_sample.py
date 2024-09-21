import numpy as np
import pandas as pd
import discord
import os
import argparse
from colorama import init, Fore, Style
init(autoreset=True)
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic, ExpSineSquared, WhiteKernel


def parse_args():
    parser = argparse.ArgumentParser(description="Process CSV file for nuclear physics data.")
    parser.add_argument("csv_file", help="Path to the CSV file containing nuclear physics data")
    parser.add_argument("Element", help="Element of the CSV file to be analyzed")
    args = parser.parse_args()
    return args

def print_message(message, color=Fore.GREEN, style=Style.BRIGHT):
    print(f"\n{color}[+]{Fore.RESET} {message}\n")

def main():
    args = parse_args()
    try:
        data = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: File not found - {args.csv_file}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file is empty - {args.csv_file}")
        return
    data = pd.read_csv(args.csv_file)
    Z_i = data["Z_i"]
    A = data["A"]
    E = data["E"]
    data_array = np.asarray(data)

    # Extract features and labels
    X_train = data_array[:, 0]
    y_train = data_array[:, 1]

    X_train = X_train.reshape(-1, 1)

    # Fit models with different numbers of components
    components_range = range(1, 11)

    # Initialize variables to store AIC values and GMM models
    aic_values = []
    gmm_models = []

    # Iterate over different numbers of components
    for n_components in components_range:
        # Create and fit GMM
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(data)
        
        # Store AIC value and model
        aic_values.append(gmm.aic(data))
        gmm_models.append(gmm)

    # Find the index of the minimum AIC value
    best_index = np.argmin(aic_values)

    # Retrieve the best GMM model and its number of components
    best_gmm = gmm_models[best_index]
    best_n_components = components_range[best_index]

    # Use the best GMM model for further analysis
    print_message(f"The best number of components is: {best_n_components}")
    best_gmm.fit(data)
    print_message("Model training Warnings and Errors")

    sample, _ = best_gmm.sample(10000)
    sort_sample = np.argsort(sample[:,0])

    samples = pd.DataFrame(sample, columns=["A", "Fission Yield","Z","N", "E"])
    samples.to_csv(f'.\\Results\\Charge_Yields\\Samples_GMM\\{args.Element}_GMM_Z.csv', index=False)
    sample = sample[sort_sample]

    X_test = sample[:,0]
    y_test = sample[:,1]

    X_test = X_test.reshape(-1,1)

    kernel = ConstantKernel() * RBF(length_scale=10, length_scale_bounds=(1e-4, 50)) * \
    RationalQuadratic(length_scale=1, alpha=1, length_scale_bounds=(1e-4, 50), alpha_bounds=(1e-07, 1)) + WhiteKernel(noise_level=0.1) 
    #ExpSineSquared(length_scale=20, periodicity=2)

    # Train Gaussian Process Regressor
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=250)
    gp.fit(X_train, y_train)

    # Predict with uncertainty
    y_pred_1, sigma_1 = gp.predict(X_test, return_std=True)
    y_pred_1 = np.clip(y_pred_1, 0, None)
    print_message("Plotting")
    R_2 = gp.score(X_train, y_train)
    R2 = str(R_2).split('.')[1]
    R2 = R2[:3]
    plt.figure(figsize=(11,7))
    plt.xlim(min(X_train),max(X_train))
    plt.ylim(0,max(y_train)+0.15*max(y_train))
    plt.plot(X_test, y_pred_1,'b-', linewidth=1.5, label=u'Prediction')
    plt.plot(X_train, y_train,'r.', markersize=5,color='darkorange', label=u'Observation')
    plt.fill_between(X_test[:,0], y_pred_1 - 1.96*sigma_1,y_pred_1 + 1.96*sigma_1, alpha=0.2, color='navy', label=u'95% Confidence Interval')
    plt.title(args.Element)
    plt.xlabel('Charge', fontsize=14)
    plt.ylabel("Fission Yield", fontsize=14)
    plt.text(min(X_train)+0.1*min(X_train),max(y_train),f'$R^2= 0.{R2}$', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.show()

    print_message("Sampling...")
    X1 = np.atleast_2d(np.linspace(X_train.min(),X_train.max(), 10000)).T
    y_samples = gp.sample_y(X1)

    sampled_data_df = pd.DataFrame({
    'Z': X1.flatten(),
    'Fission Yield': y_samples[:, 0]})
    sampled_data_df["Z_i"] = Z_i.iloc[0]
    sampled_data_df["A"] = A.iloc[0]
    sampled_data_df["E"] = E.iloc[0]
    sampled_data_df.to_csv(f'.\\Results\\Charge_Yields\\Samples_GPR_WK\\{args.Element}_GPR_Z.csv', index=False)

    # Loss value
    theta = gp.kernel_.theta
    nll_loss = -gp.log_marginal_likelihood(theta)

    # with open('.\Results\Loss.txt', 'a') as f:
    #     f.write(f"{args.Element} Loss : {nll_loss}\n\n")
    #     f.close()
    TOKEN = 'MTI2NzgyNDk2NDI5NjM3NjM4MQ.GD0R3V.Z2R8hGL6Y7Wi2pA-555WOGKyl6HZExyQKThDmc'  # Replace with your bot token
    CHANNEL_ID = 1267829109107589261  # Replace with your channel ID

    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f'Συνδέθηκε ο {client.user}')
        channel = client.get_channel(CHANNEL_ID)
        await channel.send('Ο κώδικας έτρεξε καυλιάρη! Πάμε να τραβήξουμε κανέναν μπέκο τώρα')
        await client.close()

    client.run(TOKEN)
    print_message(f'Negative Log Likelihood Loss: {nll_loss}')



if __name__ == "__main__":
    main()