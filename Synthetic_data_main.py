#--------------------------------------------------------------
#Reference -Modeling Tabular data using Conditional GAN Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni
# Synthetic data generation methods in healthcare: A review on open-source tools and methods Vasileios C. Pezoulas , Dimitrios I. Zaridis, Eugenia Mylonaa,, Christos Androutsos, Kosmas Apostolidisa, Nikolaos S. Tachos, Dimitrios I. Fotiadis 
# Synthetic Data Vault (SDV) Official Documentation
# Dataset- 
# COVID-19 patient's symptoms, status, and medical history. Meir Nizri(Owner),The dataset was provided by the Mexican government
# Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. 
# @inproceedings{ctgan,
# title={Modeling Tabular data using Conditional GAN},
# author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
# booktitle={Advances in Neural Information Processing Systems},
# year={2019}
# }
#--------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from ctgan import CTGAN
from sdv.single_table import CTGANSynthesizer
import os
# %%

# Output folder for plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Dataset Folder
fnpath = 'datasets'
fn = 'Covid_Data.csv'

real_data = pd.read_csv(os.path.join(fnpath, fn))

# Replace invalid values with NaN
missing_value_cols = [
    "INTUBED", "PNEUMONIA", "PREGNANT", "DIABETES", "COPD", "ASTHMA",
    "INMSUPR", "HIPERTENSION", "OTHER_DISEASE", "CARDIOVASCULAR",
    "OBESITY", "RENAL_CHRONIC", "TOBACCO", "ICU"
]
for col in missing_value_cols:
    if col in real_data.columns:
        real_data[col] = pd.to_numeric(real_data[col], errors='coerce')
        real_data[col] = real_data[col].replace([97, 98, 99], np.nan)

# Drop non-useful column
real_data.drop(columns=["DATE_DIED"], errors='ignore', inplace=True)

# Replace NaN in AGE
real_data["AGE"] = pd.to_numeric(real_data["AGE"], errors='coerce')
real_data["AGE"].fillna(real_data["AGE"].mean(), inplace=True)
real_data = real_data.sample(n=1000)

# Categorical columns
categorical_columns = [
    "SEX", "PATIENT_TYPE", "INTUBED", "PNEUMONIA", "PREGNANT",
    "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
    "OTHER_DISEASE", "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC",
    "TOBACCO", "ICU"
]

# Convert categorical columns to string
for col in categorical_columns:
    if col in real_data.columns:
        real_data[col] = real_data[col].astype(str)

print("\n Preview of Preprocessed Real Data:")
print(real_data.head())
print("Shape",real_data.shape)

# %%
# Creating Metadata for SDV Library Synthesizers
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

# CTGAN
print("\n Training CTGAN model...")
ctgan_synthesizer = CTGANSynthesizer(metadata,verbose =True)
ctgan_synthesizer.fit(real_data)

synthetic_data_ctgan = ctgan_synthesizer.sample(num_rows=1000)

# %%
# Loss value for CTGAN
import plotly.io as pio
pio.renderers.default = 'browser'
#pio.renderers.default = 'browser' 
ctgan_synthesizer.get_loss_values()
fig = ctgan_synthesizer.get_loss_values_plot()
fig.update_layout(title="Loss Curves for CTGAN model over epochs for COVID-19 Dataset")
fig.show()
fig.write_image(os.path.join(output_dir, "ctgan_loss_plot.png"))
# %%
# Diagonostic Report for CTGAN
from sdv.evaluation.single_table import run_diagnostic

diagnostic_report = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data_ctgan,
    metadata=metadata, verbose=True)

diagnostic_report.get_details(property_name='Data Structure')
diagnostic_report.get_details(property_name='Data Validity')
# %%
#Quality Report for CTGAN

from sdv.evaluation.single_table import evaluate_quality

quality_report = evaluate_quality(
    real_data=real_data,
    synthetic_data=synthetic_data_ctgan,
    metadata=metadata)

# %%
# Example of Data coloumn compared to orignal

from sdv.evaluation.single_table import get_column_plot

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data_ctgan,
    metadata=metadata,
    column_name='AGE' # Change coloumn name to visualize different coloumns
)
fig.update_layout(title="Age coloumn frequency distribution in CTGAN generated synthetic vs orignal dataset in COVID-19 dataset")
fig.show()

fig.write_image(os.path.join(output_dir, "CTGAN_data_for_covid_AGE_column.png"))

# %%

# TVAE
print("\n Training TVAE model...")
tvae = TVAESynthesizer(metadata)
tvae.fit(real_data)
print(" TVAE Training Completed!")

synthetic_data_tvae = tvae.sample(num_rows=1000)
synthetic_data_tvae.columns = real_data.columns
synthetic_data_tvae["AGE"] = synthetic_data_tvae["AGE"].round().astype(int).clip(1, 100)

# Save synthetic data
CTGAN_covid_fn = "Covid_Synthetic_Data_CTGAN.csv"
TVAE_covid_fn = "Covid_Synthetic_Data_TVAE.csv"
synthetic_data_ctgan.to_csv(os.path.join(fnpath, CTGAN_covid_fn), index=False)
synthetic_data_tvae.to_csv(os.path.join(fnpath, TVAE_covid_fn), index=False)

print("\n Synthetic datasets saved")


# %%
# Diagonostic Report for TVAE

from sdv.evaluation.single_table import run_diagnostic

diagnostic_report = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data_tvae,
    metadata=metadata, verbose=True)

diagnostic_report.get_details(property_name='Data Structure')
diagnostic_report.get_details(property_name='Data Validity')
# %%

# Quality Report for TVAE
from sdv.evaluation.single_table import evaluate_quality

quality_report = evaluate_quality(
    real_data=real_data,
    synthetic_data=synthetic_data_tvae,
    metadata=metadata)

# %%

# Example of Data coloumn compared to orignal

from sdv.evaluation.single_table import get_column_plot

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data_tvae,
    metadata=metadata,
    column_name='AGE' #Change coloumn name for different coloumns
)
fig.update_layout(title="Age coloumn frequency distribution in TVAE generated synthetic vs orignal dataset in COVID-19 dataset")
fig.show()

fig.write_image(os.path.join(output_dir, "TVAE_Covid_data_AGE.png"))
# %%
#CopulaGANSynthesizer

from sdv.single_table import GaussianCopulaSynthesizer

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data)

gauss_synthetic_data = synthesizer.sample(num_rows=1000)

# %%
# Example of Data coloumn compared to orignal

from sdv.evaluation.single_table import get_column_plot

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=gauss_synthetic_data,
    metadata=metadata,
    column_name='AGE' #Change coloumn name for different coloumns
)
fig.update_layout(title="Age coloumn frequency distribution in Copula GAN Synthesizer generated synthetic vs orignal dataset in COVID-19 dataset")
fig.show()
fig.write_image(os.path.join(output_dir, "CopulaGANSynthesizer_Covid_data_AGE.png"))
# %%

##Training AI model(s)

# Label Encoding for RandomForest
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    real_data[col] = le.fit_transform(real_data[col])
    synthetic_data_ctgan[col] = le.transform(synthetic_data_ctgan[col])
    synthetic_data_tvae[col] = le.transform(synthetic_data_tvae[col])
    label_encoders[col] = le



# RandomForest on CTGAN Data for demonstration of  SHAP

X_ctgan = synthetic_data_ctgan.drop(columns=["ICU"])
y_ctgan = synthetic_data_ctgan["ICU"].astype(int)
X_train_ctgan, X_test_ctgan, y_train_ctgan, y_test_ctgan = train_test_split(X_ctgan, y_ctgan, test_size=0.2, random_state=42)
model_ctgan = RandomForestClassifier(n_estimators=100, random_state=42)
model_ctgan.fit(X_train_ctgan, y_train_ctgan)
print("\n RandomForest Model Trained for CTGAN")

# SHAP on RamdomForest trained CTGAN
print(" Generating SHAP Summary Plot for CTGAN data...")
explainer_ctgan = shap.Explainer(model_ctgan, X_train_ctgan)
shap_values_ctgan = explainer_ctgan(X_test_ctgan)
plt.title("SHAP Summary Plot – CTGAN data for COVID-19 dataset")
shap.plots.bar(shap_values_ctgan[:, :, 1], show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"shap_ctgan.png"))
plt.show()

# RandomForest on TVAE data
X_tvae = synthetic_data_tvae.drop(columns=["ICU"])
y_tvae = synthetic_data_tvae["ICU"].astype(int)
X_train_tvae, X_test_tvae, y_train_tvae, y_test_tvae = train_test_split(X_tvae, y_tvae, test_size=0.2, random_state=42)
model_tvae = RandomForestClassifier(n_estimators=100, random_state=42)
model_tvae.fit(X_train_tvae, y_train_tvae)
print("\n RandomForest Model Trained for TVAE")

# SHAP on RandomForest Trained on TVAE data
print(" Generating SHAP Summary Plot for TVAE data...")
explainer_tvae = shap.Explainer(model_tvae, X_train_tvae)
shap_values_tvae = explainer_tvae(X_test_tvae)
plt.title("SHAP Summary Plot – TVAE data for COVID-19 dataset")
shap.plots.bar(shap_values_tvae[:, :, 1], show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"shap_tvae.png"))
plt.show()

# %%


# --- Plot Comparison: Real vs CTGAN vs TVAE ---

def plot_numerical_distribution(column, real, ctgan, tvae):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(real[column], label="Real", fill=True)
    sns.kdeplot(ctgan[column], label="CTGAN", fill=True)
    sns.kdeplot(tvae[column], label="TVAE", fill=True)
    plt.title(f"Distribution Comparison: {column}")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"{column}_distribution_comparison.png"))
    plt.show()

# %%

def plot_categorical_distribution(column, real, ctgan, tvae):
    # Define the mapping: 1 = 'Yes', 2 = 'No'
    value_map = {1: 'Yes', 2: 'No'}

    # Safely map values; ignore anything not in the map (e.g., 0)
    real_mapped = real[column].map(value_map).dropna()
    ctgan_mapped = ctgan[column].map(value_map).dropna()
    tvae_mapped = tvae[column].map(value_map).dropna()

    # Calculate normalized value counts
    real_counts = real_mapped.value_counts(normalize=True).sort_index()
    ctgan_counts = ctgan_mapped.value_counts(normalize=True).sort_index()
    tvae_counts = tvae_mapped.value_counts(normalize=True).sort_index()

    # Union of all categories across datasets
    categories = sorted(set(real_counts.index) | set(ctgan_counts.index) | set(tvae_counts.index))

    # Build the DataFrame for plotting
    df_plot = pd.DataFrame({
        "Category": categories,
        "Real": [real_counts.get(cat, 0) for cat in categories],
        "CTGAN": [ctgan_counts.get(cat, 0) for cat in categories],
        "TVAE": [tvae_counts.get(cat, 0) for cat in categories]
    })

    # Melt for Seaborn
    df_plot = pd.melt(df_plot, id_vars="Category", var_name="Dataset", value_name="Proportion")

    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_plot, x="Category", y="Proportion", hue="Dataset")
    plt.title(f"Categorical Comparison: {column}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{column}_categorical_comparison.png"))


# %%

'''
def plot_categorical_distribution(column, real, ctgan, tvae):
    plt.figure(figsize=(8, 5))
    real_counts = real[column].value_counts(normalize=True).sort_index()
    ctgan_counts = ctgan[column].value_counts(normalize=True).sort_index()
    tvae_counts = tvae[column].value_counts(normalize=True).sort_index()
    categories = sorted(set(real_counts.index) | set(ctgan_counts.index) | set(tvae_counts.index))

    df_plot = pd.DataFrame({
        "Category": categories,
        "Real": [real_counts.get(cat, 0) for cat in categories],
        "CTGAN": [ctgan_counts.get(cat, 0) for cat in categories],
        "TVAE": [tvae_counts.get(cat, 0) for cat in categories]
    })
    df_plot = pd.melt(df_plot, id_vars="Category", var_name="Dataset", value_name="Proportion")
    sns.barplot(data=df_plot, x="Category", y="Proportion", hue="Dataset")
    plt.title(f"Categorical Comparison: {column}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"{column}_categorical_comparison.png"))
    plt.show()
'''
# %%


# Example comparisons
plot_numerical_distribution("AGE", real_data, synthetic_data_ctgan, synthetic_data_tvae)
plot_categorical_distribution("ICU", real_data, synthetic_data_ctgan, synthetic_data_tvae)
plot_categorical_distribution("PREGNANT", real_data, synthetic_data_ctgan, synthetic_data_tvae)
plot_categorical_distribution("OBESITY", real_data, synthetic_data_ctgan, synthetic_data_tvae)
plot_categorical_distribution("COPD", real_data, synthetic_data_ctgan, synthetic_data_tvae)



#Real vs Synthetic Data (Statistical Summary)
print("\n Summary Statistics - Real Data:")
print(real_data.describe())

print("\n Summary Statistics - Synthetic Data CTGAN:")
print(synthetic_data_ctgan.describe())

print("\n Summary Statistics - Synthetic Data TVAE:")
print(synthetic_data_tvae.describe())


# Print summary statistics
print("\n Summary Statistics Comparison:")
print("\n AGE")
print("Real:")
print(real_data["AGE"].describe())
print("\nCTGAN:")
print(synthetic_data_ctgan["AGE"].describe())
print("\nTVAE:")
print(synthetic_data_tvae["AGE"].describe())
print("\nCopulaGANSynthesize:")
print(gauss_synthetic_data["AGE"].describe())

print("\n ICU")
print("Real:")
print(real_data["ICU"].value_counts(normalize=True))
print("\nCTGAN:")
print(synthetic_data_ctgan["ICU"].value_counts(normalize=True))
print("\nTVAE:")
print(synthetic_data_tvae["ICU"].value_counts(normalize=True))
print("\nCopulaGANSynthesize:")
print(gauss_synthetic_data["ICU"].describe())

print("\n OBESITY")
print("Real:")
print(real_data["OBESITY"].value_counts(normalize=True))
print("\OBESITY:")
print(synthetic_data_ctgan["OBESITY"].value_counts(normalize=True))
print("\nTVAE:")
print(synthetic_data_tvae["OBESITY"].value_counts(normalize=True))
print("\nCopulaGANSynthesize:")
print(gauss_synthetic_data["OBESITY"].describe())

# %%

# For Cardio Data
# COSC-4207 - Seminars in Computer Science
# Synthetic Data Generation using CTGAN for Cardiovasicular data
# Submitted by - Sheshank Priyadarshi and Parth Chopra
# Submitted to - Professor W
#--------------------------------------------------------------
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ctgan import CTGAN
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from ctgan import CTGAN
import os
# Load and Preprocess the data

# Load the dataset
fnpath = 'datasets'
h_fn = 'heart_disease_dataset.csv'

h_real_data = pd.read_csv(os.path.join(fnpath, h_fn))

# def check_values(df):
#     return df.isin([97, 98, 99]).any()
# print("Coloumns with Nan values")
# check_values(h_real_data)

missing_value_cols =['thal', 'ca',]

# Creating Metadata for SDV Library
from sdv.metadata import Metadata
h_metadata = Metadata.detect_from_dataframe(
    data=h_real_data,
    table_name='Heart')

print(h_real_data.shape)
print(h_real_data.head())
print(h_metadata)

# %%
#CTGAN synthesizer

from sdv.single_table import CTGANSynthesizer

h_ctgan_synthesizer = CTGANSynthesizer(h_metadata,verbose =True)
h_ctgan_synthesizer.fit(h_real_data)

h_ctgan_synthetic_data = h_ctgan_synthesizer.sample(num_rows=1000)



# %%

import plotly.io as pio
#pio.renderers.default = 'browser' 
h_ctgan_synthesizer.get_loss_values()
fig = h_ctgan_synthesizer.get_loss_values_plot()
fig.update_layout(title="CTGAN loss values over epochs for Cardio Data")
fig.show()
fig.write_image(os.path.join(output_dir, "CTGAN_loss_values_over_epochs_for_Cardio_Data.png"))
# %%
# Diagnostic report CTGAN
from sdv.evaluation.single_table import run_diagnostic

diagnostic_report = run_diagnostic(
    real_data=h_real_data,
    synthetic_data=h_ctgan_synthetic_data,
    metadata=h_metadata, verbose=True)

diagnostic_report.get_details(property_name='Data Structure')
diagnostic_report.get_details(property_name='Data Validity')
# %%

## Evaluation report CTGAN
from sdv.evaluation.single_table import evaluate_quality

quality_report = evaluate_quality(
    real_data=h_real_data,
    synthetic_data=h_ctgan_synthetic_data,
    metadata=h_metadata)

# %%

#Plot
from sdv.evaluation.single_table import get_column_plot

fig = get_column_plot(
    real_data=h_real_data,
    synthetic_data=h_ctgan_synthetic_data,
    metadata=h_metadata,
    column_name='age'
)
fig.update_layout(title="CTGAN data AGE coloumn for Cardio")
fig.show()
fig.write_image(os.path.join(output_dir, "CTGAN_data_AGE_Cardio.png"))

# %%

# TVAE on Cardio

from sdv.single_table import TVAESynthesizer

h_tvae_synthesizer = TVAESynthesizer(h_metadata, verbose=True)
h_tvae_synthesizer.fit(h_real_data)

h_tvae_synthetic_data = h_tvae_synthesizer.sample(num_rows=1000)


# %%

# Diagnostic Report TVAE
from sdv.evaluation.single_table import run_diagnostic

diagnostic_report = run_diagnostic(
    real_data=h_real_data,
    synthetic_data=h_tvae_synthetic_data,
    metadata=h_metadata, verbose=True)

diagnostic_report.get_details(property_name='Data Structure')
diagnostic_report.get_details(property_name='Data Validity')
# %%
# Evaluate Report TVAE

from sdv.evaluation.single_table import evaluate_quality

quality_report = evaluate_quality(
    real_data=h_real_data,
    synthetic_data=h_tvae_synthetic_data,
    metadata=h_metadata)

# %%

from sdv.evaluation.single_table import get_column_plot

fig = get_column_plot(
    real_data=h_real_data,
    synthetic_data=h_tvae_synthetic_data,
    metadata=h_metadata,
    column_name='age'
)

fig.update_layout(title="TVAE data AGE column for Cardio")
    
fig.show()
# %%

CTGAN_heart_fn = "Heart_Synthetic_Data_CTGAN.csv"
TVAE_heart_fn = "Heart_Synthetic_Data_TVAE.csv"
h_ctgan_synthetic_data.to_csv(os.path.join(fnpath, CTGAN_heart_fn), index=False)
h_tvae_synthetic_data.to_csv(os.path.join(fnpath, TVAE_heart_fn), index=False)


# %%

#Real vs Synthetic Data (Statistical Summary)
print("\n Summary Statistics - Real Data:")
print(h_real_data.describe())

print("\n Summary Statistics - Synthetic Data CTGAN:")
print(h_ctgan_synthetic_data.describe())

print("\n Summary Statistics - Synthetic Data TVAE:")
print(h_tvae_synthetic_data.describe())




