# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CMF Computation Tool

# %%
import pandas as pd
import statistics


# %% [markdown]
# ## Compute Naive CMFs

# %%
def naive_CMF(data: pd.DataFrame, years_before_treatment=4, years_after_treatment=3):
    freq_before = (len(data[data["Relation To Treatment"] == "before treatment"].index)) / years_before_treatment
    freq_after = len(data[data["Relation To Treatment"] == "after treatment"].index) / years_after_treatment
    if freq_before == 0:
        return 1
    return freq_after/freq_before


# %% [markdown]
# ## Compute Empirical Bayes CMFs

# %% [markdown]
# **Helper Functions**

# %%
def count_curve_crashes(data: pd.DataFrame, curve_data: pd.DataFrame):
    # Helper function to calculate the total number of crashes on each curve for curve rating purposes
    
    # Finding crashes before treatment
    before_data = data[data["Relation To Treatment"] == "before treatment"]
    before_crash_counts = before_data.groupby("CurveID")["CurveID"].count().to_frame("Crashes Before")
    curve_data = curve_data.join(before_crash_counts, on="CurveID", how="left")
    curve_data.loc[:, "Crashes Before"] = curve_data.loc[:, "Crashes Before"].fillna(0)

    # Finding crashes after treatment
    after_data = data[data["Relation To Treatment"] == "after treatment"]
    after_crash_counts = after_data.groupby("CurveID")["CurveID"].count().to_frame("Crashes After")
    curve_data = curve_data.join(after_crash_counts, on="CurveID", how="left")
    curve_data.loc[:, "Crashes After"] = curve_data.loc[:, "Crashes After"].fillna(0)

    # Finding unknown crashes
    unknown_data = data[data["Relation To Treatment"] == "unknown"]
    unknown_crash_counts = unknown_data.groupby("CurveID")["CurveID"].count().to_frame("Unknown Crashes")
    curve_data = curve_data.join(unknown_crash_counts, on="CurveID", how="left")
    curve_data.loc[:, "Unknown Crashes"] = curve_data.loc[:, "Unknown Crashes"].fillna(0)

    # Finding total crashes
    total_crash_counts = data.groupby("CurveID")["CurveID"].count().to_frame("Total Crashes")
    curve_data = curve_data.join(total_crash_counts, on="CurveID", how="left")
    curve_data.loc[:, "Total Crashes"] = curve_data.loc[:, "Total Crashes"].fillna(0)
    
    return curve_data

def calculate_frequencies(curve_data: pd.DataFrame, years_before_treatment, years_after_treatment):
    # Helper function to calculate crash frequencies on curves
    
    # Calculate crash frequencies before treatment
    crash_frequency_before = curve_data["Crashes Before"] / years_before_treatment
    curve_data = pd.merge(curve_data, crash_frequency_before.to_frame("Crash Frequency Before"), left_index=True, right_index=True)
    
    # Calculate crash frequencies after treatment
    crash_frequency_after = curve_data["Crashes After"] / years_after_treatment
    curve_data = pd.merge(curve_data, crash_frequency_after.to_frame("Crash Frequency After"), left_index=True, right_index=True)    
    
    return curve_data

def calculate_curve_ratings(curve_data: pd.DataFrame):
    # Helper function to calculate curve AADT and crash frequency ratings
    
    # Clean through curve data
    curve_AADTs = curve_data["Average AADT"]
    curve_crash_counts = curve_data["Total Crashes"]
    
    # Calculate the curve AADT and crash ratings, and make sure to output the bin boundaries
    AADT_ratings, AADT_bins = pd.qcut(curve_AADTs, 3, ["Low AADT", "Medium AADT", "High AADT"], retbins=True)
    crash_ratings, crash_bins = pd.qcut(curve_crash_counts, 2, ["Low crash frequency", "High crash frequency"], retbins=True)
    
    # Join the calculated ratings to the dataset
    curve_data = pd.merge(curve_data, AADT_ratings.to_frame("AADT Rating"), left_index=True, right_index=True)
    curve_data = pd.merge(curve_data, crash_ratings.to_frame("Crash Frequency Rating"), left_index=True, right_index=True)

    return curve_data, AADT_bins, crash_bins

def calculate_SPF_frequencies(curve_data: pd.DataFrame, coefficients: pd.DataFrame):
    # Helper function to calculate SPF predicted crash frequencies for before and after treatment
    
    # Define the SPF coefficients for legibility
    years = coefficients.loc["Years"][0]
    intercept = coefficients.loc["(Intercept)"][0] / years
    divided = coefficients.loc["divided"][0] / years
    ln_deflection_angle = coefficients.loc["log(deflection_angle)"][0] / years
    length = coefficients.loc["length"][0] / years
    ln_AADT = coefficients.loc["log(AADT)"][0] / years
    ln_BBI = coefficients.loc["log(BBI)"][0] / years
    speed_diff = coefficients.loc["Speed Diff"][0] / years
    
    # Calculate the SPF frequency before
    SPF_before = intercept + (curve_data["Divided"] * divided) + (curve_data["log(devangle)"] * ln_deflection_angle) + (curve_data["Curve Length"] * length) + (curve_data["log(AADT Before)"] * ln_AADT) + (curve_data["log(BBI)"] * ln_BBI) + (curve_data["Speed Diff"] * speed_diff)
    curve_data = pd.merge(curve_data, SPF_before.to_frame("SPF Frequency Before"), left_index=True, right_index=True)

    # Calculate the SPF frequency after
    SPF_after = intercept + (curve_data["Divided"] * divided) + (curve_data["log(devangle)"] * ln_deflection_angle) + (curve_data["Curve Length"] * length) + (curve_data["log(AADT After)"] * ln_AADT) + (curve_data["log(BBI)"] * ln_BBI) + (curve_data["Speed Diff"] * speed_diff)
    curve_data = pd.merge(curve_data, SPF_after.to_frame("SPF Frequency After"), left_index=True, right_index=True)
    
    return curve_data

def calculate_cumulative_values(curve_data: pd.DataFrame, coefficients: pd.DataFrame):
    # Helper function to find aggregate values
    
    # Calculation of cumulative frequencies
    total_frequency_before = curve_data["Crash Frequency Before"].sum()
    total_frequency_after = curve_data["Crash Frequency After"].sum()
    total_spf_frequency_before = curve_data["SPF Frequency Before"].sum()
    total_spf_frequency_after = curve_data["SPF Frequency After"].sum()
    
    # Calculation of weight of SPF
    dispersion = coefficients.loc["Dispersion"][0]
    weight_of_spf = 1 / (1 + dispersion * total_spf_frequency_before)

    # Calculation of expected crashes before and after
    expected_crashes_before = (weight_of_spf * total_spf_frequency_before) + (1 - weight_of_spf) * (total_frequency_before)
    comparison_ratio = total_spf_frequency_after / total_spf_frequency_before
    expected_crashes_after = expected_crashes_before * comparison_ratio
    
    # Collect all calculated values into a dictionary
    cumulative_dict = {"Total Frequency Before" : total_frequency_before,
                       "Total Frequency After" : total_frequency_after,
                       "Total SPF Frequency Before" : total_spf_frequency_before,
                       "Total SPF Frequency After" : total_spf_frequency_after,
                       "Weight of SPF" : weight_of_spf,
                       "Expected Crashes Before" : expected_crashes_before,
                       "Comparison Ratio" : comparison_ratio,
                       "Expected Crashes After" : expected_crashes_after,
                      }
    
    return cumulative_dict

def calculate_final_outputs(curve_data: pd.DataFrame, cumulative_dict: dict):
    # Helper function to calculate variance, CMF, and standard error
    
    # Calculate variance of expected after
    variance_of_expected_after = cumulative_dict["Expected Crashes After"] * cumulative_dict["Comparison Ratio"] * (1 - cumulative_dict["Weight of SPF"])
    
    # Calculate CMF
    empirical_bayes_cmf = (cumulative_dict["Total Frequency After"] / cumulative_dict["Expected Crashes After"]) / (1 + (variance_of_expected_after / (cumulative_dict["Expected Crashes After"] ** 2)))
    
    # Calculate variance of total frequency after
    variance_of_total_frequency_after = statistics.variance(curve_data["Crash Frequency After"])

    # Calculate standard deviation of CMF
    standard_deviation = ((empirical_bayes_cmf ** 2) *
                            (
                                ((variance_of_expected_after / (cumulative_dict["Expected Crashes After"] ** 2)) + (variance_of_total_frequency_after / (cumulative_dict["Total Frequency After"] ** 2)))
                                /
                                ((1 + variance_of_expected_after / (cumulative_dict["Expected Crashes After"] ** 2) ** 2))
                            )
                         ) ** 0.5
    
    # Collect all calculated values into a dictionary
    results_dict = {"Variance of Expected After" : variance_of_expected_after,
                    "Empirical Bayes CMF" : empirical_bayes_cmf,
                    "Variance of Total Frequency After" : variance_of_total_frequency_after,
                    "CMF Standard Deviation" : standard_deviation
                   }
    return results_dict


# %% [markdown]
# **Empirical Bayes Function**

# %%
def empirical_bayes_CMF(data: pd.DataFrame, curve_data: pd.DataFrame, coefficients: pd.DataFrame, years_before_treatment=4, years_after_treatment=3):
    curve_data = count_curve_crashes(data, curve_data)
    curve_data = calculate_frequencies(curve_data, years_before_treatment, years_after_treatment)
    curve_data, AADT_bins, crash_bins = calculate_curve_ratings(curve_data)
    curve_data = calculate_SPF_frequencies(curve_data, coefficients)
    cumulative_dict = calculate_cumulative_values(curve_data, coefficients)
    results_dict = calculate_final_outputs(curve_data, cumulative_dict)
    return results_dict, AADT_bins, crash_bins


# %% [markdown]
# ## Import Data

# %%
# Read all crash data
D1_data = pd.read_excel('data/D1_Phonolite_Input.xlsx', sheet_name = "Formatting Data")
D2_data = pd.read_excel('data/D2_LWA_Input.xlsx', sheet_name = "Formatting Data")
D6_data = pd.read_excel('data/D6_HFST_Input.xlsx', sheet_name = "Formatting Data")

# Read relevant curve data
D1_curve_data = pd.read_excel('data/D1_Phonolite_Input.xlsx', sheet_name = "Curve Info", )
D6_curve_data = pd.read_excel('data/D6_HFST_Input.xlsx', sheet_name = "Curve Info")

# Create Filter Datasets
D1_single_vehicle = D1_data[D1_data["Single Vehicle"] == "Yes"]
D1_curve_crashes = D1_data[D1_data["Vehicle_Ma"].str.contains("Negotiating a Curve")]
D1_wet_road = D1_data[D1_data["Surface_Co"] == "Wet"]

D2_single_vehicle = D2_data[D2_data["Single Vehicle"] == "Yes"]
D2_curve_crashes = D2_data[D2_data["Vehicle_Ma"].str.contains("Negotiating a Curve")]
D2_wet_road = D2_data[D2_data["Surface_Co"] == "Wet"]

D6_single_vehicle = D6_data[D6_data["Single Vehicle"] == "Yes"]
D6_curve_crashes = D6_data[D6_data["Vehicle_Ma"].str.contains("Negotiating a Curve")]
D6_wet_road = D6_data[D6_data["Surface_Co"] == "Wet"]

# %%
# Import SPF coefficients
D1_total_coeff = pd.read_excel('data/D1_Phonolite_Input.xlsx', sheet_name = "Total Crashes SPF Coefficients", index_col = 0)
D1_single_vehicle_coeff = pd.read_excel('data/D1_Phonolite_Input.xlsx', sheet_name = "Single Vehicle SPF Coefficients", index_col = 0)
D1_curve_crashes_coeff = pd.read_excel('data/D1_Phonolite_Input.xlsx', sheet_name = "Curve Crashes SPF Coefficients", index_col = 0)
D1_wet_road_coeff = pd.read_excel('data/D1_Phonolite_Input.xlsx', sheet_name = "Wet Road SPF Coefficients", index_col = 0)

D6_total_coeff = pd.read_excel('data/D6_HFST_Input.xlsx', sheet_name = "Total Crashes SPF Coefficients", index_col = 0)
D6_single_vehicle_coeff = pd.read_excel('data/D6_HFST_Input.xlsx', sheet_name = "Single Vehicle SPF Coefficients", index_col = 0)
D6_curve_crashes_coeff = pd.read_excel('data/D6_HFST_Input.xlsx', sheet_name = "Curve Crashes SPF Coefficients", index_col = 0)
D6_wet_road_coeff = pd.read_excel('data/D6_HFST_Input.xlsx', sheet_name = "Wet Road SPF Coefficients", index_col = 0)

# %% [markdown]
# ## Naive CMFs

# %% [markdown]
# District 1 Phonolite

# %%
naive_CMF(D1_data), naive_CMF(D1_single_vehicle), naive_CMF(D1_curve_crashes), naive_CMF(D1_wet_road)

# %% [markdown]
# District 2 LWA

# %%
naive_CMF(D2_data), naive_CMF(D2_single_vehicle), naive_CMF(D2_curve_crashes), naive_CMF(D2_wet_road)

# %% [markdown]
# District 6 HFST

# %%
naive_CMF(D6_data), naive_CMF(D6_single_vehicle), naive_CMF(D6_curve_crashes), naive_CMF(D6_wet_road)

# %% [markdown]
# ## Empirical CMFs

# %%
empirical_bayes_CMF(D6_data, D6_curve_data, D6_total_coeff)

# %%
empirical_bayes_CMF(D6_single_vehicle, D6_curve_data, D6_single_vehicle_coeff)

# %%
