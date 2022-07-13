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
import numpy as np
import warnings
warnings.filterwarnings("error")


# %% [markdown]
# ## Naive CMFs Functions

# %%
def naive_CMF(data: pd.DataFrame, years_before_treatment=4, years_after_treatment=3):
    freq_before = (len(data[data["Relation To Treatment"] == "before treatment"].index)) / years_before_treatment
    freq_after = len(data[data["Relation To Treatment"] == "after treatment"].index) / years_after_treatment
    if freq_before == 0:
        return 1
    return freq_after/freq_before


# %% [markdown]
# ## Empirical Bayes CMFs Functions

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
    
    # Exclude curves with no crashes from ratings
    curve_data_view = curve_data.query("`Crashes Before` > 0")
    
    # Clean through curve data
    curve_AADTs = curve_data_view["Average AADT"]
    curve_crash_counts = curve_data_view["Crashes Before"]

    # Calculate the curve AADT and crash ratings, and make sure to output the bin boundaries
    AADT_ratings, AADT_bins = pd.qcut(curve_AADTs, 2, ["Low AADT", "High AADT"], retbins=True)
    crash_ratings, crash_bins = pd.cut(curve_crash_counts, np.array([0, curve_data_view["Crashes Before"].mean(), curve_data_view["Crashes Before"].max()]), labels=["Low Crash Frequency", "High Crash Frequency"], retbins=True)

    # Join the calculated ratings to CurveIDs, then join those smaller views to the dataset
    curve_AADTs = pd.merge(curve_data_view["CurveID"].to_frame(), AADT_ratings.to_frame("AADT Rating"), left_index=True, right_index=True)
    curve_crash_counts = pd.merge(curve_data_view["CurveID"].to_frame(), crash_ratings.to_frame("Crash Frequency Rating"), left_index=True, right_index=True)
    curve_data = pd.merge(curve_data, curve_AADTs, on="CurveID", how="left")
    curve_data = pd.merge(curve_data, curve_crash_counts, on="CurveID", how="left")

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
    SPF_before = np.exp(intercept + (curve_data["Divided"] * divided) + (curve_data["log(devangle)"] * ln_deflection_angle) + (curve_data["Curve Length"] * length) + (curve_data["log(AADT Before)"] * ln_AADT) + (curve_data["log(BBI)"] * ln_BBI) + (curve_data["Speed Diff"] * speed_diff))
    curve_data = pd.merge(curve_data, SPF_before.to_frame("SPF Frequency Before"), left_index=True, right_index=True)

    # Calculate the SPF frequency after
    SPF_after = np.exp(intercept + (curve_data["Divided"] * divided) + (curve_data["log(devangle)"] * ln_deflection_angle) + (curve_data["Curve Length"] * length) + (curve_data["log(AADT After)"] * ln_AADT) + (curve_data["log(BBI)"] * ln_BBI) + (curve_data["Speed Diff"] * speed_diff))
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
# **Empirical Bayes Filter by AADT and Crash Frequency Rating**

# %%
def filter_by_rating(data: pd.DataFrame, curve_data: pd.DataFrame, rating_filters: tuple):
    # Helper function to filter crash and curve data by the AADT ratings and crash frequency ratings
    
    # Clarifying the filters by making separate variables
    AADT_rating_filter = rating_filters[0]
    crash_rating_filter = rating_filters[1]
    
    # Joining the ratings to the crash data
    data = data.join(curve_data[["CurveID", "AADT Rating", "Crash Frequency Rating"]].set_index("CurveID"), on="CurveID")
    
    # Filtering the data and curve data based on the ratings
    data = data.query("`AADT Rating` == @AADT_rating_filter & `Crash Frequency Rating` == @crash_rating_filter")
    curve_data = curve_data.query("`AADT Rating` == @AADT_rating_filter & `Crash Frequency Rating` == @crash_rating_filter")
    
    return data, curve_data

def filter_calculate_cumulative_values(curve_data: pd.DataFrame, coefficients: pd.DataFrame, rating_filters: tuple):
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
    try:
        comparison_ratio = total_spf_frequency_after / total_spf_frequency_before
    except RuntimeWarning:
        comparison_ratio = np.nan
        print("The filters of " + rating_filters[0] + " and " + rating_filters[1] + " either has no curves associated with it, and the value of the comparison ratio is NaN.")
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

def filter_calculate_final_outputs(curve_data: pd.DataFrame, cumulative_dict: dict, rating_filters: tuple):
    # Helper function to calculate variance, CMF, and standard error
    
    # Calculate variance of expected after
    variance_of_expected_after = cumulative_dict["Expected Crashes After"] * cumulative_dict["Comparison Ratio"] * (1 - cumulative_dict["Weight of SPF"])

    # Calculate CMF
    empirical_bayes_cmf = (cumulative_dict["Total Frequency After"] / cumulative_dict["Expected Crashes After"]) / (1 + (variance_of_expected_after / (cumulative_dict["Expected Crashes After"] ** 2)))

    # Calculate variance of total frequency after
    try:
        variance_of_total_frequency_after = statistics.variance(curve_data["Crash Frequency After"])
    except Exception:
        variance_of_total_frequency_after = np.nan
        print("The filters of " + rating_filters[0] + " and " + rating_filters[1] + " either has one or no curves associated with it, and the values in the results dictionary will all be NaN.")

    # Calculate standard deviation of CMF
    try:
        standard_deviation = ((empirical_bayes_cmf ** 2) *
                                (
                                    ((variance_of_expected_after / (cumulative_dict["Expected Crashes After"] ** 2)) + (variance_of_total_frequency_after / (cumulative_dict["Total Frequency After"] ** 2)))
                                    /
                                    ((1 + variance_of_expected_after / (cumulative_dict["Expected Crashes After"] ** 2) ** 2))
                                )
                             ) ** 0.5
    except RuntimeWarning:
        standard_deviation = np.nan
        print("The filters of " + rating_filters[0] + " and " + rating_filters[1] + " has no crashes after treatment, and so the EB CMF defaults to 0 and the Stdev cannot be calculated.")

    # Collect all calculated values into a dictionary
    results_dict = {"Variance of Expected After" : variance_of_expected_after,
                    "Empirical Bayes CMF" : empirical_bayes_cmf,
                    "Variance of Total Frequency After" : variance_of_total_frequency_after,
                    "CMF Standard Deviation" : standard_deviation
                   }

    return results_dict

def filter_empirical_bayes_CMF(data: pd.DataFrame, curve_data: pd.DataFrame, coefficients: pd.DataFrame, rating_filters: tuple, years_before_treatment=4, years_after_treatment=3):
    curve_data = count_curve_crashes(data, curve_data)
    curve_data = calculate_frequencies(curve_data, years_before_treatment, years_after_treatment)
    curve_data, AADT_bins, crash_bins = calculate_curve_ratings(curve_data)
    data, curve_data = filter_by_rating(data, curve_data, rating_filters)
    curve_data = calculate_SPF_frequencies(curve_data, coefficients)
    cumulative_dict = filter_calculate_cumulative_values(curve_data, coefficients, rating_filters)
    results_dict = filter_calculate_final_outputs(curve_data, cumulative_dict, rating_filters)
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
display(empirical_bayes_CMF(D6_data, D6_curve_data, D6_total_coeff))
display(empirical_bayes_CMF(D6_single_vehicle, D6_curve_data, D6_single_vehicle_coeff))
display(empirical_bayes_CMF(D6_curve_crashes, D6_curve_data, D6_curve_crashes_coeff))
display(empirical_bayes_CMF(D6_wet_road, D6_curve_data, D6_wet_road_coeff))

# %%
display(empirical_bayes_CMF(D1_data, D1_curve_data, D1_total_coeff))
display(empirical_bayes_CMF(D1_single_vehicle, D1_curve_data, D1_single_vehicle_coeff))
display(empirical_bayes_CMF(D1_curve_crashes, D1_curve_data, D1_curve_crashes_coeff))
display(empirical_bayes_CMF(D1_wet_road, D1_curve_data, D1_wet_road_coeff))

# %% [markdown]
# ## D6 HFST Filtered by AADT and Crash Frequency Ratings

# %%
rating_filters = [("Low AADT", "Low Crash Frequency"),
           ("High AADT", "Low Crash Frequency"),
           ("Low AADT", "High Crash Frequency"),
           ("High AADT", "High Crash Frequency"),
          ]

# %% [markdown]
# **D6 Total**

# %%
D6_total_filters = []
for filter in rating_filters:
    results_dict, AADT_bins, crash_bins = filter_empirical_bayes_CMF(D6_data, D6_curve_data, D6_total_coeff, filter)
    results_dict.update({"AADT Rating" : filter[0],
                         "Crash Frequency Rating" : filter[1],
                         "AADT Bins" : AADT_bins,
                         "Crash Frequency Bins" : crash_bins}
                        )
    D6_total_filters.append(results_dict)
D6_total_filters_df = pd.DataFrame(D6_total_filters)
display(D6_total_filters_df)

index_order = ["Low Crash Frequency", "High Crash Frequency"]
column_order = ["Low AADT", "High AADT"]
display(D6_total_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))
display(D6_total_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))

# %% [markdown]
# **D6 Single Vehicle**

# %%
D6_single_vehicle_filters = []
for filter in rating_filters:
    results_dict, AADT_bins, crash_bins = filter_empirical_bayes_CMF(D6_single_vehicle, D6_curve_data, D6_single_vehicle_coeff, filter)
    results_dict.update({"AADT Rating" : filter[0],
                         "Crash Frequency Rating" : filter[1],
                         "AADT Bins" : AADT_bins,
                         "Crash Frequency Bins" : crash_bins}
                        )
    D6_single_vehicle_filters.append(results_dict)
D6_single_vehicle_filters_df = pd.DataFrame(D6_single_vehicle_filters)
display(D6_single_vehicle_filters_df)

index_order = ["Low Crash Frequency", "High Crash Frequency"]
column_order = ["Low AADT", "High AADT"]

display(D6_single_vehicle_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))
display(D6_single_vehicle_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))

# %% [markdown]
# **D6 Curve Crashes**

# %%
D6_curve_crashes_filters = []
for filter in rating_filters:
    results_dict, AADT_bins, crash_bins = filter_empirical_bayes_CMF(D6_curve_crashes, D6_curve_data, D6_curve_crashes_coeff, filter)
    results_dict.update({"AADT Rating" : filter[0],
                         "Crash Frequency Rating" : filter[1],
                         "AADT Bins" : AADT_bins,
                         "Crash Frequency Bins" : crash_bins}
                        )
    D6_curve_crashes_filters.append(results_dict)
D6_curve_crashes_filters_df = pd.DataFrame(D6_curve_crashes_filters)
display(D6_curve_crashes_filters_df)

index_order = ["Low Crash Frequency", "High Crash Frequency"]
column_order = ["Low AADT", "High AADT"]
display(D6_curve_crashes_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))
display(D6_curve_crashes_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))

# %% [markdown]
# **D6 Wet Road**

# %%
D6_wet_road_filters = []
for filter in rating_filters:
    results_dict, AADT_bins, crash_bins = filter_empirical_bayes_CMF(D6_wet_road, D6_curve_data, D6_wet_road_coeff, filter)
    results_dict.update({"AADT Rating" : filter[0],
                         "Crash Frequency Rating" : filter[1],
                         "AADT Bins" : AADT_bins,
                         "Crash Frequency Bins" : crash_bins}
                        )
    D6_wet_road_filters.append(results_dict)
D6_wet_road_filters_df = pd.DataFrame(D6_wet_road_filters)
display(D6_wet_road_filters_df)

index_order = ["Low Crash Frequency", "High Crash Frequency"]
column_order = ["Low AADT", "High AADT"]
display(D6_wet_road_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))
display(D6_wet_road_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))

# %% [markdown]
# ## D1 Phonolite Filtered by AADT and Crash Frequency Ratings

# %% [markdown]
# **D1 Total**

# %%
D1_total_filters = []
for filter in rating_filters:
    results_dict, AADT_bins, crash_bins = filter_empirical_bayes_CMF(D1_data, D1_curve_data, D1_total_coeff, filter)
    results_dict.update({"AADT Rating" : filter[0],
                         "Crash Frequency Rating" : filter[1],
                         "AADT Bins" : AADT_bins,
                         "Crash Frequency Bins" : crash_bins}
                        )
    D1_total_filters.append(results_dict)
D1_total_filters_df = pd.DataFrame(D1_total_filters)
display(D1_total_filters_df)

index_order = ["Low Crash Frequency", "High Crash Frequency"]
column_order = ["Low AADT", "High AADT"]
display(D1_total_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))
display(D1_total_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))

# %% [markdown]
# **D1 Single Vehicle**

# %%
D1_single_vehicle_filters = []
for filter in rating_filters:
    results_dict, AADT_bins, crash_bins = filter_empirical_bayes_CMF(D1_single_vehicle, D1_curve_data, D1_single_vehicle_coeff, filter)
    results_dict.update({"AADT Rating" : filter[0],
                         "Crash Frequency Rating" : filter[1],
                         "AADT Bins" : AADT_bins,
                         "Crash Frequency Bins" : crash_bins}
                        )
    D1_single_vehicle_filters.append(results_dict)
D1_single_vehicle_filters_df = pd.DataFrame(D1_single_vehicle_filters)
display(D1_single_vehicle_filters_df)

index_order = ["Low Crash Frequency", "High Crash Frequency"]
column_order = ["Low AADT", "High AADT"]
display(D1_single_vehicle_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))
display(D1_single_vehicle_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))

# %% [markdown]
# **D1 Curve Crashes**

# %%
D1_curve_crashes_filters = []
for filter in rating_filters:
    results_dict, AADT_bins, crash_bins = filter_empirical_bayes_CMF(D1_curve_crashes, D1_curve_data, D1_curve_crashes_coeff, filter)
    results_dict.update({"AADT Rating" : filter[0],
                         "Crash Frequency Rating" : filter[1],
                         "AADT Bins" : AADT_bins,
                         "Crash Frequency Bins" : crash_bins}
                        )
    D1_curve_crashes_filters.append(results_dict)
D1_curve_crashes_filters_df = pd.DataFrame(D1_curve_crashes_filters)
display(D1_curve_crashes_filters_df)

index_order = ["Low Crash Frequency", "High Crash Frequency"]
column_order = ["Low AADT", "High AADT"]
display(D1_curve_crashes_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))
display(D1_curve_crashes_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))

# %% [markdown]
# **D1 Wet Road**

# %%
D1_wet_road_filters = []
for filter in rating_filters:
    results_dict, AADT_bins, crash_bins = filter_empirical_bayes_CMF(D1_wet_road, D1_curve_data, D1_wet_road_coeff, filter)
    results_dict.update({"AADT Rating" : filter[0],
                         "Crash Frequency Rating" : filter[1],
                         "AADT Bins" : AADT_bins,
                         "Crash Frequency Bins" : crash_bins}
                        )
    D1_wet_road_filters.append(results_dict)
D1_wet_road_filters_df = pd.DataFrame(D1_wet_road_filters)
display(D1_wet_road_filters_df)

index_order = ["Low Crash Frequency", "High Crash Frequency"]
column_order = ["Low AADT", "High AADT"]
display(D1_wet_road_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))
display(D1_wet_road_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0))

# %% [markdown]
# ## Debug

# %%
# data=D1_data
# curve_data=D1_curve_data
# coefficients=D1_total_coeff
# years_before_treatment=4
# years_after_treatment=3

# curve_data = count_curve_crashes(data, curve_data)
# curve_data = calculate_frequencies(curve_data, years_before_treatment, years_after_treatment)
# ####################################################################################################### curve_data, AADT_bins, crash_bins = calculate_curve_ratings(curve_data)
# # Clean through curve data
# curve_AADTs = curve_data_view["Average AADT"]
# curve_crash_counts = curve_data_view["Crashes Before"]

# # Calculate the curve AADT and crash ratings, and make sure to output the bin boundaries
# AADT_ratings, AADT_bins = pd.qcut(curve_AADTs, 2, ["Low AADT", "High AADT"], retbins=True)
# crash_ratings, crash_bins = pd.cut(curve_crash_counts, np.array([0, curve_data_view["Crashes Before"].mean(), curve_data_view["Crashes Before"].max()]), labels=["Low Crash Frequency", "High Crash Frequency"], retbins=True)

# # Join the calculated ratings to CurveIDs, then join those smaller views to the dataset
# curve_AADTs = pd.merge(curve_data_view["CurveID"].to_frame(), AADT_ratings.to_frame("AADT Rating"), left_index=True, right_index=True)
# curve_crash_counts = pd.merge(curve_data_view["CurveID"].to_frame(), crash_ratings.to_frame("Crash Frequency Rating"), left_index=True, right_index=True)
# curve_data = pd.merge(curve_data, curve_AADTs, on="CurveID", how="left")
# curve_data = pd.merge(curve_data, curve_crash_counts, on="CurveID", how="left")

# %%
# display(curve_data)

# %%
