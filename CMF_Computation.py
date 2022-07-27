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
# **Calculating ratings**

# %%
def calculate_curve_ratings(data: pd.DataFrame, curve_data: pd.DataFrame):
    # Helper function to calculate curve AADT and crash frequency ratings
    
    # Finding crashes before treatment
    crash_before_data = data.loc[data["Relation To Treatment"] == "before treatment"]
    crash_before_counts = crash_before_data.groupby("CurveID")["CurveID"].count().to_frame("Crashes Before")

    # Making the crash ratings
    crash_ratings = pd.cut(crash_before_counts["Crashes Before"], np.array([0, 3, crash_before_counts["Crashes Before"].max()]), labels=["Low Crash Frequency", "High Crash Frequency"], include_lowest=True).to_frame("Crash Frequency Rating")

    # Finding average AADTs
    curve_AADTs = curve_data[["CurveID","Average AADT Before"]].set_index("CurveID")

    # Making the AADT ratings
    AADT_ratings = pd.cut(curve_AADTs["Average AADT Before"], np.array([0, 2000, curve_data["Average AADT Before"].max()]), labels=["Low AADT", "High AADT"], include_lowest=True).to_frame("AADT Rating")
    
    # Join the calculated ratings to CurveIDs, then join those smaller views to the dataset
    curve_data = pd.merge(curve_data, crash_ratings, on="CurveID", how="left")
    curve_data.loc[:, "Crash Frequency Rating"] = curve_data.loc[:, "Crash Frequency Rating"].fillna("Low Crash Frequency")
    curve_data = pd.merge(curve_data, AADT_ratings, on="CurveID", how="left")
    
    try:
        # Finding the instersection crashes before treatment
        intersection_crash_before_data = crash_before_data.loc[data["Intersection related"] == 1]
        intersection_crash_before_counts = intersection_crash_before_data.groupby("CurveID")["CurveID"].count().to_frame("Intersection Crashes Before")

        # Making the intersection crash ratings
        intersection_crash_ratings = pd.cut(intersection_crash_before_counts["Intersection Crashes Before"], np.array([0, 0.5, intersection_crash_before_counts["Intersection Crashes Before"].max()]), labels=["Low Intersection Crash Frequency", "High Intersection Crash Frequency"], include_lowest=True).to_frame("Intersection Crash Frequency Rating")

        # Adding to curve data
        curve_data = pd.merge(curve_data, intersection_crash_ratings, on="CurveID", how="left")
        curve_data.loc[:, "Intersection Crash Frequency Rating"] = curve_data.loc[:, "Intersection Crash Frequency Rating"].fillna("Low Intersection Crash Frequency")
    except:
        pass

    return data, curve_data


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
    curve_data = calculate_SPF_frequencies(curve_data, coefficients)
    cumulative_dict = calculate_cumulative_values(curve_data, coefficients)
    results_dict = calculate_final_outputs(curve_data, cumulative_dict)
    return results_dict


# %% [markdown]
# **Empirical Bayes Filter by AADT and Crash Frequency Rating**

# %%
def filter_by_rating(data: pd.DataFrame, curve_data: pd.DataFrame, rating_filters):
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

def filter_calculate_cumulative_values(curve_data: pd.DataFrame, coefficients: pd.DataFrame, rating_filters):
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

def filter_calculate_final_outputs(curve_data: pd.DataFrame, cumulative_dict: dict, rating_filters):
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

def filter_empirical_bayes_CMF(data: pd.DataFrame, curve_data: pd.DataFrame, coefficients: pd.DataFrame, rating_filters, years_before_treatment=4, years_after_treatment=3):
    curve_data = count_curve_crashes(data, curve_data)
    curve_data = calculate_frequencies(curve_data, years_before_treatment, years_after_treatment)
    data, curve_data = filter_by_rating(data, curve_data, rating_filters)
    curve_data = calculate_SPF_frequencies(curve_data, coefficients)
    cumulative_dict = filter_calculate_cumulative_values(curve_data, coefficients, rating_filters)
    results_dict = filter_calculate_final_outputs(curve_data, cumulative_dict, rating_filters)
    return results_dict


# %%
def int_filter_by_rating(data: pd.DataFrame, curve_data: pd.DataFrame, rating_filters: tuple):
    # Helper function to filter crash and curve data by the AADT ratings and crash frequency ratings

    # Clarifying the filters by making separate variables
    AADT_rating_filter = rating_filters[0]
    crash_rating_filter = rating_filters[1]
    intersection_crash_rating_filter = rating_filters[2]

    # Joining the ratings to the crash data
    data = data.join(curve_data[["CurveID", "AADT Rating", "Crash Frequency Rating", "Intersection Crash Frequency Rating"]].set_index("CurveID"), on="CurveID")

    # Filtering the data and curve data based on the ratings
    data = data.query("`AADT Rating` == @AADT_rating_filter & `Crash Frequency Rating` == @crash_rating_filter & `Intersection Crash Frequency Rating` == @intersection_crash_rating_filter")
    curve_data = curve_data.query("`AADT Rating` == @AADT_rating_filter & `Crash Frequency Rating` == @crash_rating_filter & `Intersection Crash Frequency Rating` == @intersection_crash_rating_filter")
    
    return data, curve_data

def int_filter_calculate_cumulative_values(curve_data: pd.DataFrame, coefficients: pd.DataFrame, rating_filters: tuple):
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
        print("The filters of " + rating_filters[0] + ", " + rating_filters[1] + ", and " + rating_filters[2] + " either has no curves associated with it, and the value of the comparison ratio is NaN.")
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

def int_filter_calculate_final_outputs(curve_data: pd.DataFrame, cumulative_dict: dict, rating_filters: tuple):
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
        print("The filters of " + rating_filters[0] + " and " + rating_filters[1] + ", and" + rating_filters[2] + " has no crashes after treatment, and so the EB CMF defaults to 0 and the Stdev cannot be calculated.")

    # Collect all calculated values into a dictionary
    results_dict = {"Variance of Expected After" : variance_of_expected_after,
                    "Empirical Bayes CMF" : empirical_bayes_cmf,
                    "Variance of Total Frequency After" : variance_of_total_frequency_after,
                    "CMF Standard Deviation" : standard_deviation
                   }

    return results_dict

def int_filter_empirical_bayes_CMF(data: pd.DataFrame, curve_data: pd.DataFrame, coefficients: pd.DataFrame, rating_filters: tuple, years_before_treatment=4, years_after_treatment=3):
    curve_data = count_curve_crashes(data, curve_data)
    curve_data = calculate_frequencies(curve_data, years_before_treatment, years_after_treatment)
    data, curve_data = int_filter_by_rating(data, curve_data, rating_filters)
    curve_data = calculate_SPF_frequencies(curve_data, coefficients)
    cumulative_dict = int_filter_calculate_cumulative_values(curve_data, coefficients, rating_filters)
    results_dict = int_filter_calculate_final_outputs(curve_data, cumulative_dict, rating_filters)
    return results_dict


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

# Calculate AADT and crash frequency before ratings
D1_data, D1_curve_data = calculate_curve_ratings(D1_data, D1_curve_data)
D6_data, D6_curve_data = calculate_curve_ratings(D6_data, D6_curve_data)

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
D6_int = D6_data[D6_data["Intersection related"] == 1]

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

# %%
# District 1 Naive CMFs
D1_total_naive = naive_CMF(D1_data)
D1_single_vehicle_naive = naive_CMF(D1_single_vehicle)
D1_curve_crashes_naive = naive_CMF(D1_curve_crashes)
D1_wet_road_naive = naive_CMF(D1_wet_road)

# %%
# District 2 Naive CMFs
D2_total_naive = naive_CMF(D2_data)
D2_single_vehicle_naive = naive_CMF(D2_single_vehicle)
D2_curve_crashes_naive = naive_CMF(D2_curve_crashes)
D2_wet_road_naive = naive_CMF(D2_wet_road)

# %%
# District 6 Naive CMFs
D6_total_naive = naive_CMF(D6_data)
D6_single_vehicle_naive = naive_CMF(D6_single_vehicle)
D6_curve_crashes_naive = naive_CMF(D6_curve_crashes)
D6_wet_road_naive = naive_CMF(D6_wet_road)

# %%
naive_list = [D1_total_naive, D1_single_vehicle_naive, D1_curve_crashes_naive, D1_wet_road_naive,
              D2_total_naive, D2_single_vehicle_naive, D2_curve_crashes_naive, D2_wet_road_naive,
              D6_total_naive, D6_single_vehicle_naive, D6_curve_crashes_naive, D6_wet_road_naive]
district_list = ["District 1 Phonolite", "District 1 Phonolite", "District 1 Phonolite", "District 1 Phonolite",
                 "District 2 LWA", "District 2 LWA", "District 2 LWA", "District 2 LWA",
                 "District 6 HFST", "District 6 HFST", "District 6 HFST", "District 6 HFST"]
filter_list = ["All Crashes", "Single Vehicle Crashes", "Curve Crashes", "Wet Road Crashes",
               "All Crashes", "Single Vehicle Crashes", "Curve Crashes", "Wet Road Crashes",
               "All Crashes", "Single Vehicle Crashes", "Curve Crashes", "Wet Road Crashes"]
naive_df = pd.DataFrame(naive_list, index=[district_list, filter_list], columns=["Naive Bayes CMF"])
naive_df

# %% [markdown]
# ## Empirical CMFs

# %%
# District 1 EB CMFs
D1_total_EB = empirical_bayes_CMF(D1_data, D1_curve_data, D1_total_coeff)
D1_single_vehicle_EB = empirical_bayes_CMF(D1_single_vehicle, D1_curve_data, D1_single_vehicle_coeff)
D1_curve_crashes_EB = empirical_bayes_CMF(D1_curve_crashes, D1_curve_data, D1_curve_crashes_coeff)
D1_wet_road_EB = empirical_bayes_CMF(D1_wet_road, D1_curve_data, D1_wet_road_coeff)

# %%
# District 6 EB CMFs
D6_total_EB = empirical_bayes_CMF(D6_data, D6_curve_data, D6_total_coeff)
D6_single_vehicle_EB = empirical_bayes_CMF(D6_single_vehicle, D6_curve_data, D6_single_vehicle_coeff)
D6_curve_crashes_EB = empirical_bayes_CMF(D6_curve_crashes, D6_curve_data, D6_curve_crashes_coeff)
D6_wet_road_EB = empirical_bayes_CMF(D6_wet_road, D6_curve_data, D6_wet_road_coeff)
D6_int_EB = empirical_bayes_CMF(D6_int, D6_curve_data, D6_total_coeff)

# %%
EB_data = [(D1_total_EB["Empirical Bayes CMF"], D1_total_EB["CMF Standard Deviation"]),
           (D1_single_vehicle_EB["Empirical Bayes CMF"], D1_single_vehicle_EB["CMF Standard Deviation"]),
           (D1_curve_crashes_EB["Empirical Bayes CMF"], D1_curve_crashes_EB["CMF Standard Deviation"]),
           (D1_wet_road_EB["Empirical Bayes CMF"], D1_wet_road_EB["CMF Standard Deviation"]),
           (D6_total_EB["Empirical Bayes CMF"], D6_total_EB["CMF Standard Deviation"]),
           (D6_single_vehicle_EB["Empirical Bayes CMF"], D6_single_vehicle_EB["CMF Standard Deviation"]),
           (D6_curve_crashes_EB["Empirical Bayes CMF"], D6_curve_crashes_EB["CMF Standard Deviation"]),
           (D6_wet_road_EB["Empirical Bayes CMF"], D6_wet_road_EB["CMF Standard Deviation"]),
           (D6_int_EB["Empirical Bayes CMF"], D6_int_EB["CMF Standard Deviation"])
          ]
district_list_EB = ["District 1 Phonolite", "District 1 Phonolite", "District 1 Phonolite", "District 1 Phonolite",
                    "District 6 HFST", "District 6 HFST", "District 6 HFST", "District 6 HFST", "District 6 HFST"]
filter_list_EB = ["All Crashes", "Single Vehicle Crashes", "Curve Crashes", "Wet Road Crashes",
                  "All Crashes", "Single Vehicle Crashes", "Curve Crashes", "Wet Road Crashes", "Intersection Crashes"]
EB_df = pd.DataFrame(EB_data, index=[district_list_EB, filter_list_EB], columns=["Empirical Bayes CMF", "CMF Standard Deviation"])
EB_df

# %% [markdown]
# ## D6 HFST Filtered by AADT and Crash Frequency Ratings

# %%
# Test to see if 3 crashes/year is a proper cutoff for prior crash frequency rating
D6_data.groupby("CurveID")["CurveID"].count().describe()

# %%
filters = [("Low AADT", "Low Crash Frequency"),
           ("High AADT", "Low Crash Frequency"),
           ("Low AADT", "High Crash Frequency"),
           ("High AADT", "High Crash Frequency"),
          ]

# %%
D6_total_filters = []
for filter in filters:
    results_dict = filter_empirical_bayes_CMF(D6_data, D6_curve_data, D6_total_coeff, filter)
    results_dict.update({"AADT Rating" : filter[0],
                         "Crash Frequency Rating" : filter[1]}
                        )
    D6_total_filters.append(results_dict)
D6_total_filters_df = pd.DataFrame(D6_total_filters)

index_order = ["Low Crash Frequency", "High Crash Frequency"]
column_order = ["Low AADT", "High AADT"]

D6_total_filters_CMF_table = D6_total_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)
D6_total_filters_STD_table = D6_total_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)

display(D6_total_filters_CMF_table.style.set_caption("D6 HFST Total Crashes CMFs").set_table_styles([{'selector': 'caption',
                                                                                                    'props': [('color', 'cyan'), ('font-size', '20px')]
                                                                                                   }]))
display(D6_total_filters_STD_table.style.set_caption("D6 HFST Total Crashes CMF Standard Deviations").set_table_styles([{'selector': 'caption',
                                                                                                                       'props': [('color', 'cyan'), ('font-size', '20px')]
                                                                                                                      }]))

# %% [markdown]
# ## D6 Split by AADT, crash frequency, and intersection crash frequency

# %%
D6_curve_data.groupby("Intersection Crash Frequency Rating")["Intersection Crash Frequency Rating"].count()

# %%
filters = [("Low AADT", "Low Crash Frequency", "Low Intersection Crash Frequency"),
          ("High AADT", "Low Crash Frequency", "Low Intersection Crash Frequency"),
          ("Low AADT", "High Crash Frequency", "Low Intersection Crash Frequency"),
          ("High AADT", "High Crash Frequency", "Low Intersection Crash Frequency"),
          ("Low AADT", "Low Crash Frequency", "High Intersection Crash Frequency"),
          ("High AADT", "Low Crash Frequency", "High Intersection Crash Frequency"),
          ("Low AADT", "High Crash Frequency", "High Intersection Crash Frequency"),
          ("High AADT", "High Crash Frequency", "High Intersection Crash Frequency"),
         ]

# %%
D6_int_total_filters = []
for filter in filters:
    results_dict = int_filter_empirical_bayes_CMF(D6_data, D6_curve_data, D6_total_coeff, filter)
    results_dict.update({"AADT Rating" : filter[0],
                         "Crash Frequency Rating" : filter[1],
                         "Intersection Crash Frequency Rating" : filter[2]
                        })
    D6_int_total_filters.append(results_dict)
D6_int_total_filters_df = pd.DataFrame(D6_int_total_filters)
D6_int_total_filters_df

# %% [markdown]
# ## Debug

# %%
# data = D6_data
# curve_data = D6_curve_data
# coefficients = D6_total_coeff
# years_before_treatment = 4
# years_after_treatment = 3
# rating_filters = filters[3]

# # curve_data = count_curve_crashes(data, curve_data)
# # curve_data = calculate_frequencies(curve_data, years_before_treatment, years_after_treatment)
# # curve_data
# # data, curve_data = int_filter_by_rating(data, curve_data, rating_filters)
# # curve_data = calculate_SPF_frequencies(curve_data, coefficients)
# # cumulative_dict = int_filter_calculate_cumulative_values(curve_data, coefficients, rating_filters)
# # results_dict = int_filter_calculate_final_outputs(curve_data, cumulative_dict, rating_filters)

# results_dict = int_filter_empirical_bayes_CMF(data, curve_data, coefficients, rating_filters, years_before_treatment=4, years_after_treatment=3)
# results_dict

# %% [markdown]
# ### Results further below aren't used

# %%
# D6_single_vehicle_filters = []
# for filter in rating_filters:
#     results_dict = filter_empirical_bayes_CMF(D6_single_vehicle, D6_curve_data, D6_single_vehicle_coeff, filter)
#     results_dict.update({"AADT Rating" : filter[0],
#                          "Crash Frequency Rating" : filter[1]}
#                         )
#     D6_single_vehicle_filters.append(results_dict)
# D6_single_vehicle_filters_df = pd.DataFrame(D6_single_vehicle_filters)

# index_order = ["Low Crash Frequency", "High Crash Frequency"]
# column_order = ["Low AADT", "High AADT"]

# D6_single_vehicle_filters_CMF_table = D6_single_vehicle_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)
# D6_single_vehicle_filters_STD_table = D6_single_vehicle_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)

# display(D6_single_vehicle_filters_CMF_table.style.set_caption("D6 HFST Single Vehicle Crashes CMFs").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))
# display(D6_single_vehicle_filters_STD_table.style.set_caption("D6 HFST Single Vehicle Crashes CMF Standard Deviations").set_table_styles([{'selector': 'caption',
#                                                                                                                                 'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                                                }]))

# %%
# D6_curve_crashes_filters = []
# for filter in rating_filters:
#     results_dict = filter_empirical_bayes_CMF(D6_curve_crashes, D6_curve_data, D6_curve_crashes_coeff, filter)
#     results_dict.update({"AADT Rating" : filter[0],
#                          "Crash Frequency Rating" : filter[1]}
#                         )
#     D6_curve_crashes_filters.append(results_dict)
# D6_curve_crashes_filters_df = pd.DataFrame(D6_curve_crashes_filters)

# index_order = ["Low Crash Frequency", "High Crash Frequency"]
# column_order = ["Low AADT", "High AADT"]

# D6_curve_crashes_filters_CMF_table = D6_curve_crashes_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)
# D6_curve_crashes_filters_STD_table = D6_curve_crashes_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)

# display(D6_curve_crashes_filters_CMF_table.style.set_caption("D6 HFST Curve Crashes CMFs").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))
# display(D6_curve_crashes_filters_STD_table.style.set_caption("D6 HFST Curve Crashes CMF Standard Deviations").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))

# %%
# D6_wet_road_filters = []
# for filter in rating_filters:
#     results_dict = filter_empirical_bayes_CMF(D6_wet_road, D6_curve_data, D6_wet_road_coeff, filter)
#     results_dict.update({"AADT Rating" : filter[0],
#                          "Crash Frequency Rating" : filter[1]}
#                         )
#     D6_wet_road_filters.append(results_dict)
# D6_wet_road_filters_df = pd.DataFrame(D6_wet_road_filters)

# index_order = ["Low Crash Frequency", "High Crash Frequency"]
# column_order = ["Low AADT", "High AADT"]

# D6_wet_road_filters_CMF_table = D6_wet_road_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)
# D6_wet_road_filters_STD_table = D6_wet_road_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)

# display(D6_wet_road_filters_CMF_table.style.set_caption("D6 HFST Wet Road Crashes CMFs").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))
# display(D6_wet_road_filters_STD_table.style.set_caption("D6 HFST Wet Road Crashes CMF Standard Deviations").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))

# %% [markdown]
# ## D1 Phonolite Filtered by AADT and Crash Frequency Ratings

# %%
# D1_total_filters = []
# for filter in rating_filters:
#     results_dict = filter_empirical_bayes_CMF(D1_data, D1_curve_data, D1_total_coeff, filter)
#     results_dict.update({"AADT Rating" : filter[0],
#                          "Crash Frequency Rating" : filter[1]}
#                         )
#     D1_total_filters.append(results_dict)
# D1_total_filters_df = pd.DataFrame(D1_total_filters)

# index_order = ["Low Crash Frequency", "High Crash Frequency"]
# column_order = ["Low AADT", "High AADT"]

# D1_total_filters_CMF_table = D1_total_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)
# D1_total_filters_STD_table = D1_total_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)

# display(D1_total_filters_CMF_table.style.set_caption("D1 Phonolite Total Crashes CMFs").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))
# display(D1_total_filters_STD_table.style.set_caption("D1 Phonolite Total Crashes CMF Standard Deviations").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))

# %%
# D1_single_vehicle_filters = []
# for filter in rating_filters:
#     results_dict = filter_empirical_bayes_CMF(D1_single_vehicle, D1_curve_data, D1_single_vehicle_coeff, filter)
#     results_dict.update({"AADT Rating" : filter[0],
#                          "Crash Frequency Rating" : filter[1]}
#                         )
#     D1_single_vehicle_filters.append(results_dict)
# D1_single_vehicle_filters_df = pd.DataFrame(D1_single_vehicle_filters)

# index_order = ["Low Crash Frequency", "High Crash Frequency"]
# column_order = ["Low AADT", "High AADT"]

# D1_single_vehicle_filters_CMF_table = D1_single_vehicle_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)
# D1_single_vehicle_filters_STD_table = D1_single_vehicle_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)

# display(D1_single_vehicle_filters_CMF_table.style.set_caption("D1 Phonolite Single Vehicle Crashes CMFs").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))
# display(D1_single_vehicle_filters_STD_table.style.set_caption("D1 Phonolite Single Vehicle Crashes CMF Standard Deviations").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))

# %%
# D1_curve_crashes_filters = []
# for filter in rating_filters:
#     results_dict = filter_empirical_bayes_CMF(D1_curve_crashes, D1_curve_data, D1_curve_crashes_coeff, filter)
#     results_dict.update({"AADT Rating" : filter[0],
#                          "Crash Frequency Rating" : filter[1]}
#                         )
#     D1_curve_crashes_filters.append(results_dict)
# D1_curve_crashes_filters_df = pd.DataFrame(D1_curve_crashes_filters)

# index_order = ["Low Crash Frequency", "High Crash Frequency"]
# column_order = ["Low AADT", "High AADT"]

# D1_curve_crashes_filters_CMF_table = D1_curve_crashes_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)
# D1_curve_crashes_filters_STD_table = D1_curve_crashes_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)

# display(D1_curve_crashes_filters_CMF_table.style.set_caption("D1 Phonolite Curve Crashes CMFs").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))
# display(D1_curve_crashes_filters_STD_table.style.set_caption("D1 Phonolite Curve Crashes CMF Standard Deviations").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))

# %%
# D1_wet_road_filters = []
# for filter in rating_filters:
#     results_dict = filter_empirical_bayes_CMF(D1_wet_road, D1_curve_data, D1_wet_road_coeff, filter)
#     results_dict.update({"AADT Rating" : filter[0],
#                          "Crash Frequency Rating" : filter[1]}
#                         )
#     D1_wet_road_filters.append(results_dict)
# D1_wet_road_filters_df = pd.DataFrame(D1_wet_road_filters)

# index_order = ["Low Crash Frequency", "High Crash Frequency"]
# column_order = ["Low AADT", "High AADT"]

# D1_wet_road_filters_CMF_table = D1_wet_road_filters_df.pivot_table("Empirical Bayes CMF", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)
# D1_wet_road_filters_STD_table = D1_wet_road_filters_df.pivot_table("CMF Standard Deviation", index="Crash Frequency Rating", columns="AADT Rating").reindex(column_order, axis=1).reindex(index_order, axis=0)

# display(D1_wet_road_filters_CMF_table.style.set_caption("D1 Phonolite Wet Road Crashes CMFs").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))
# display(D1_wet_road_filters_STD_table.style.set_caption("D1 Phonolite Wet Road Crashes CMF Standard Deviations").set_table_styles([{'selector': 'caption',
#                                                                                                              'props': [('color', 'cyan'), ('font-size', '20px')]
#                                                                                                             }]))
