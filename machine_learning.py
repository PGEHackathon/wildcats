"""
(1) Instantiate a model
(2) Train the model to your data
(3) Make prediction with your data
(4) Model tuning (such as cross validation)
"""
import seaborn as sns
import matplotlib.pyplot as plt


def get_interest_features(x, y, z, dataframe):
    dataframe = dataframe[[x, y, z]]
    return dataframe


def create_characteristics_plot(interest_dataframe):
    sns.scatterplot(y=interest_dataframe[interest_dataframe.columns[0]],
                    x=interest_dataframe[interest_dataframe.columns[1]],
                    hue=interest_dataframe[interest_dataframe.columns[2]])
    plt.title(f"{interest_dataframe.columns[0]} over {interest_dataframe.columns[1]}");

