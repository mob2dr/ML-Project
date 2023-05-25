import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def read_data_from_file(path, delimiter):
    """
    Read data from a file using pandas.

    Args:
        path (str): The file path.
        delimiter (str): The delimiter used in the file.

    Returns:
        pandas.DataFrame or None: The loaded data as a DataFrame if successful, or None if an error occurs.

    Raises:
        FileNotFoundError: If the file is not found.
        pd.errors.ParserError: If there is an error parsing the file.

    """
    try:
        df = pd.read_csv(path, delimiter=delimiter)
        return df
    except FileNotFoundError:
        print("File not found!")
    except pd.errors.ParserError:
        print("Error parsing the file. Please check the delimiter.")

def select_categorical_columns(data):
    """
    Select categorical columns from a DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame.

    Returns:
        list: A list of column names containing categorical variables.

    """
    categorical_columns = data.select_dtypes(include='object').columns.tolist()
    return categorical_columns

def select_numerical_columns(data):
    """
    Select numerical columns from a DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame.

    Returns:
        list: A list of column names containing numerical variables.

    """
    numerical_columns = data.select_dtypes(include='number').columns.tolist()
    return numerical_columns

def encode_data(data, columns, encoding_type):
    """
    Encodes the specified columns in the data using either one-hot encoding or label encoding.
    
    Args:
        data (pandas.DataFrame): The input data.
        columns (list): The columns to be encoded.
        encoding_type (str): The type of encoding to be performed. Can be either 'one-hot' or 'label'.
    
    Returns:
        encoded_data (pandas.DataFrame): The encoded data.
    """
    encoded_data = data.copy()
    
    for column in columns:
        if encoding_type == 'one-hot':
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_column = encoder.fit_transform(encoded_data[[column]])
            encoded_column = pd.DataFrame(encoded_column, columns=encoder.get_feature_names_out([column]))
            encoded_data = pd.concat([encoded_data, encoded_column], axis=1)
            encoded_data.drop(column, axis=1, inplace=True)
        elif encoding_type == 'label':
            encoder = LabelEncoder()
            encoded_data[column] = encoder.fit_transform(encoded_data[column])
        else:
            raise ValueError("Invalid encoding_type. Supported values are 'one-hot' and 'label'.")
    
    return encoded_data

def delete_columns(data, columns):
    """
    Deletes the specified columns from the data.
    
    Args:
        data (pandas.DataFrame): The input data.
        columns (list): The columns to be deleted.
    
    Returns:
        modified_data (pandas.DataFrame): The data with specified columns deleted.
    """
    modified_data = data.drop(columns, axis=1)
    return modified_data

def visulization_with_target(df, column, target):
    """
    Perform visualization of a categorical column with respect to a target variable.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The name of the categorical column to visualize.
        target (str): The name of the target variable.

    Returns:
        None

    """
    print(column)
    print(df[column].value_counts(normalize=True))
    sns.countplot(data=df, x=column, hue=target)
    plt.xticks(rotation=45)
    plt.show()

def visulization_num_with_target(df, column):
    print(column)
    print('skwed',df[column].skew())
    sns.boxplot(df[column])
    plt.show()

def Rel_chi_square(df, column, target):
    """
    Perform the chi-square test of independence between a categorical column and a target variable.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The name of the categorical column.
        target (str): The name of the target variable.

    Returns:
        None

    """
    # create a contingency table
    print(column)
    contingency_table = pd.crosstab(df[column], df[target])

    # perform chi-square test
    stat, p, dof, expected = chi2_contingency(contingency_table)

    # print the results
    print('Chi-square statistic:', stat)
    print('p-value:', p)


def apply_log_transformation(df, columns):
    """
    Apply logarithmic transformation to specified columns in a DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        columns (list): A list of column names to be transformed.

    Returns:
        pandas.DataFrame: A new DataFrame with specified columns logarithmically transformed.

    Example:
        data = {'Column1': [1, 2, 3, 4, 5],
                'Column2': [10, 20, 30, 40, 50]}
        df = pd.DataFrame(data)
        transformed_data = apply_log_transformation(df, ['Column1'])
        print(transformed_data)

    """
    # Copy the DataFrame to avoid modifying the original data
    transformed_df = df.copy()

    # Apply logarithmic transformation to the specified columns
    for column in columns:
        transformed_df[column] = np.log(transformed_df[column]+1)

    return transformed_df

def handle_outliers(df, columns, factor=1.5):
    """
    Handle outliers in specific columns of a DataFrame using higher and lower bounds based on the 5-number summary.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        columns (list): A list of column names to handle outliers.
        factor (float): The factor to multiply the interquartile range (IQR) for determining the bounds.
                        By default, it is set to 1.5.

    Returns:
        pandas.DataFrame: The DataFrame with outliers replaced by the corresponding higher or lower bound.

    Example:
        data = {'Column1': [10, 15, 20, 25, 200],
                'Column2': [5, 8, 12, 18, 120]}
        df = pd.DataFrame(data)
        processed_df = handle_outliers(df, ['Column1'])
        print(processed_df)

    """
    processed_df = df.copy()

    for column in columns:
        data = processed_df[column].copy()
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        processed_df[column] = np.where((data < lower_bound) | (data > upper_bound),
                                        np.clip(data, lower_bound, upper_bound),
                                        data)

    return processed_df

def calculate_relation(data, target_column, col):
    """
    Calculate the relation between a categorical target variable and a numerical column.

    Parameters:
        data (str): DataFrame
        target_column (str): The name of the categorical target variable column.
        col (str): The name of the numerical column.

    Returns:
        DataFrame: A DataFrame containing the relation between the target variable and the numerical column.
    """    

    # Group the data by the target variable and calculate summary statistics for the numerical column
    relation = data.groupby(target_column)[col].describe()

    return relation

def calculate_missing_ratio(dataframe):
    """
    Calculate the ratio of missing values and the data type of every column in the given DataFrame.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing two columns - 'Missing Ratio' and 'Data Type'.
            The 'Missing Ratio' column represents the ratio of missing values for each column,
            and the 'Data Type' column represents the data type of each column.
            The index of the DataFrame represents the column names.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, None, 4],
        ...                    'B': [None, 2, 3, None],
        ...                    'C': [None, None, None, None]})
        >>> missing_info = calculate_missing_ratio(df)
        >>> print(missing_info)
           Missing Ratio Data Type
        A           0.25     int64
        B           0.50   float64
        C           1.00   float64
    """
    missing_ratio = dataframe.isnull().sum() / len(dataframe)
    data_types = dataframe.dtypes
    missing_info = pd.DataFrame({'Missing Ratio': missing_ratio, 'Data Type': data_types})
    return missing_info




