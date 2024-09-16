import os
import re
import logging
import pandas as pd
import numpy as np
import requests

def concatenate_dataframes(df_list: list) -> pd.DataFrame:
    """
    Concatenates a list of DataFrames into a single DataFrame.

    This function combines multiple DataFrames into one, creating a single DataFrame that contains 
    all the rows from the individual DataFrames. If the list is empty, it returns an empty DataFrame.

    Parameters:
        df_list (list): List of DataFrames to concatenate. Can be empty if no DataFrames are provided.

    Returns:
        DataFrame: The concatenated DataFrame or an empty DataFrame if the list is empty. In case of 
                   any error during concatenation, logs the error and returns an empty DataFrame.
    """
    try:
        if df_list:
            return pd.concat(df_list, ignore_index=True)
        else:
            logging.warning("No dataframes found in list, returning empty dataframe...")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error concatenating DataFrames: {e}")
        return pd.DataFrame()

def ingest_police_data(base_dir: str, police_forces: list, dataset: str = 'street') -> pd.DataFrame:
    """
    Ingests DataFrames from CSV files located in the specified directory structure.

    This function reads all CSV files matching the specified dataset and police forces from 
    directories named in the format YYYY-MM within the base directory. It collects DataFrames 
    from these files and adds metadata column for police force.

    Parameters:
        base_dir (str): The base directory where the data is stored. 
                        Expected to contain subdirectories named YYYY-MM.
        police_forces (list): List of police forces to investigate. Each force will be used 
                              to locate specific CSV files.
        dataset (str): Type of dataset from police data. Default is 'street'.

    Returns:
        pd.DataFrame: Each DataFrame corresponds to a CSV file that was read. If no files 
                            are found, returns an empty list.
    """
    logging.info(f"Starting police data ingestion from {base_dir}")

    police_df_lst = []

    # List all subdirectories in the base directory
    directories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    directories.sort()

    if not directories:
        logging.warning(f"No valid directories found in {base_dir}")
        return concatenate_dataframes(police_df_lst) # return empty list

    # Process each directory
    for directory in directories:
        try:
            # Check if directory follows the format 'YYYY-MM'
            if not re.match(r"^\d{4}-\d{2}$", directory):
                logging.warning(f"Directory name '{directory}' is not in the format YYYY-MM, skipping...")
                continue # Ignore this directory and move on to the next that follows the correct format

            # Extract year and month from the directory name
            year, month_str = directory.split('-')

            current_dir = os.path.join(base_dir, directory)

            # Process each police force
            for force in police_forces:

                file_name = f"{year}-{month_str}-{force}-{dataset}.csv" # Define the file name
                file_path = os.path.join(current_dir, file_name) # update the file path

                if os.path.exists(file_path):

                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(file_path, header = True)

                    # Add column with metadata
                    df['Police force'] = force

                    # append list of dataframes with imported df
                    police_df_lst.append(df) 

                else:
                    # if path does not exist, warning
                    logging.warning(f"File not found: {file_path}")

        except ValueError as e:
            # Raise error regarding directory name
            logging.error(f"Unexpected error processing directory {directory}: {e}")

    # return list of police dataframes, ready to be concatenated
    return concatenate_dataframes(police_df_lst)

def ingest_data(path, header=[]):
    """
    Imports a CSV file or all CSV files in a given directory and concatenates them into a single DataFrame.
    
    Parameters:
        path (str): The file path or directory path where the CSV files are located.
        header (list): List of headers. If empty, the default headers from the file will be used.
    
    Returns:
        pd.DataFrame: A single pandas DataFrame representing all files called
    """
    
    # Get the file or folder name for logging
    file_folder_name = os.path.basename(os.path.normpath(path))
    logging.info(f"Starting {' '.join(file_folder_name.split('-'))} ingestion from {path}")

    if os.path.isfile(path) and path.endswith('.csv'):
        csv_files = [path]  # Single CSV file case
    elif os.path.isdir(path):
        csv_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]  # Directory case
    else:
        logging.error(f"Invalid path: {path} is neither a CSV file nor a directory.")
        return pd.DataFrame()  # Return empty DataFrame for invalid path

    if not csv_files:
        logging.warning(f"No CSV files found in the directory: {path}")
        return pd.DataFrame()  # Return empty DataFrame if no CSV files found

    # Initialize an empty list to hold DataFrames
    dataframes = []
    
    # Loop through each CSV file and load it into a DataFrame
    for file in csv_files:
        try:
            # Load the CSV file into a DataFrame
            if header:
                df = pd.read_csv(file, header=None)
                df.columns = header
            else:
                df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            logging.error(f"Error reading {file}: {str(e)}")

    return concatenate_dataframes(dataframes)

def del_cols(df: pd.DataFrame, cols:list):
    """
    Delete unnecessary columns from the DataFrame.

    Parameters:
        df (pd.DataFrame): Dataframe to delete columns from
        cols (list): List of columns to delete from dataframe

    Returns:
        df (pd.DataFrame): New version of dataframe with deleted columns
    """
    
    # drop columns
    df.drop(columns=cols, inplace=True)

    # Return new df
    return df

def del_na(df: pd.DataFrame, cols:list = []):
    """
    Delete rows with NA values from specified column

    Parameters:
        df (pd.DataFrame): Dataframe to delete NA rows from
        cols (list): List of columns that have na values for deletion

    Returns:
        clean_df (pd.DataFrame): New cleaner dataframe with rows with NA's removed 
    """

    if not cols:
        # drop any rows with na across the dataset
        clean_df = df.dropna()
    else:
        # drop columns with na, subsetted by defined columns
        clean_df = df.dropna(subset = cols)

    # return new df
    return clean_df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deletes duplicate rows from dataset

    Parameters:
        df (pd.DataFrame): Dataframe to remove duplicates from
        cols (list): Columns to subset duplicate deletions

    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed.
    """

    # Remove duplicate rows, keeping only the first occurrence
    df = df.drop_duplicates(keep='first')

    return df

def reformat_date(df: pd.DataFrame, date_column: str):
    """
    Extracts month and year from a specified date column and adds them as new columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the date column.
        date_column (str): The name of the column with date information.
        date_format (str): The format of the date in the date_column (optional).
            If None, 'YYYY-MM-DD HH:MM' will be used as the default format.

    Returns:
        df (pd.DataFrame): The DataFrame with additional 'Month' and 'Year' columns.
    """
    try:
        # Convert the date column to datetime with the specified format
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # Log warning if there are any NaT values after conversion
        if df[date_column].isna().any():
            logging.warning(f"Some dates in column '{date_column}' could not be parsed and were set to NaT.")

        # Extract month and year from the datetime column
        df[date_column] = df[date_column].dt.strftime('%Y-%m')

    except Exception as e:
        logging.error(f"Error reformatting date in column '{date_column}': {e}")

    return df

def export_data(df: pd.DataFrame, base_dir: str):
    """
    Export pandas dataframe as a csv in working directory

    Parameters:
        df (pd.DataFrame): DataFrame to be exported
        base_dir (str): Path to where csv file will be saved

    Returns:
        None
    """
    try:
        # Export dataframe to csv given directory
        df.to_csv(base_dir, index = False)
    except Exception as e:
        logging.error(f"Error exporting data to file {base_dir}: {e}")

def get_postcode(lon, lat):
    """
    Retrieving postcodes based on longitude and latitudes from Postcodes.io API 

    Parameters:
        long: Longitude number
        lat: Latitude number
        
    Returns:
        str: Postcode
    """
    url = f"https://api.postcodes.io/postcodes?lon={lon}&lat={lat}"

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            return data['result'][0]['postcode']
        elif response.status_code in [404, 400, 500]:
            logging.error(f"Error retrieving postcode from postcode.io: Error {response.status_code}")
            return "unknown"
        else: return "unknown"
    except Exception as e:
        logging.error(f"Error retrieving postcode from postcode.io: {e}")
        return "unknown1"

def add_postcodes(df: pd.DataFrame, latitude_col: str = 'Latitude', longitude_col: str = 'Longitude'):
    """
    Add postcodes to a DataFrame based on latitude and longitude columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing latitude and longitude columns.
        latitude_col (str): The name of the latitude column (default is 'Latitude').
        longitude_col (str): The name of the longitude column (default is 'Longitude').

    Returns:
        pd.DataFrame: The original DataFrame with a new 'postcode' column added.
    """
    try:
        # Apply the get_postcode function to each row, using dynamic column names
        df['Postcode'] = df.apply(lambda row: get_postcode(row[latitude_col], row[longitude_col]), axis=1)
    except Exception as e:
        logging.error(f"Error adding postcodes to dataframe: {e}")

    return df

def letters_to_category_names(df: pd.DataFrame, col_to_convert: str, conversion_dict: dict) -> pd.DataFrame:
    """
    Converts a column of category letter codes to full category names using a mapping dictionary.

    Parameters:
        df (pd.DataFrame): DataFrame containing the column to convert.
        col_to_convert (str): Name of the column consisting of category letter codes.
        conversion_dict (dict): Dictionary mapping letter codes to full category names.
    
    Returns:
        pd.DataFrame: DataFrame with the specified column converted to category names.
    """
    try:
        # Use map to apply the conversion dictionary to the column
        df[col_to_convert] = df[col_to_convert].map(conversion_dict).fillna("Unknown")

        # Check for unknown values
        if df[col_to_convert].str.contains("Unknown").any():
            logging.warning(f"Column '{col_to_convert}' contains unknown values.")
    except Exception as e:
        logging.error(f"Error converting letter codes to category names: {e}")
    
    return df

def group_categories(df: pd.DataFrame, col: str, grouping_dict: dict, new_col: str = 'GroupedCategories') -> pd.DataFrame:
    """
    Group categories together into a new column based on a grouping dictionary.
    The original column values are mapped to the keys of the dictionary.
    
    Parameters: 
        df (pd.DataFrame): DataFrame with categories to group.
        col (str): Column title with categories to be grouped.
        grouping_dict (dict): Dictionary mapping group names to lists of specific categories.
        new_col (str): Name of the new column that will contain the grouped categories.
    
    Returns:
        pd.DataFrame: DataFrame with the new column that has category groupings.
    """
    try:
        # Reverse the dictionary: mapping each category to its group name
        reverse_dict = {item: group for group, items in grouping_dict.items() for item in items}
        
        # Apply the mapping
        df[new_col] = df[col].map(reverse_dict).fillna("Unknown")

    except Exception as e:
        logging.error(f"Error grouping categories from column '{col}': {e}")
        return None
    
    return df

def concat_cols(df: pd.DataFrame, columns: list, new_col: str = 'ConcatenatedColumn') -> pd.DataFrame:
    """
    Concatenate items from multiple columns into a new column.

    Parameters:
        df (pd.DataFrame): DataFrame containing the columns to concatenate.
        columns (list): A list of column titles to concatenate.
        new_col (str): Title of the new column with concatenated values.

    Returns:
        pd.DataFrame: DataFrame with the new concatenated column.
    """
    try:
        # Convert all values to strings and join them with '-'
        df[new_col] = df[columns].astype(str).apply(lambda row: '-'.join(row), axis=1)
    except Exception as e:
        logging.error(f"Error concatenating column: {e}")
        return None

    return df

def aggregate(df: pd.DataFrame, index: list, aggregations: dict):
    """
    Agrregate dataframes based on index and provided aggregations

    Parameters:
        df (pd.DataFrame): Dataframe to be aggregated
        index (list): List of columns to group aggregatiuons by
        aggregations (dict): Dictionary of column titles (keys) and the aggregations made on them (values)
    
    Returns:
        pd.DataFrame: Aggregated dataframe
    """
    try:
        # group by and follow by aggregating columns
        df = df.groupby(index).agg(aggregations)
    except Exception as e:
        logging.error(f"Error in aggregating dataframe: {e}")
        return pd.DataFrame({}) # return empty dataframe

    return df

def merge_dataframes(dataframes, on_column):
    """
    Merges multiple DataFrames using left joins on a specified column.

    Args:
        dataframes (list): A list of DataFrames to merge.
        on_column (str): The column to merge on.

    Returns:
        pd.DataFrame: Merged dataframe
    """

    if len(dataframes) < 2:
        logging.error("At least two DataFrames must be provided in order to merge.")
        return pd.DataFrame({})  # Or raise an exception if appropriate

    merged_df = dataframes[0].copy()

    try:
        for df in dataframes[1:]:
            merged_df = pd.merge(merged_df, df, on=on_column, how='outer')
    except Exception as e:
        logging.error(f"Error in merging datasets: {e}")

    return merged_df

def retitle_columns(df: pd.DataFrame, column_conversions: dict) -> pd.DataFrame:
    """
    Renames columns in a DataFrame based on a provided dictionary.

    Parameters:
        df (pd.DataFrame): The DataFrame to modify.
        column_conversions (dict): A dictionary mapping old column names to new column names.

    Returns:
        pd.DataFrame: A new DataFrame with the renamed columns.
    """

    try:
        return df.rename(columns=column_conversions)
    except Exception as e:
        logging.error(f"Error in retitling columns: {e}")
        return df
