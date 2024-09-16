import pytest
import os
import pandas as pd
from io import StringIO
import unittest
from unittest import mock
import logging
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
from unittest.mock import patch, Mock

from data_handling import * 

class TestIngestPoliceData(unittest.TestCase):
    @mock.patch("os.listdir")
    @mock.patch("os.path.isdir")
    @mock.patch("os.path.exists")
    def test_valid_policefile_ingestion(self, mock_exists, mock_isdir, mock_listdir):
        """Tests police_file_ingestion() for when the function runs as it should"""

        mock_listdir.return_value = ['2022-01', '2022-02']
        mock_isdir.return_value = True
        mock_exists.return_value = True
        forces = ['essex', 'sussex']

        # Mock CSV data
        csv_data = {
            'col1': [1, 2],
            'col2': [3, 4]
        }
        df=pd.DataFrame(csv_data)

        # Set up the mock to return an example DataFrame
        with mock.patch("pandas.read_csv", return_value=df):
        # mock_read_csv.return_value = pd.DataFrame(csv_data)

            # Call the function
            df = ingest_police_data("fake\\dir", forces)

            self.assertEqual(df.shape, (8, 3))  # assert the size of the imported concatenated data

    @mock.patch("os.listdir")
    @mock.patch("os.path.isdir")
    @mock.patch("os.path.exists")
    def test_valid_policefile_ingestion(self, mock_exists, mock_isdir, mock_listdir):
        """Tests police_file_ingestion() for when the function runs as it should"""

        mock_listdir.return_value = ['2022-01', '2022-02']
        mock_isdir.return_value = True
        mock_exists.return_value = True
        forces = ['essex', 'sussex']

        # Mock CSV data
        csv_data = {
            'col1': [1, 2],
            'col2': [3, 4]
        }
        df=pd.DataFrame(csv_data)

        # Set up the mock to return an example DataFrame
        with mock.patch("pandas.read_csv", return_value=df):
        # mock_read_csv.return_value = pd.DataFrame(csv_data)

            # Call the function
            df = ingest_police_data("fake\\dir", forces)

            self.assertIn('Police force', df.columns)  # check police force column was added to the columns


    @mock.patch("os.listdir")
    @mock.patch("os.path.isdir")
    def test_novalid_directories_policefile_ingestion(self, mock_isdir, mock_listdir):
        """
        Tests police_file_ingestion() for when the given directory has no folders in them
        """
        # Example files within directory
        mock_listdir.return_value = ['2022-01.csv', '2022-02.xlsx']

        # All values in directory are not folders
        mock_isdir.return_value = False

        forces = ['essex', 'sussex']

        # Call the function
        df = ingest_police_data("fake\\dir", forces)

        self.assertTrue(df.empty)  # assert that df is empty

    @mock.patch("os.listdir")
    @mock.patch("os.path.isdir")
    @mock.patch("os.path.exists")
    def test_invalid_directoryname_policefile_ingestion(self, mock_exists, mock_isdir, mock_listdir):
        """
        Tests police_file_ingestion() for when there are directory names that do not follow the correct format
        """
        
        mock_listdir.return_value = ['123-456']
        mock_isdir.return_value = True
        mock_exists.return_value = True
        forces = ['essex', 'sussex']

        with self.assertLogs(level='WARNING') as log:
            df = ingest_police_data("fake\\dir", forces)  # Call the function that logs the warning

        # Assert that the warning message is present in the logs
        self.assertIn("Directory name '123-456' is not in the format YYYY-MM", log.output[0])

        # If directories are not in correct format, read no files
        self.assertTrue(df.empty)  # assert that df is empty

    @mock.patch("os.listdir")
    @mock.patch("os.path.isdir")
    @mock.patch("os.path.exists")
    def test_filenotfound_policefile_ingestion(self, mock_exists, mock_isdir, mock_listdir):
        """
        Tests police_file_ingestion() for when there are files to be looked for but do not exist
        """
        
        mock_listdir.return_value = ['2022-01']
        mock_isdir.return_value = True
        mock_exists.return_value = False
        forces = ['essex']

        base_dir = "fake\\dir"
        expected_filepath = os.path.join(base_dir, '2022-01', '2022-01-essex-street.csv')

        with self.assertLogs(level='WARNING') as log:
            df = ingest_police_data(base_dir, forces)  # Call the function that logs the warning

        # Assert that the warning message is present in the logs
        self.assertIn(f"File not found: {expected_filepath}", log.output[0])

        # If directories are not in correct format, read no files
        self.assertTrue(df.empty)  # assert that df is empty

class TestConcatenateDataframes(unittest.TestCase):
    def test_empty_list(self):
        with self.assertLogs(level='WARNING') as log:
            df = concatenate_dataframes([])  # Call the function that logs the warning

        # Assert that the warning message is present in the logs
        self.assertIn("No dataframes found in list, returning empty dataframe...", log.output[0])

        # If directories are not in correct format, read no files
        self.assertTrue(df.empty)  # assert that df is empty

    def test_single_dataframe(self):
        df = pd.DataFrame({'A': [1, 2, 3]})

        result = concatenate_dataframes([df])

        self.assertEqual(result.shape, (3, 1))

    def test_multiple_dataframes(self):
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [7, 8, 9]})
        df2 = pd.DataFrame({'A': [4, 5, 6], 'B': [10, 11, 12]})

        result = concatenate_dataframes([df1, df2])

        self.assertEqual(result.shape, (6, 2))

class TestIngestData(unittest.TestCase):
    # As I am using unittest, I cannot use fixtures, therefore creating a setUp() method for what I would use for fixtures
    def setUp(self):
        self.dataframe = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4]
        })
        self.fake_dir = "fake\\dir"
        self.lst_dir = ["file1.csv", "file2.csv", "file3.xlsx", "file4.csv", "file5.txt"]
        self.lst_dir_nocsv = ["file3.xlsx", "file4.xlsx", "file5.txt"]
    
    # The test method using mock patching and the pytest fixture
    @mock.patch("os.path.basename")
    @mock.patch("os.path.isfile")
    @mock.patch("os.path.isdir")
    @mock.patch("pandas.read_csv")
    def test_single_csv_file(self, mock_read_csv, mock_isdir, mock_isfile, mock_basename):
        """Test function for when a single CSV file is passed through"""
        
        # Mocking os.path.basename
        file_name = 'file.csv'
        mock_basename.return_value = file_name
        # mock_endswith.return_value = True

        # Mocking os.path.isfile and os.path.isdir to simulate a file path input
        mock_isfile.return_value = True
        mock_isdir.return_value = False  # Not a directory, so treat as a file

        # Mocking pd.read_csv to return the fixture data
        mock_read_csv.return_value = self.dataframe

        # Call the function with a fake path
        result = ingest_data(os.path.join(self.fake_dir, file_name))

        # Assert that the result matches the expected DataFrame
        self.assertEqual(result.shape, self.dataframe.shape)

    # test directory containing csv_files
    @mock.patch("os.path.basename")
    @mock.patch("os.path.isfile")
    @mock.patch("os.path.isdir")
    @mock.patch("os.listdir")
    @mock.patch("pandas.read_csv")
    def test_directory_with_csv(self, mock_read_csv, mock_listdir, mock_isdir, mock_isfile, mock_basename):
        """Test function when folder of csv's are passed through"""
        mock_basename.return_value = "folder"
        mock_isfile.return_value = False
        mock_isdir.return_value = True

        mock_listdir.return_value = self.lst_dir
        mock_read_csv.return_value = self.dataframe

        return_df = ingest_data(self.fake_dir)

        self.assertEqual(return_df.shape, (3*2, 2)) # Should ingest 3 dataframes with 2 rows and 2 columns
        
    # test directory containing csv_files
    @mock.patch("os.path.basename")
    @mock.patch("os.path.isfile")
    @mock.patch("os.path.isdir")
    @mock.patch("os.listdir")
    @mock.patch("pandas.read_csv")
    def test_directory_with_no_csv(self, mock_read_csv, mock_listdir, mock_isdir, mock_isfile, mock_basename):
        """Test function when path is a folder that has no csvs"""
        mock_basename.return_value = "folder"
        mock_isfile.return_value = False
        mock_isdir.return_value = True

        mock_listdir.return_value = self.lst_dir_nocsv
        mock_read_csv.return_value = self.dataframe

        return_df = ingest_data(self.fake_dir)

        self.assertTrue(return_df.empty) # Should be empty as no csv files were ingested
    
    # test directory containing csv_files
    @mock.patch("os.path.basename")
    @mock.patch("os.path.isfile")
    @mock.patch("os.path.isdir")
    @mock.patch("os.listdir")
    @mock.patch("pandas.read_csv")
    def test_directory_with_invalid_dir(self, mock_read_csv, mock_listdir, mock_isdir, mock_isfile, mock_basename):
        """Test function when path is not a file nor folder"""
        mock_basename.return_value = "folder"
        mock_isfile.return_value = False
        mock_isdir.return_value = False

        mock_read_csv.return_value = self.dataframe

        with mock.patch('logging.error') as mock_error:
            return_df = ingest_data(self.fake_dir)

        mock_error.assert_called_once_with(f"Invalid path: {self.fake_dir} is neither a CSV file nor a directory.")

class TestDelCols(unittest.TestCase):
    def test_delete_existing_columns(self):
        """Tests if the function correctly deletes specified columns from the DataFrame."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        result = del_cols(df, ['B', 'C'])
        self.assertListEqual(result.columns.tolist(), ['A'])

    def test_delete_non_existent_column(self):
        """Tests if the function raises an appropriate error or warning when a column to be deleted doesn't exist."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        with self.assertRaises(KeyError):  # Assuming KeyError is raised for non-existent columns
            del_cols(df, ['C'])

    def test_empty_cols_list(self):
        """Tests if the function does nothing when the `cols` list is empty."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = del_cols(df, [])
        self.assertListEqual(result.columns.tolist(), ['A', 'B'])

class TestDelNa(unittest.TestCase):
    def test_delete_rows_with_na_across_dataset(self):
        """
        Tests if the function correctly deletes rows with NA values across all columns when `cols` is empty.
        """
        df_dict = {
            'A': [1, 2, None], 
            'B': [4, None, 6]
        }
        df = pd.DataFrame(df_dict)
        result = del_na(df)
        self.assertEqual(result.shape, (1, 2)) # There should only be one row with 2 columns

    def test_delete_rows_with_na_in_specified_columns(self):
        """
        Tests if the function correctly deletes rows with NA values in the specified columns.
        """
        # Create a DataFrame with pd.NA
        df = pd.DataFrame({'A': [1, 2, pd.NA], 'B': [pd.NA, 5, 6]})

        # Call your del_na function (which removes rows based on NaN in the specified columns)
        result = del_na(df, ['A'])

        # Create the expected result
        expected_result = pd.Series([pd.NA, 5], name='B')

        # Compare the two Series (result['B'] and expected_result) using assert_series_equal
        assert_series_equal(result['B'], expected_result, check_dtype=False)

    def test_handle_dataframe_with_no_na_values(self):
        """
        Tests if the function returns the original DataFrame unchanged if there are no NA values.
        """
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = del_na(df)
        assert_frame_equal(result, df)  # Assert that the original DataFrame is returned unchanged

class TestRemoveDuplicates(unittest.TestCase):
    def test_remove_duplicate_rows(self):
        """Tests if the function correctly removes duplicate rows."""
        df = pd.DataFrame({'A': [1, 1, 2], 'B': ['a', 'a', 'b']})
        result = remove_duplicates(df)
        self.assertEqual(result.shape, (2, 2))

    def test_handle_dataframe_with_no_duplicates(self):
        """Tests if the function returns the original DataFrame unchanged if there are no duplicates."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        result = remove_duplicates(df)
        assert_frame_equal(result, df)

    def test_handle_dataframe_with_all_duplicate_rows(self):
        """Tests if the function correctly removes all but one duplicate row."""
        df = pd.DataFrame({'A': [1, 1, 1], 'B': ['a', 'a', 'a']})
        result = remove_duplicates(df)
        self.assertEqual(result.shape, (1, 2))

    def test_handle_mixed_data_types(self):
        """Tests if the function correctly handles DataFrames with mixed data types."""
        df = pd.DataFrame({'A': [1, 1, 2], 'B': ['a', 'a', 'b'], 'C': [True, True, False]})
        result = remove_duplicates(df)
        self.assertEqual(result.shape, (2, 3))

class TestReformatDate(unittest.TestCase):
    def test_valid_date_formats(self):
        """Tests if the function correctly handles valid date formats."""

        df = pd.DataFrame({'date': ['2021-04-29 00:00', '2021-04-29 00:00', '2021-04-29 00:00']})
        result = reformat_date(df, 'date')

        # Check if the 'date' column values are in the YYYY-MM format
        for date_value in result['date']:
            self.assertRegex(str(date_value), r"\d{4}-\d{2}")
    
    def test_invalid_dates(self):
        """Tests if function returns warnings for date column containing NaNs"""

        # One item not a date
        df = pd.DataFrame({'date': ['2021-04-29 00:00', '100-100-100', '2021-04-29 00:00']})
        result = reformat_date(df, 'date')

        # Check if the 'date' column has a single NaN value
        return_df = reformat_date(df, 'date')

        self.assertEqual(return_df['date'].isna().sum(), 1)

class TestGetPostcode(unittest.TestCase):

    @mock.patch('requests.get')
    def test_valid_lonlat(self, mock_get):
        """Test for when a valid API request is made"""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": 200,
            "result": [
                {"postcode": "RM8 1AS"}
            ]
        }
        mock_get.return_value = mock_response

        lon = 0.14
        lat = 51.0

        # Call the function
        postcode = get_postcode(lon, lat)

        # Assert expected result
        self.assertEqual(postcode, "RM8 1AS")

    @mock.patch('requests.get')
    def test_invalid_lonlat(self, mock_get):
        """Test for when an invalid API request is made"""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "status": 404,
            "result": [
                {"postcode": "RM8 1AS"}
            ]
        }
        mock_get.return_value = mock_response

        lon = 0.14
        lat = 51.0

        with mock.patch('logging.error') as mock_error:
            # Call the function
            postcode = get_postcode(lon, lat)

        mock_error.assert_called_once_with("Error retrieving postcode from postcode.io: Error 404")

class test_add_postcodes(unittest.TestCase):
    @mock.patch("data_handling.get_postcode")
    def test_true_postcode(self, mock_get_postcode):
        mock_get_postcode = 'RM8 3PJ'

        df = pd.DataFrame({
            'Latitude': [1, 2, 3],
            'Longitude': [4, 5, 6]
        })

        return_df = add_postcodes(df)

        self.assertTrue('Postcode' in list(return_df.columns))

class TestLettersToCategoryNames(unittest.TestCase):

    def test_successful_conversion(self):
        """
        Tests successful conversion of letter codes to category names.
        """
        df = pd.DataFrame({'category_code': ['A', 'B', 'C']})
        conversion_dict = {'A': 'Category A', 'B': 'Category B', 'C': 'Category C'}
        expected_output = ['Category A', 'Category B', 'Category C']

        result_df = letters_to_category_names(df, 'category_code', conversion_dict)

        self.assertListEqual(list(result_df['category_code']), expected_output)

    def test_unknown_values(self):
        """
        Tests handling of unknown values.
        """
        df = pd.DataFrame({'category_code': ['A', 'B', 'D', 'L']})
        conversion_dict = {'A': 'Category A', 'B': 'Category B'}
        expected_unmapped_categories = 2
        result_df = letters_to_category_names(df, 'category_code', conversion_dict)

        self.assertEqual(len(result_df[result_df['category_code']=='Unknown']), expected_unmapped_categories)

class TestGroupCategories(unittest.TestCase):

    def test_successful_grouping(self):
        """
        Tests successful grouping of categories.
        """
        df = pd.DataFrame({'categories': ['A', 'B', 'C', 'D']})
        grouping_dict = {'Group 1': ['A', 'B'], 'Group 2': ['C', 'D']}
        expected_df = pd.DataFrame({'categories': ['A', 'B', 'C', 'D'], 'GroupedCategories': ['Group 1', 'Group 1', 'Group 2', 'Group 2']})

        result_df = group_categories(df, 'categories', grouping_dict)

        self.assertTrue(result_df.equals(expected_df))

    def test_unknown_values(self):
        """
        Tests handling of unknown values.
        """
        df = pd.DataFrame({'categories': ['A', 'B', 'E']})
        grouping_dict = {'Group 1': ['A', 'B'], 'Group 2': ['C', 'D']}

        result_df = group_categories(df, 'categories', grouping_dict)

        self.assertIn('Unknown', result_df['GroupedCategories'].values)

    def test_invalid_column_name(self):
        """
        Tests handling of invalid column names.
        """
        df = pd.DataFrame({'categories': ['A', 'B', 'C']})
        grouping_dict = {'Group 1': ['A', 'B'], 'Group 2': ['C', 'D']}

        result_df = group_categories(df, 'invalid_column', grouping_dict)

        self.assertIsNone(result_df)

    def test_invalid_grouping_dict(self):
        """
        Tests handling of an invalid grouping dictionary.
        """
        df = pd.DataFrame({'categories': ['A', 'B', 'C']})
        grouping_dict = None

        result_df = group_categories(df, 'categories', grouping_dict)

        self.assertIsNone(result_df)

class TestConcatCols(unittest.TestCase):

    def test_successful_concatenation(self):
        """
        Tests successful concatenation of multiple columns.
        """
        df = pd.DataFrame({'col1': ['A', 'B', 'C'], 'col2': ['1', '2', '3']})
        expected_df = pd.DataFrame({'col1': ['A', 'B', 'C'], 'col2': ['1', '2', '3'], 'ConcatenatedColumn': ['A-1', 'B-2', 'C-3']})

        result_df = concat_cols(df, ['col1', 'col2'])

        self.assertTrue(result_df.equals(expected_df))

    def test_invalid_column_name(self):
        """
        Tests handling of invalid column names.
        """
        df = pd.DataFrame({'col1': ['A', 'B', 'C'], 'col2': ['1', '2', '3']})

        result_df = concat_cols(df, ['invalid_column', 'col2'])

        self.assertIsNone(result_df)

    def test_different_data_types(self):
        """
        Tests handling of different data types.
        """
        df = pd.DataFrame({'col1': ['A', 'B', 'C'], 'col2': [1, 2, 3]})

        result_df = concat_cols(df, ['col1', 'col2'])

        self.assertIn('A-1', result_df['ConcatenatedColumn'].values)

class TestAggregate(unittest.TestCase):

    def test_successful_aggregation(self):
        """
        Tests successful aggregation of a DataFrame.
        """
        df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
        index = ['group']
        aggregations = {'col1': 'sum', 'col2': 'mean'}
        expected_df = pd.DataFrame({'col1': [3, 7], 'col2': [5.5, 7.5]}, index=['A', 'B'])

        result_df = aggregate(df, index, aggregations)

        self.assertTrue(result_df.equals(expected_df))

    def test_invalid_index_column(self):
        """
        Tests handling of an invalid index column.
        """
        df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
        index = ['invalid_column']
        aggregations = {'col1': 'sum', 'col2': 'mean'}

        result_df = aggregate(df, index, aggregations)

        self.assertTrue(result_df.empty)


    def test_invalid_aggregation_function(self):
        """
        Tests handling of an invalid aggregation function.
        """
        df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
        index = ['group']
        aggregations = {'col1': 'invalid_agg', 'col2': 'mean'}


        result_df = aggregate(df, index, aggregations)

        self.assertTrue(result_df.empty)

    def test_empty_dataframe(self):
        """
        Tests handling of an empty DataFrame.
        """
        df = pd.DataFrame()
        index = ['group']
        aggregations = {'col1': 'sum', 'col2': 'mean'}

        result_df = aggregate(df, index, aggregations)

        self.assertTrue(result_df.empty)

    def test_invalid_aggregations_dict(self):
        """
        Tests handling of an invalid aggregations dictionary.
        """
        df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
        index = ['group']
        aggregations = None

        result_df = aggregate(df, index, aggregations)

        self.assertTrue(result_df.empty)

class TestMergeDataframes(unittest.TestCase):

    def test_successful_merge(self):
        """
        Tests successful merging of multiple DataFrames.
        """
        df1 = pd.DataFrame({'id': [1, 2, 3], 'col1': ['A', 'B', 'C']})
        df2 = pd.DataFrame({'id': [1, 3, 4], 'col2': [10, 20, 30]})
        dataframes = [df1, df2]
        expected_df = pd.DataFrame({'id': [1, 2, 3, 4], 'col1': ['A', 'B', 'C', None], 'col2': [10, None, 20, 30]})

        result_df = merge_dataframes(dataframes, 'id')

        self.assertTrue(result_df.equals(expected_df))

    def test_insufficient_dataframes(self):
        """
        Tests handling of insufficient DataFrames.
        """
        dataframes = [pd.DataFrame({'id': [1, 2, 3], 'col1': ['A', 'B', 'C']})]

        with mock.patch('logging.error') as mock_error:
            result_df = merge_dataframes(dataframes, 'id')

        mock_error.assert_called_once_with("At least two DataFrames must be provided in order to merge.")

class TestRetitleColumns(unittest.TestCase):

    def test_successful_retitling(self):
        """
        Tests successful renaming of columns.
        """
        df = pd.DataFrame({'old_column1': [1, 2, 3], 'old_column2': [4, 5, 6]})
        column_conversions = {'old_column1': 'new_column1', 'old_column2': 'new_column2'}
        expected_columns = ["new_column1", "new_column2"]

        result_df = retitle_columns(df, column_conversions)

        self.assertListEqual(list(result_df.columns), expected_columns)

    def test_invalid_column_name(self):
        """
        Tests handling of an invalid column name in the conversion dictionary.
        """
        df = pd.DataFrame({'old_column1': [1, 2, 3], 'old_column2': [4, 5, 6]})
        column_conversions = {'invalid_column': 'new_column1', 'old_column2': 'new_column2'}

        result_df = retitle_columns(df, column_conversions)
        expected_columns = ['old_column1', 'new_column2']

        self.assertListEqual(list(result_df.columns), expected_columns)

if __name__ == '__main__':
    unittest.main()