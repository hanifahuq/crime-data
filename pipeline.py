from data_handling import *
import logging

# Constants
LOCAL_DATA_PATH = './'
LOG_FILE = os.path.join(LOCAL_DATA_PATH, 'pipeline.log')

# Raw data files
RAW_POLICE = os.path.join(LOCAL_DATA_PATH, 'police-data')
RAW_DEPRIVATION = os.path.join(LOCAL_DATA_PATH, 'deprivation-data')
RAW_PRICEPAID = os.path.join(LOCAL_DATA_PATH, 'price-paid-data')

# Staged data files
STAGED_POLICE = os.path.join(LOCAL_DATA_PATH, 'staged_police.csv')
STAGED_DEPRIVATION = os.path.join(LOCAL_DATA_PATH, 'staged_deprivation.csv')
STAGED_PRICEPAID = os.path.join(LOCAL_DATA_PATH, 'staged_pricepaid.csv')

# Primary data files
PRIMARY_DATA = os.path.join(LOCAL_DATA_PATH, 'primary_data.csv')

# Reporting data files
REPORTING_TIME_DATA = os.path.join(LOCAL_DATA_PATH, 'reporting_time_data.csv')
REPORTING_PROPERTYTYPE_DATA = os.path.join(LOCAL_DATA_PATH, 'reporting_propertytype_data.csv')
REPORTING_POSTCODE_DATA = os.path.join(LOCAL_DATA_PATH, 'reporting_postcode_data.csv')

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

### STAGING ###

def staging():
    """
    Ingest the data, apply cleaning, and store to CSV files for staging.

    Parameters:
        None
    
    Returns:
        None
    """
    logging.info("Starting data staging")

    # Assign header list for pricepaid_df
    pricepaid_columns = [
    "Transaction unique identifier",
    "Price",
    "Date of Transfer",
    "Postcode",
    "Property Type",
    "Old/New",
    "Tenure",
    "PAON",
    "SAON",
    "Street",
    "Locality",
    "Town/City",
    "District",
    "County",
    "PPD Category Type",
    "Record Status"
    ]

    # define columns to delete
    police_del_cols = [
        'Context', 
        'Reported by', 
        'Falls within', 
        'Location', 
        'LSOA code', 
        'LSOA name', 
        'Last outcome category'
    ]

    deprivation__del_cols = [
        'Postcode Status', 
        'LSOA Code', 
        'LSOA Name',
        'Index of Multiple Deprivation Rank'
    ]

    pricepaid_del_cols = [
    "Date of Transfer",
    "Duration",
    "PAON",
    "SAON",
    "Street",
    "Locality",
    "Town/City",
    "District",
    "County",
    "Record Status"
    ]

    # ingest raw
    police_df = ingest_police_data(RAW_POLICE)
    deprivation_df = ingest_data(RAW_DEPRIVATION)
    pricepaid_df = ingest_data(RAW_PRICEPAID, pricepaid_columns)

    try:
        # Apply transformations
        logging.info("Cleaning data...")
        
        # Police data transformations
        police_df = del_cols(police_df, police_del_cols)
        police_df = del_na(police_df)

        # Deprivation data transformations
        deprivation_df = del_cols(deprivation_df, deprivation__del_cols)

        # Price paid data transformations
        pricepaid_df = reformat_date(pricepaid_df, 'Date of Transfer')
        pricepaid_df = del_na(pricepaid_df, ['Postcode'])
        pricepaid_df = del_cols(pricepaid_df, pricepaid_del_cols)
        
        # Save staging files to CSV
        export_data(police_df, STAGED_POLICE)
        export_data(deprivation_df, STAGED_DEPRIVATION)
        export_data(pricepaid_df, STAGED_PRICEPAID)
        
        logging.info("Data staging completed successfully")
        
    except Exception as e:
        logging.error(f"Error during data staging: {e}")

### PRIMARY ###
def primary():
    """
    Primary Layer: Store the transformed data to a CSV file.
    """
    logging.info("Starting Data Processing")

    # ingest staging
    police_df = ingest_data(STAGED_POLICE)
    deprivation_df = ingest_data(STAGED_DEPRIVATION)
    pricepaid_df = ingest_data(STAGED_PRICEPAID)

    # Define category conversion dictionaries
    property_type_conversions = {
        'D': "Detached",
        'S': "Semi-Detached",
        'T': "Terraced",
        'F': "Flats/Maisonettes",
        'O': "Other"
    }

    old_new_conversions = {
        'Y': 'New',
        'N': 'Old'
    }

    tenure_conversions = {
        'F': 'Freehold',
        'L': 'Leasehold'
    }

    # Define the grouping of categories
    grouped_property_types = {
        "House": ["Detached", "Semi-Detached", "Terraced"],
        "Flat": ["Flats/Maisonettes"],
        "Other": ["Other"]
    }
    
    try:
        # Add postcodes
        police_df = add_postcodes(police_df)

        # Apply primary transformations
        # Convert letter codes to words
        pricepaid_df = letters_to_category_names(pricepaid_df, 'Property Type', property_type_conversions)
        pricepaid_df = letters_to_category_names(pricepaid_df, 'Tenure', tenure_conversions)
        pricepaid_df = letters_to_category_names(pricepaid_df, 'Old/New', old_new_conversions)

        # Group categories together
        pricepaid_df = group_categories(pricepaid_df, 'Property Type', grouped_property_types, 'Property')

        # Concat columns for aggregation purposes: Each property type will have it's own category
        pricepaid_df = concat_cols(pricepaid_df, 'Property Type', 'Old/New', 'Combined Property Type')

        all_data = merge_dataframes([police_df, pricepaid_df, deprivation_df], 'Postcode')
        
        # Save to CSV
        export_data(all_data, PRIMARY_DATA)
        
        logging.info("Data Processing completed successfully")
    except Exception as e:
        logging.error(f"Error during data processing: {e}")


### REPORTING ####
def reporting():
    """
    Reporting Layer: Store the aggregated reporting data to a CSV file.
    """
    logging.info("Starting reporting layer.")

    df = ingest_data(PRIMARY_DATA)

    overtime_agg = aggregate(df, ['Postcode', 'Date'], {'Price': 'describe', 'Crime ID': 'count'})

    aggregations = {
    'Crime ID': 'count',
    'Price': ['mean', 'min', 'max', 'std'], 
    'Index of Multiple Deprivation Decile': ['mean', 'min', 'max', 'std']
    }

    property_pricing_agg = aggregate(df, ['Postcode', 'Combined Property Type', 'Tenure'], aggregations)

    postcode_agg = aggregate(df, ['Postcode'], aggregations)

    export_data(overtime_agg, REPORTING_TIME_DATA)
    export_data(property_pricing_agg, REPORTING_PROPERTYTYPE_DATA)
    export_data(postcode_agg, REPORTING_POSTCODE_DATA)

### MAIN PIPELINE ###
def main(pipeline='all'):
    logging.info("Pipeline execution started")

    try:
        if pipeline in ['all', 'staging', 'primary', 'reporting']:
            staging()
            logging.info("Staging execution completed successfully")
            if pipeline == 'staging':
                # If only staging is requested, print success and return
                logging.info("Pipeline run complete")
                return
            # Process the staged data
            primary()
            logging.info("Primary execution completed successfully")
            if pipeline == 'primary':
                # If only primary is requested, print success and return 
                logging.info("Pipeline run complete")
                return
            # Generate reports based on processed data
            reporting()
            logging.info("Reporting execution completed successfully")
            if pipeline == 'reporting':
                logging.info("Pipeline run complete")
                return
            logging.info("Full pipeline run complete")
        else:
            # Inform the user about an invalid pipeline stage input
            logging.critical("Invalid pipeline stage specified. Please choose 'staging', 'primary', 'reporting', or 'all'.")
    except Exception as e:
        # Catch and print any exceptions occurred during pipeline execution
        logging.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()

