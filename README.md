# RegionalCrimeAnalysis

Overview:

This project aims to expand upon an existing two-force crime analysis by incorporating 2 third-party datasets. By doing so, we will be able to explore crime trends at a regional level and investigate potential correlations with other factors. Additionally, the feasibility of a postcode-level analysis to link crimes to properties on the market will be explored.


Key Features of Requirements:

Data Pipeline: A well-structured and readable pipeline that combines existing and new datasets, ensuring data quality and reproducibility.
Data Layers: Clear separation of data handling and processing logic for better maintainability.
Unit Testing: Comprehensive unit tests to verify the correctness and robustness of the pipeline.
Version Control: Use of a version control system (e.g., GitHub) to track changes and collaborate effectively.
Third-Party Dataset: Integration of a relevant external dataset to enrich the analysis.
Pipeline Best Practices: Adherence to coding best practices, including data layers, logging, and unit testing.
GitHub Repository: Storage of all project code, including the pipeline, unit tests, and documentation.

Files:
data_handling.py: Contains all of my functions created
pipeline_tests.py: Tests of functions from data_handling
pipeline.py: The final pipeline using functions from data handling

Data Sources:
Crime data: [Police.uk](https://data.police.uk/data/)
Estate prices paid (Over 2021-2022) - [Price Paid Data - GOV.UK (www.gov.uk)](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads#single-file)
Deprivation (Review last released in 2019) - [English indices of deprivation 2019 (opendatacommunities.org)](https://imd-by-geo.opendatacommunities.org/imd/2019/area)


Contributions to the project are welcome. Please follow the standard GitHub workflow for creating pull requests and addressing issues.
