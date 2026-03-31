# DemandCast (Project 1)

## What is the project?
DemandCast is an individual machine learning project focused on predicting hourly NYC yellow taxi demand by pickup zone. This repository is the foundation for a multi-week ML workflow: setting up a clean and reproducible project structure, exploring and understanding the data, engineering features, training and evaluating models, and eventually serving predictions in a dashboard.

For Project 1, collaboration is encouraged for troubleshooting and discussion, but each student completes and submits their own standalone repository and implementation.

## What is the data?
The project uses the NYC TLC Yellow Taxi Trip Records dataset, beginning with January 2024 trip data. The raw data is provided as a parquet file and includes trip-level records such as pickup/dropoff timestamps, pickup and dropoff location IDs, trip distance, fare-related fields, and passenger counts.

Primary source:
- https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

In this project, raw files are stored locally in the `data/` directory (gitignored), and transformed feature datasets are generated during preprocessing.

## What are we predicting?
We are predicting hourly taxi demand per pickup zone, where demand is defined as the number of trips starting in a specific pickup location within a given hour.

Target framing:
- Unit: pickup zone (`PULocationID`)
- Time granularity: hourly
- Target: trip count (demand) per zone per hour

This prediction supports better short-term demand planning and forms the core supervised learning objective for the DemandCast pipeline.
