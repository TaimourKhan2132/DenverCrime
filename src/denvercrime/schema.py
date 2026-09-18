"""Column contract for the Denver "Crime" dataset (FeatureServer layer 324).

Field list verified against the live service on 2026-09-18. One row = one offense; an
incident (INCIDENT_ID) can have several offenses.
"""

STRING_COLUMNS = (
    "INCIDENT_ID",
    "OFFENSE_ID",
    "OFFENSE_CODE",
    "OFFENSE_TYPE_ID",
    "OFFENSE_CATEGORY_ID",
    "INCIDENT_ADDRESS",
    "DISTRICT_ID",
    "PRECINCT_ID",
    "NEIGHBORHOOD_ID",
)
DATE_COLUMNS = ("FIRST_OCCURRENCE_DATE", "LAST_OCCURRENCE_DATE", "REPORTED_DATE")
FLOAT_COLUMNS = ("GEO_LON", "GEO_LAT", "VICTIM_COUNT")
INT_COLUMNS = ("OFFENSE_CODE_EXTENSION", "GEO_X", "GEO_Y", "IS_CRIME", "IS_TRAFFIC")

# Columns the pipeline actually needs; everything else is carried along but optional.
REQUIRED_COLUMNS = (
    "INCIDENT_ID",
    "OFFENSE_ID",
    "OFFENSE_CATEGORY_ID",
    "FIRST_OCCURRENCE_DATE",
    "LAST_OCCURRENCE_DATE",
    "REPORTED_DATE",
    "GEO_LON",
    "GEO_LAT",
    "IS_CRIME",
    "IS_TRAFFIC",
)

# Date format used by the Hub CSV export, e.g. "5/24/2025 6:55:00 AM".
HUB_CSV_DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"
