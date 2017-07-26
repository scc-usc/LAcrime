#TRACK_TERMS = ["murder", "theft", "genocid", "steal", "attack", "barratri", "captital offens", "car jack", "crime", "feloni", "hijack", "kill", "hit and run", "assault", "rob", "shoplift", "gun", "knife", "burglari", "kidnap", "molest", "rape","polic", "sheriff", "lapd","shoot"]

CONNECTION_STRING = "sqlite:///tweets_LA_geo_28may2.db"
CSV_NAME = "tweets_LA_geo_28may.csv"
TABLE_NAME = "crime_data"

try:
    from private import *
except Exception:
    pass
