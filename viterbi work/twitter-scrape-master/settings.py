#TRACK_TERMS = ["LA crime", "#LASD", "LACrimeStopper1"]

#TRACK_TERMS = ["murder", "theft", "genocide", "stealing", "attack", "barratry", "captital offense", "car jacking", "crime", "felony", "hijack", "kill", "hit and run", "assault", "robbed", "shoplift", "gun", "knife", "burglary", "kidnap", "molest", "rape"]
TRACK_TERMS = ["murder", "theft", "genocid", "steal", "attack", "barratri", "captital offens", "car jack", "crime", "feloni", "hijack", "kill", "hit and run", "assault", "rob", "shoplift", "gun", "knife", "burglari", "kidnap", "molest", "rape","polic", "sheriff", "lapd","shoot"]

CONNECTION_STRING = "sqlite:///tweets_LA_28may.db"
CSV_NAME = "tweets_LA_28may.csv"
TABLE_NAME = "crime_data"

try:
    from private import *
except Exception:
    pass
