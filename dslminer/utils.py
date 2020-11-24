from datetime import datetime

def make_date(datestring):
    year = datestring[0:4]
    month = datestring[4:6]
    day = "01"
    date = year+"-"+month+"-"+day
    return date

def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def months_between(d1, d2):
    # d1 = datetime.strptime(d1, "%Y-%m-%d")
    # d2 = datetime.strptime(d2, "%Y-%m-%d")
    num_months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
    return num_months
