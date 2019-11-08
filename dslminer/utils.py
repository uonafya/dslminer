def make_date(datestring):
    year = datestring[0:4]
    month = datestring[4:6]
    day = "01"
    date = year+"-"+month+"-"+day
    return date
