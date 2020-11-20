def make_date(datestring):
    year = datestring[0:4]
    month = datestring[4:6]
    day = "01"
    date = year+"-"+month+"-"+day
    return date

def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month