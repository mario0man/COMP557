from bayesian.bbn import *
from bayesian.utils import make_key

#True for Christian, False for Other
def f_religion(religion):
    if religion is True:
        return 0.73
    return 0.27

#True for white, False for Other
def f_race(race):
    if race is True:
        return 0.724
    return 0.276
#True for middleclass + rich, False for working class +
def f_income(income):
    if income is True:
        return 0.58
    return 0.42

def f_gender(gender):
    return 0.5

#True for young voters, false for the rest
def f_age(age):
    if age is True:
        return 0.31
    return 0.69

#True for City/Suburb, False for Rural/Country
def f_location(race, religion, income, location):
    table = dict()
    table['tttt'] = 0.68
    table['tttf'] = 0.32
    table['ttft'] = 0.05
    table['ttff'] = 0.95
    table['tftt'] = 0.98
    table['tftf'] = 0.02
    table['tfft'] = 0.94
    table['tfff'] = 0.06
    table['fttt'] = 0.90
    table['fttf'] = 0.10
    table['ftft'] = 0.60
    table['ftff'] = 0.40
    table['fftt'] = 0.94
    table['fftf'] = 0.06
    table['ffft'] = 0.75
    table['ffff'] = 0.25
    key = ''
    key = key + 't' if race else key + 'f'
    key = key + 't' if religion else key + 'f'
    key = key + 't' if income else key + 'f'
    key = key + 't' if location else key + 'f'
    return table[key]


#True went to college
def f_college(race, income, college):
    table = dict()
    table['ttt'] = 0.90
    table['ttf'] = 0.10
    table['tft'] = 0.20
    table['tff'] = 0.80
    table['ftt'] = 0.70
    table['ftf'] = 0.30
    table['fft'] = 0.05
    table['fff'] = 0.95
    key = ''
    key = key + 't' if race else key + 'f'
    key = key + 't' if income else key + 'f'
    key = key + 't' if college else key + 'f'
    return table[key]

def f_green(college, age, green):
    table = dict()
    table['ttt'] = 0.95
    table['ttf'] = 0.05
    table['tft'] = 0.60
    table['tff'] = 0.40
    table['ftt'] = 0.80
    table['ftf'] = 0.20
    table['fft'] = 0.25
    table['fff'] = 0.75
    key = ''
    key = key + 't' if college else key + 'f'
    key = key + 't' if age else key + 'f'
    key = key + 't' if green else key + 'f'
    return table[key]

#True for-abortion, False against abortion
def f_abortion(location, college, abortion):
    table = dict()
    table['ttt'] = 0.95
    table['ttf'] = 0.05
    table['tft'] = 0.79
    table['tff'] = 0.21
    table['ftt'] = 0.40
    table['ftf'] = 0.60
    table['fft'] = 0.03
    table['fff'] = 0.97
    key = ''
    key = key + 't' if location else key + 'f'
    key = key + 't' if college else key + 'f'
    key = key + 't' if abortion else key + 'f'
    return table[key]

#True for-immigration, False against immigration
def f_immigration(location, college, immigration):
    table = dict()
    table['ttt'] = 0.90
    table['ttf'] = 0.10
    table['tft'] = 0.60
    table['tff'] = 0.40
    table['ftt'] = 0.30
    table['ftf'] = 0.70
    table['fft'] = 0.20
    table['fff'] = 0.80
    key = ''
    key = key + 't' if location else key + 'f'
    key = key + 't' if college else key + 'f'
    key = key + 't' if immigration else key + 'f'
    return table[key]

def f_conservative(immigration, abortion, conservative):
    table = dict()
    table['ttt'] = 0.01
    table['ttf'] = 0.99
    table['tft'] = 0.90
    table['tff'] = 0.10
    table['ftt'] = 0.70
    table['ftf'] = 0.30
    table['fft'] = 0.99
    table['fff'] = 0.01
    key = ''
    key = key + 't' if immigration else key + 'f'
    key = key + 't' if abortion else key + 'f'
    key = key + 't' if conservative else key + 'f'
    return table[key]

def f_liberal(immigration, green, liberal):
    table = dict()
    table['ttt'] = 0.94
    table['ttf'] = 0.06
    table['tft'] = 0.82
    table['tff'] = 0.18
    table['ftt'] = 0.62
    table['ftf'] = 0.38
    table['fft'] = 0.02
    table['fff'] = 0.98
    key = ''
    key = key + 't' if immigration else key + 'f'
    key = key + 't' if green else key + 'f'
    key = key + 't' if liberal else key + 'f'
    return table[key]

#True for Democrat, False for Republican
def f_prev(prev):
    if prev is True:
        return 0.49
    return 0.51

def f_republican(conservative, prev, republican):
    table = dict()
    table['ttt'] = 0.60
    table['ttf'] = 0.40
    table['tft'] = 0.98
    table['tff'] = 0.02
    table['ftt'] = 0.01
    table['ftf'] = 0.99
    table['fft'] = 0.40
    table['fff'] = 0.60
    key = ''
    key = key + 't' if conservative else key + 'f'
    key = key + 't' if prev else key + 'f'
    key = key + 't' if republican else key + 'f'
    return table[key]

def f_democrat(liberal, prev, democrat):
    table = dict()
    table['ttt'] = 0.99
    table['ttf'] = 0.01
    table['tft'] = 0.65
    table['tff'] = 0.35
    table['ftt'] = 0.40
    table['ftf'] = 0.60
    table['fft'] = 0.01
    table['fff'] = 0.99
    key = ''
    key = key + 't' if liberal else key + 'f'
    key = key + 't' if prev else key + 'f'
    key = key + 't' if democrat else key + 'f'
    return table[key]

if __name__ == '__main__':
    g = build_bbn(
        f_race, f_religion, f_income, f_gender, f_age,
        f_location, f_college, f_immigration, f_abortion, f_green,
        f_conservative, f_liberal, f_prev,
        f_republican, f_democrat)
