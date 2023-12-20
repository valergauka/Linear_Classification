import numpy as np
import pandas as pd

def parse_stackoverflow_data():
    data = pd.read_csv('eggs.csv.gz', dtype=str, keep_default_na=False)

    # Conversion functions
    def convert_boolean(value):
        return 1.0 if value == 'y' else 0.0

    def convert_integer(value):
        return float(value) if value != 'NA' else 0.0

    def convert_career_sat(value):
        mapping = {'vd': -2.0, 'sd': -1.0, 'ne': 0.0, 'NA': 0.0, 'ss': 1.0, 'vs': 2.0}
        return mapping.get(value, 0.0)

    def convert_ed_level(value):
        mapping = {'other': 0.0, 'bachelors': 1.0, 'masters': 1.5, 'doctoral': 2.0}
        return mapping.get(value, 0.0)

    def convert_mgr_idiot(value):
        mapping = {'NA': -1.0, 'not': -1.0, 'some': 0.0, 'very': 1.0}
        return mapping.get(value, 0.0)

    def convert_op_sys(value):
        mapping = {'win': -1.0, 'mac': 0.0, 'NA': 0.0, 'tux': 1.0, 'BSD': 1.0}
        return mapping.get(value, 0.0)

    def convert_open_sourcer(value):
        mapping = {'never': 0.0, 'year': 0.5, 'month-year': 1.0, 'month': 2.0}
        return mapping.get(value, 0.0)

    def convert_org_size(value):
        if value == 'NA':
            return 0.0
        else:
            a, b = map(float, value.split('-'))
            return np.log(a)

    data['MgrWant'] = data['MgrWant'].apply(convert_boolean)
    data['Age'] = data['Age'].apply(convert_integer)
    data['CodeRevHrs'] = data['CodeRevHrs'].apply(convert_integer)
    data['ConvertedComp'] = data['ConvertedComp'].apply(convert_integer)
    data['Dependents'] = data['Dependents'].apply(convert_boolean)
    data['DevEnvironVSC'] = data['DevEnvironVSC'].apply(convert_boolean)
    data['DevTypeFullStack'] = data['DevTypeFullStack'].apply(convert_boolean)
    data['EdLevel'] = data['EdLevel'].apply(convert_ed_level)
    data['EduOtherMOOC'] = data['EduOtherMOOC'].apply(convert_boolean)
    data['Extraversion'] = data['Extraversion'].apply(convert_boolean)
    data['GenderIsMan'] = data['GenderIsMan'].apply(convert_boolean)
    data['Hobbyist'] = data['Hobbyist'].apply(convert_boolean)
    data['MgrIdiot'] = data['MgrIdiot'].apply(convert_mgr_idiot)
    data['OpSys'] = data['OpSys'].apply(convert_op_sys)
    data['OpenSourcer'] = data['OpenSourcer'].apply(convert_open_sourcer)
    data['OrgSize'] = data['OrgSize'].apply(convert_org_size)
    data['Student'] = data['Student'].apply(convert_boolean)
    data['UndergradMajorIsComputerScience'] = data['UndergradMajorIsComputerScience'].apply(convert_boolean)
    data['UnitTestsProcess'] = data['UnitTestsProcess'].apply(convert_boolean)
    data['WorkWeekHrs'] = data['WorkWeekHrs'].apply(convert_integer)
    data['YearsCode'] = data['YearsCode'].apply(convert_integer)
    data['YearsCodePro'] = data['YearsCodePro'].apply(convert_integer)

    data = data.drop(columns=['Country', 'Respondent', 'EduOtherSelf'])

    return data
