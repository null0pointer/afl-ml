import os.path

def raw_round_path(s, r):
    return 'data/raw/' + str(s) + '/rounds/' + 'round' + str(r) + '.json'
    
def raw_match_path(s, r):
    return 'data/raw/' + str(s) + '/matches/' + 'match' + str(r) + '.json'
    
def csv_rounds_path():
    return 'data/csv/rounds.csv'
    
def csv_matches_path():
    return 'data/csv/matches.csv'
    
def csv_emas_path():
    return 'data/csv/emas.csv'
    
def csv_inputs_path():
    return 'data/csv/inputs.csv'
    
def round_exists(s, r):
    return os.path.isfile(raw_round_path(s, r))
    
def match_exists(s, r):
    return os.path.isfile(raw_match_path(s, r))
    
def remove_trailing_newlines(text):
    while text[-1] == '\n':
        text = text[:-1]
    return text
    
def read_csv(path):
    csv_lines = []
    with open(path) as csv:
        csv_text = csv.read()
        csv_text = remove_trailing_newlines(csv_text)
        csv_lines = [line.split(',') for line in csv_text.split('\n')]
        csv.close()
    return csv_lines
    
def write_csv(csv_lines, path):
    with open(path, 'w') as csv:
        for line in csv_lines:
            str_line = ','.join(line)
            csv.write(str_line + '\n')
        csv.close()
