import file_utils as fu

SEASON_INDEX = 0
ROUND_INDEX = 1
TEAM_INDEX = 2
VALUES_START_INDEX = 3

ALPHA = 0.1

def iterate_emas(previous, current, alpha):
    emas = []
    for i in range(len(previous)):
        p = previous[i]
        c = current[i]
        n = alpha * c + (1 - alpha) * p
        emas.append(n)
    return emas

def calculate_emas(alpha=ALPHA):
    csv_lines = fu.read_csv(fu.csv_rounds_path())
    ema_lines = [csv_lines[0]]
    csv_lines = csv_lines[1:]

    previous_emas = {}

    for line in csv_lines:
        s = line[SEASON_INDEX]
        r = line[ROUND_INDEX]
        team = line[TEAM_INDEX]
        ema_line = []
        current_emas = []
        current_values = line[VALUES_START_INDEX:]
        current_values = [float(x) for x in current_values]

        if team in previous_emas:
            previous = previous_emas[team]
            current_emas = iterate_emas(previous, current_values, alpha)
        else:
            current_emas = current_values

        previous_emas[team] = current_emas
        ema_line = [s, r, team] + current_emas
        ema_line = [str(x) for x in ema_line]
        ema_lines.append(ema_line)

    fu.write_csv(ema_lines, fu.csv_emas_path())

if __name__ == "__main__":
    calculate_emas()
