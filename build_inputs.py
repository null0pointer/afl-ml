import file_utils as fu

SEASON_INDEX = 0
ROUND_INDEX = 1
TEAM_INDEX = 2
VALUES_START_INDEX = 3

GOALS_INDEX = 3
BEHINDS_INDEX = 4

HOME_INDEX = 2
AWAY_INDEX = 3

def round_team_stats_index(s, r, team, stats):
    for index in range(len(stats)):
        line = stats[index]
        if line[SEASON_INDEX] == s and line[ROUND_INDEX] == r and line[TEAM_INDEX] == team:
            return index
    return None

def previous_team_stats_index(start_index, team, stats):
    index = start_index - 1
    while index >= 0:
        if stats[index][TEAM_INDEX] == team:
            return index
        index -= 1
    return None

def score(goals, behinds):
    return 6 * goals + behinds

def update_teams_seen(team, teams_seen):
    if team in teams_seen:
        teams_seen[team] = teams_seen[team] + 1
    else:
        teams_seen[team] = 1

def elo_for_team(team, elos):
    if team in elos:
        return elos[team]
    else:
        return 1200

def update_elos(winning_team, losing_team, elos):
    winning_elo = elo_for_team(winning_team, elos)
    losing_elo = elo_for_team(losing_team, elos)
    new_winning_elo, new_losing_elo = new_elo(winning_elo, losing_elo)
    elos[winning_team] = new_winning_elo
    elos[losing_team] = new_losing_elo

def new_elo(winner, loser):
    new_winner = winner + 32 * (1 - expected_score(winner, loser))
    new_loser = loser + 32 * (0 - expected_score(loser, winner))
    return (new_winner, new_loser)

def expected_score(first, second):
    return 1 / (1 + (10**((second-first) / 400)))

def build_inputs():
    matches = fu.read_csv(fu.csv_matches_path())
    matches = matches[1:]

    rounds = fu.read_csv(fu.csv_rounds_path())
    rounds = rounds[1:]

    emas = fu.read_csv(fu.csv_emas_path())
    emas = emas[1:]

    input_lines = []

    teams_seen = {}
    elos = {}

    for match in matches:
        s = match[SEASON_INDEX]
        r = match[ROUND_INDEX]
        home_team = match[HOME_INDEX]
        away_team = match[AWAY_INDEX]

        update_teams_seen(home_team, teams_seen)
        update_teams_seen(away_team, teams_seen)

        home_round_stats_index = round_team_stats_index(s, r, home_team, rounds)
        away_round_stats_index = round_team_stats_index(s, r, away_team, rounds)
        if home_round_stats_index == None or away_round_stats_index == None:
            continue

        home_ema_index = previous_team_stats_index(home_round_stats_index, home_team, emas)
        away_ema_index = previous_team_stats_index(away_round_stats_index, away_team, emas)
        if home_ema_index == None or away_ema_index == None:
            continue

        home_round_stats = rounds[home_round_stats_index]
        away_round_stats = rounds[away_round_stats_index]
        home_score = score(int(float(home_round_stats[GOALS_INDEX])), int(float(home_round_stats[BEHINDS_INDEX])))
        away_score = score(int(float(away_round_stats[GOALS_INDEX])), int(float(away_round_stats[BEHINDS_INDEX])))

        if home_score == away_score:
            continue

        if teams_seen[home_team] < 10 or teams_seen[away_team] < 10:
            continue

        winner = '1' if home_score > away_score else '0'
        home_emas = emas[home_ema_index][VALUES_START_INDEX:]
        away_emas = emas[away_ema_index][VALUES_START_INDEX:]

        input_lines.append([winner] + home_emas + [str(elo_for_team(home_team, elos))] + away_emas + [str(elo_for_team(away_team, elos))])

        if home_score > away_score:
            update_elos(home_team, away_team, elos)
        else:
            update_elos(away_team, home_team, elos)

    fu.write_csv(input_lines, fu.csv_inputs_path())
    print(elos)

if __name__ == "__main__":
    build_inputs()
