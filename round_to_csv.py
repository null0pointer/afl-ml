import json
import file_utils as fu

list_key = 'lists'
team_details_key = 'team'
team_key = 'teamAbbr'
stats_key = 'stats'
totals_key = 'totals'
totals_keys = ['goals', 'behinds', 'kicks', 'handballs', 'marks', 'bounces', 'tackles', 'contestedPossessions', 'uncontestedPossessions', 'inside50s', 'marksInside50', 'contestedMarks', 'hitouts', 'onePercenters', 'disposalEfficiency', 'clangers', 'freesFor', 'freesAgainst', 'rebound50s']#, 'goalAssists', 'goalAccuracy']
clearances_key = 'clearances'
clearances_keys = ['centreClearances', 'stoppageClearances']

def is_all_zeroes(stats):
    for stat in stats:
        if not float(stat) == 0:
            return False
    return True

def round_json_to_csv():
    s = 2001
    r = 1

    lines = [['season', 'round'] + [team_key] + totals_keys + clearances_keys]
    while fu.round_exists(s, r):
        while fu.round_exists(s, r):
            season_round = [str(s), str(r)]
            with open(fu.raw_round_path(s, r)) as raw_data_file:
                json_data = json.load(raw_data_file)
                team_rounds = json_data[list_key]
                for team_round in team_rounds:
                    team = team_round[team_details_key]
                    team_name = [str(team[team_key])]
                    all_stats = team_round[stats_key]
                    totals = all_stats[totals_key]
                    stats = []
                    for key in totals_keys:
                        stats.append(str(totals[key]))
                    clearances = totals[clearances_key]
                    for key in clearances_keys:
                        stats.append(str(clearances[key]))

                    if not is_all_zeroes(stats):
                        line = season_round + team_name + stats
                        lines.append(line)

            r += 1
        s += 1
        r = 1

    fu.write_csv(lines, fu.csv_rounds_path())

if __name__ == "__main__":
    round_json_to_csv()
