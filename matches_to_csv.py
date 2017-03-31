import json
import file_utils as fu

matches_key = 'items'

match_key = 'match'
home_key = 'homeTeam'
away_key = 'awayTeam'
team_key = 'abbr'

def matches_json_to_csv():
    lines = [['season', 'round', 'home', 'away']]

    s = 2001
    r = 1

    while fu.match_exists(s, r):
        while fu.match_exists(s, r):
            season_round = [str(s), str(r)]
            with open(fu.raw_match_path(s, r)) as raw_data_file:
                json_data = json.load(raw_data_file)
                matches = json_data[matches_key]
                for match in matches:
                    teams = match[match_key]
                    home_team = teams[home_key]
                    home_team = home_team[team_key]
                    away_team = teams[away_key]
                    away_team = away_team[team_key]

                    lines.append(season_round + [home_team, away_team])

            r += 1
        s += 1
        r = 1

    fu.write_csv(lines, fu.csv_matches_path())

if __name__ == "__main__":
    matches_json_to_csv()
