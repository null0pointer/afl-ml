import requests

TOKEN = '1973c9518bde816d8f41c3f3ea100406'

def get_round_url(y, r):
    url = 'http://www.afl.com.au/api/cfs/afl/statsCentre/teams'
    url = url + '?competitionId=' + season_id_for_year(y)
    url = url + '&roundId=' + round_id_for_year_and_round(y, r)
    return url

def get_match_url(y, r):
    url = 'http://www.afl.com.au/api/cfs/afl/matchItems/round/' + round_id_for_year_and_round(y, r)
    return url

def season_id_for_year(y):
    return 'CD_S' + str(y) + '014'

def round_id_for_year_and_round(y, r):
    r_string = '%02d' % r
    return 'CD_R' + str(y) + '014' + r_string

def raw_round_path_for_season_and_round(s, r):
    return 'data/raw/' + str(s) + '/rounds/' + 'round' + str(r) + '.json'

def raw_match_path_for_season_and_round(s, r):
    return 'data/raw/' + str(s) + '/matches/' + 'match' + str(r) + '.json'

def save_json_text(json_text, outpath):
    f = open(outpath, 'w+')
    f.write(json_text)
    f.close()

def fetch_data(token):
    for year in range(2001, 2017):
        round_num = 1
        while True:
            print('Fetching season ' + str(year) + ' round ' + str(round_num))

            r = requests.get(get_round_url(year, round_num), headers={'X-media-mis-token': TOKEN})
            if r.status_code == 404:
                break;
            round_outpath = raw_round_path_for_season_and_round(year, round_num)
            save_json_text(r.text, round_outpath)

            r = requests.get(get_match_url(year, round_num), headers={'X-media-mis-token': TOKEN})
            if r.status_code == 404:
                break;
            match_outpath = raw_match_path_for_season_and_round(year, round_num)
            save_json_text(r.text, match_outpath)

            round_num += 1

    print('Done.')

if __name__ == "__main__":
    fetch_data(TOKEN)
