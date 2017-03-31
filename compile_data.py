import round_to_csv as r2csv
import matches_to_csv as m2csv
import calculate_emas as emas
import build_inputs as bi

def compile_data():
    print('- Converting round JSON to CSV')
    r2csv.round_json_to_csv()
    print('- Converting match JSON to CSV')
    m2csv.matches_json_to_csv()
    print('- Calculating EMAs')
    emas.calculate_emas()
    print('- Building model inputs')
    bi.build_inputs()
    print('Done.')

if __name__ == "__main__":
    compile_data()
