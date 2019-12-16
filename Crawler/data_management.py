import os
import json
from datetime import datetime
from Crawler.crawler_radiomics import load_study_data


def find_missing_filed(summary_file, log_file, out_file):

    # Load summary file
    summary = load_study_data(summary_file)

    # Load log file
    with open(log_file) as f:
        log = json.load(f)

    # Summary animals
    sum_animals = summary['Kirsch lab iD']
    sum_animals = ['K' + str(int(i)) for i in sum_animals]
    first_scan = list(summary['Date of MRI image    1st'])
    second_scan = list(summary['Date of MRI image     2nd'])
    sum_df = {sum_animals[i]: [first_scan[i], second_scan[i]] for i in range(len(sum_animals))}

    # Log animals
    log_animals = list(log.keys())

    # Get all animals names, OR
    all_animals = sum_animals + log_animals
    seen = set()
    unique = []
    for x in all_animals:
        if x not in seen:
            unique.append(x)
            seen.add(x)

    # Get overlap
    overlap_animals = [i for i in sum_animals if i in log_animals]
    overlap_animals = overlap_animals + [i for i in log_animals if i in sum_animals and not overlap_animals]

    # Get unique to summary
    sum_unique = [i for i in sum_animals if i not in log_animals]

    # Get unique to MR data
    log_unique = [i for i in log_animals if i not in sum_animals]

    # Write results to file
    n_dashes = 70
    f = open(out_file, 'w')
    f.write('Comparison of animals found in MR scans directories and the Summary Excel sheet.\n')
    f.write('Shown are animal IDs and the date of the first and second listed MR scans.\n')
    f.write('%d unique animal IDs have been found:\n' % len(unique))
    f.write('\t%d from the Summary\n\t%d from the MR data\n' % (len(sum_animals), len(log_animals)))
    f.write('-'*n_dashes + '\n\n')

    f.write('Animals found only in MR data (missing from Summary sheets): %d animals\n' % len(log_unique))
    f.write('\tID\t\t\tMR Data Dates\n')
    for an in log_unique:
        date1 = datetime.strptime(log[an]['StudyDate'][0], '%Y%m%d').date()
        try:
            date2 = datetime.strptime(log[an]['StudyDate'][1], '%Y%m%d').date()
        except:
            date2 = ''

        f.write('\t%s\t\t%s\t%s\n' % (an, date1, date2))
    f.write('\n' + '-' * n_dashes + '\n')

    f.write('Animals found only in Summary sheet (missing from MR data): %d animals\n' % len(sum_unique))
    f.write('\tID\t\t\tSummary Dates\n')
    for an in sum_unique:
        date1 = sum_df[an][0].date()
        try:
            date2 = sum_df[an][1].date()
        except:
            date2 = ''
        f.write('\t%s\t\t%s\t%s\n' % (an, date1, date2))
    f.write('\n' + '-' * n_dashes + '\n')

    f.write('Animals matched in both data sources: %d animals\n' % len(overlap_animals))
    f.write('\tID\t\t\tSummary Dates\t\t\t\tMR Data Dates\n')
    for an in overlap_animals:
        sdate1 = sum_df[an][0].date()
        try:
            sdate2 = sum_df[an][1].date()
        except:
            sdate2 = ''

        ldate1 = datetime.strptime(log[an]['StudyDate'][0], '%Y%m%d').date()
        try:
            ldate2 = datetime.strptime(log[an]['StudyDate'][1], '%Y%m%d').date()
        except:
            ldate2 = ''

        f.write('\t%s\t\t%s\t%s\t\t%s\t%s\n' % (an, sdate1, sdate2, ldate1, ldate2))
    f.write('\n' + '-' * n_dashes + '\n')

    f.write('All unique animal IDs from both sources: %d animals\n' % len(unique))
    f.write('\tID\t\tSummary\t\tMR Data\n')
    for an in unique:
        s = ''
        mr = ''
        if an in sum_animals: s = 'X'
        if an in log_animals: mr = 'X'
        f.write('\t%s\t\t   %s\t\t   %s\n' % (an, s, mr))
    f.write('\n' + '-' * n_dashes + '\n')

    f.close()


if __name__ == "__main__":

    # summary_file = '/media/justify/b7TData/Results/Summary.xlsx'
    # log_file = '/media/justify/b7TData/Results/processing_log.json'
    # out_file = '/media/justify/b7TData/Results/animal_matching.txt'

    summary_file = '/media/matt/Seagate Expansion Drive/b7TData/Results/Summary.xlsx'
    log_file = '/media/matt/Seagate Expansion Drive/b7TData/Results/processing_log.json'
    out_file = '/media/matt/Seagate Expansion Drive/b7TData/Results/animal_matching.txt'

find_missing_filed(summary_file, log_file, out_file)