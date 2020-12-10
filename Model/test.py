import csv

if __name__ == '__main__':
    tsv_file = open('../Dataset/covid_test_public.tsv', encoding="utf8")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        print(row[2])