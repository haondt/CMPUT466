import requests, re, os, time, csv
def rand_id(minrank, maxrank, season=None):
    # Search url for random replay
    if season == None:
        season =''
    else:
        season = str(season)
    url = 'https://ballchasing.com/?title=&player-name=&playlist=13&season%s=&min-rank=%d&max-rank=%d&map=&replay-after=&replay-before=&upload-after=&upload-before=&roulette=1' % (season, minrank, maxrank)

    # get redirected url
    url = requests.get(url).url

    # extract id
    return re.search(r'/[^/]+$', url).group(0)[1:]

def download_csv(ident, path):
    # build csv fetch url
    url = 'https://ballchasing.com/dl/stats/players/%s/%s-players.csv' % tuple(ident for i in range(2))

    # save to file
    open(path, 'wb').write(requests.get(url).content)

def redownload_data():
    # Redownload files that were pulled too quickly, and just consist of a 403 error page
    ranks = [
        'bronze',
        'silver',
        'gold',
        'platinum',
        'diamond',
        'champion',
        'grand champion'
    ]
    divs = [1,2,3]


    # Go through folders
    for rank in ranks:
        for div in divs:
            d = '%s %d' % (rank, div)
            if rank == ranks[-1]:
                d = rank
            # The html pages are much larger than the csv files, find them and
            # re-download them
            for f in os.listdir(d):
                p = d + "/" + f
                if os.path.getsize(p) > 8000:
                    _id = f.split('.')[0]
                    download_csv(_id, p)
                    print(p)
                    time.sleep(5)
            if rank == ranks[-1]:
                break

def compile_files():
    # combine all individual csvs into one master file
    # Get all csv files in root folder
    files = [f for flist in [[os.path.join(root, _f) for _f in files] for root, dirs, files in os.walk('.')] for f in flist]
    files = [f for f in files if f.split('.')[-1] == 'csv']
    labels = None
    all_lines = []

    # Get each filename with folder, and add a column to the entry for rank
    for fname in files:
        rank = os.path.basename(os.path.dirname((fname)))
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            lines = [row for row in reader]

            if labels == None:
                labels = ['rank'] + lines[0]

            for row in lines[1:]:
                line = [rank] + row
                all_lines.append(line)

    # write all lines to combined csv
    with open('data_compiled.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(labels)
        for line in all_lines:
            writer.writerow(line)

def initial_download():
    # Download up to
    # ranks are expressed as numbers
    # 1 = bronze 1, 3 = bronze 3, 3 = silver 1, etc

    ranks = [
        'bronze',
        'silver',
        'gold',
        'platinum',
        'diamond',
        'champion'
    ]
    ranks = [(ranks[i//3] + ' ' + str((i%3)+1), i+1) for i in range(18)] + [('grand champion', 19)]

    for s in ranks:
        # grab the stats from a bronze game
        ids = {}
        # Grab 400 games from season 14 and 100 games from season 13,
        # for up to 500*6=3000 entries per rank
        for count_range in [(14, 0, 400), (13, 400, 500)]:
            j = 0
            count = count_range[1]
            while count < count_range[2]:
                time.sleep(2)
                _id = rand_id(s[1], s[1], season=count_range[0])
                if _id not in ids:
                    ids[_id] = None
                    time.sleep(2)
                    download_csv(_id, '%s/%s.csv' % (s[0], _id))
                    print(s[0], count, _id)
                    j = 0
                    count += 1
                else:
                    j += 1
                    print(s[0], count, j)
                    # Give up if we try 25 times without anything fresh
                    if j >= 25:
                        break


def main():
    initial_download()
    redownload_data()
    compile_files()


if __name__ == '__main__':
    main()


