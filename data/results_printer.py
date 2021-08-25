import csv

with open("Random  Data/PSNR Results.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    separator = " & "
    for row in csv_reader:
        print(separator.join(row), " \\", "\\")
        print("\\hline")
