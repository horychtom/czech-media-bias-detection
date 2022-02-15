import csv

counter=0
with open('sents.txt','r') as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        if int(row[2]) > 1:
            counter+=1

print(counter)