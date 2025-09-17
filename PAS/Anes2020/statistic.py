import csv
import numpy as np
import ollama
import pandas as pd
def ol(messages):
    stream = ollama.chat(
        model='llama3.3',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans

def sta(value):
    if value>=0 and value<20:
        return 0
    elif value>=20 and value<40:
        return 1
    elif value>=40 and value<60:
        return 2
    elif value>=60 and value<80:
        return 3
    elif value>=80 and value<100:
        return 4
    else:
        return 5
def main():
    folder_path = './Data/Anes2020/'
    input_path = folder_path + 'selected_anes2020.csv'
    output_path = folder_path + 'a20_descriptions_fsimulation.txt'
    df = pd.read_csv(input_path) 


    pelosi = [0,0,0,0,0,0]
    fauci = [0,0,0,0,0,0]
    rubio = [0,0,0,0,0,0]


    with open(input_path, mode='r', newline='') as infile,open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        #writer = csv.writer(outfile)
        for i, row in enumerate(reader):
        
            pelosi[sta(int(row['ftpelosi1']))] = pelosi[sta(int(row['ftpelosi1']))] + 1
            fauci[sta(int(row['ftfauci1']))] = fauci[sta(int(row['ftfauci1']))] + 1
            rubio[sta(int(row['ftrubio1']))] = rubio[sta(int(row['ftrubio1']))] + 1
            # outfile.write(prompt+'\n')
            
    pelosirate = [round(a/(i+1),3) for a in pelosi]
    faucirate = [round(a/(i+1),3) for a in fauci]
    rubiorate = [round(a/(i+1),3) for a in rubio]        
    print("pelosi:",pelosirate)
    print("fauci:",faucirate)
    print("rubio:",rubiorate)
    return

if __name__ == "__main__":
    main()