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
    return value-1
    
def main():
    folder_path = './Data/GSS/'
    input_path = folder_path + '1.csv'
    output_path = folder_path + 'a20_descriptions_fsimulation.txt'
    df = pd.read_csv(input_path) 


    arms = [0,0,0]
    crime = [0,0,0]
    educ = [0,0,0]


    with open(input_path, mode='r', newline='') as infile,open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        #writer = csv.writer(outfile)
        for i, row in enumerate(reader):
        
            arms[sta(int(row['natarms']))] = arms[sta(int(row['natarms']))] + 1
            crime[sta(int(row['natcrime']))] = crime[sta(int(row['natcrime']))] + 1
            educ[sta(int(row['nateduc']))] = educ[sta(int(row['nateduc']))] + 1
            # outfile.write(prompt+'\n')
            
    armsrate = [round(a/(i+1),3) for a in arms]
    crimerate = [round(a/(i+1),3) for a in crime]
    educrate = [round(a/(i+1),3) for a in educ]        
    print("arms:",armsrate)
    print("crime:",crimerate)
    print("educ:",educrate)
    return

if __name__ == "__main__":
    main()