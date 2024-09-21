"""
By: Anusha Umashankar 
Email:anusha.u.raju@gmail.com
Github: https://github.com/Anusha-raju
Date: 21-09-2024
"""


## imports 
import pandas as pd
from transformers import pipeline
import re




## patterns to find time, day and duration
time_pattern = r'\b(\d{1,2}-\d{1,2}:\d{2} ?(am|pm|AM|PM)?|\d{1,2}:\d{2} ?(am|pm|AM|PM)?-\d{1,2}:\d{2} ?(am|pm|AM|PM)?|\d{1,2}-\d{1,2} ?(am|pm|AM|PM)?)\b'

day_pattern = r'\b(monday|mon|tuesday|tue|wednesday|wed|thursday|thur|friday|fri|saturday|sat|sunday|sun|mondays|tuesdays|wednesdays|thursdays|fridays|saturdays|sundays)\b'
duration_pattern = r'\b(\d+\.\d+tx/wk|\d+\.\d+ hours/week)\b'


def has_numbers(inputString:str)->bool:
    """
    This function identifies if there is any number/digit in the sentence.
    Args:
    inputString (str): Input String
    Returns:
    bool: true if there is number/digit in the string.
    """
    return bool(re.search(r'\d', inputString))


#importing input file

patient_record_url = 'Patient Report - Treatment Plan Sample Data.xlsx'
patient_record_df = pd.read_excel(patient_record_url, engine='openpyxl')
#creating the output file
output_df = pd.DataFrame({'patient_id':[],'condition':[],'day':[],'time':[],'duration':[],'sentence': []})

#entity groups in consideration
t_list = ['SIGN_SYMPTOM', 'THERAPEUTIC_PROCEDURE', 'MEDICATION', 'DIAGNOSTIC_PROCEDURE', 'BIOLOGICAL_STRUCTURE', 'DETAILED_DESCRIPTION']

#pipeline to use the ner model
pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')
for data in patient_record_df['Interventions / Frequency:19']:
    clean_text = data.split('\n')
    sentences = []
    for sentence in clean_text:
        results = pipe(sentence)
        if results:
            for entity in results:
                if ('entity_group' in entity) and (entity['entity_group'] in t_list):
                    ##filtering all the elements from the sentences
                    if has_numbers(sentence):
                        patient_id = patient_record_df.index.get_loc(patient_record_df[patient_record_df['Interventions / Frequency:19'] == data].index[0])
                        new_row = {'patient_id':patient_id,'sentence':sentence}
                        matched_sentence=sentence
                        matches = re.findall(time_pattern, matched_sentence) 
                        if matches:
                            cleaned_matches = [match[0] for match in matches if match]
                            new_row['time'] = cleaned_matches[0]
                            matched_sentence = matched_sentence.replace(new_row['time'], "")
                        matches = re.findall(day_pattern, matched_sentence, flags=re.IGNORECASE)
                        if matches:
                            new_row['day'] = matches[0]
                            for mm in matches:
                                matched_sentence = matched_sentence.replace(mm, "")
                        matches = re.findall(duration_pattern, matched_sentence)
                        if matches:
                            new_row['duration'] = matches[0]
                            matched_sentence = matched_sentence.replace(matches[0], "")
                        cleaned_string = re.sub(r'[^a-zA-Z\s]', '', matched_sentence.strip())
                        for anyy in ['am','AM','pm','PM']:
                            if anyy in cleaned_string:
                                new_row['time'] = new_row['time'] + anyy
                                cleaned_string = cleaned_string.replace(anyy, "")
                                break

                        new_row['condition'] = cleaned_string.strip()

                        output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)
                    break

output_df.to_csv('output.csv', index=False)
