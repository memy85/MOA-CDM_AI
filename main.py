import os
import subprocess

cdm_ai_script_list=[
    "./0_cohort/0_create_cohort_person_in_db.py",
    "./1_importsql/0_readDB.py",
    "./2_preprocessing_xgboost/0_preprocessing_xgboost.py",
    "./3_xgboost_classification/0_xgboost.py", 
    "./4_preprocessing_lstm/0_preprocessing_lstm.py",
    "./5_bi-lstm_attention_classification/0_lstm_attention.py"
]

cdm_data_analysis_list=[
    "./9_code_data_visualization/0_data_visualization.py"
]

def question_yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False
        
def main():
    print('main executed')
    for script in cdm_ai_script_list:
        if question_yes_or_no("Execute the '{}' script".format(script)):
            cwd = os.getcwd()
            filefullpath = os.path.abspath(script)
            os.chdir(os.path.dirname(filefullpath))
            command = 'python {}'.format(filefullpath)
            print(command)
            retcode = subprocess.call(command, shell=True)
            os.chdir(cwd)
            print("retcode : {}".format(retcode))
    
    for script in cdm_data_analysis_list:
        if question_yes_or_no("Execute the '{}' script".format(script)):
            cwd = os.getcwd()
            filefullpath = os.path.abspath(script)
            os.chdir(os.path.dirname(filefullpath))
            command = 'python {}'.format(os.path.abspath(script))
            print(command)
            subprocess.call(command, shell=True)
            os.chdir(cwd)
            print("retcode : {}".format(retcode))
    
if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
    