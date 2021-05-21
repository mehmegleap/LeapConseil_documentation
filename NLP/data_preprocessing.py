from utils import *
from selenium import webdriver
from bs4 import BeautifulSoup


def process_scientific_words_data(data_path=scientific_path, saving_file_name='result_scientific_words_only.txt'):
    """
    Args:
        data, str, path to the scientific data words
    Returns:
        output file
    """
    with open(data_path, encoding='utf-8') as f:
        lines = f.readlines()
    data = ''.join(lines)
    data = convert_data_to_lower_case(data)
    data = remove_numbers(data)
    data = remove_punctuation(data)
    data = remove_stop_words(data)
    data = remove_apostrophe(data)
    data = stemming(data)
    data = remove_duplicates(data)
    data = remove_stop_words(data)
    with open(f'{saving_data_path}/{saving_file_name}' , "a+", encoding='utf-8') as f:
        f.write(data)

def process_abreviation_words_data(data_path=scientific_path, saving_file_name='result_abreviation.txt'):
    with open(data_path, encoding='utf-8') as f : 
        lines = f.readlines()
    data = ''.join(lines)
    data = convert_data_to_lower_case(data)
    data = remove_duplicates(data)
    with open(f'{saving_data_path}/{saving_file_name}' , "a+", encoding='utf-8') as f:
        f.write(data)

def extract_data_from_website():
    op = webdriver.ChromeOptions()
    op.add_argument('headless')
    driver = webdriver.Chrome("./chromedriver",options=op)
    with open("web_dataset.txt" , "a+") as web:
        for i in range(2400,2570):#2570
            print(i)
            driver.get("http://dictionnaire.academie-medecine.fr/index.php?q=&page=" + str(i))
            content = driver.page_source
            soup = BeautifulSoup(content)
            web.write(soup.find('div', attrs={'class':'screenContainer'}).text + " ")
            
def extract_arborescence_cim10(data_path=chapitre_19_path):
    """
    Extract arborescence from chap19 for cim10
    """
    with open('datasets/chapitre19.txt', 'r', encoding='utf-8') as f:
        chap19 = f.readlines()
    ramq_data = pd.read_excel(chapitre_19_RAMQ_path)
    chapter_num = '19'
    chapter_name = 'Lésions traumatiques, empoisonnements et certaines autres conséquences de causes externes'
    column_list = ['Chapitre_num', 'Chapitre_nom', 'Categorie', 'Sous_Categorie','Sous_Sous_Categorie', 'Sous_Sous_Sous_Categorie', 'diagnostic_num', 'diagnostic']
    r = re.compile(r"([A-Z]\d+\-[A-Z]\d+.*)|([A-Z]\d\d .*)|([A-Z]\d\d\d .*)")
    results = []
    for line in chap19:
        results.append(r.findall(line))
    tuples = []
    categorie = ''
    sous_categorie = ''
    sous_sous_categorie = ''
    arbo_df = pd.DataFrame(columns = column_list)
    for i in range(len(results)):
        if results[i][0][0] != '':
            categorie = results[i][0][0]
        elif results[i][0][1] != '':
            sous_categorie = results[i][0][1]
            if i+1 < len(results) and results[i+1][0][2] == '' :
                sous_sous_categorie = ''
                arbo_df = arbo_df.append(get_arbo_chap_19_df(ramq_data=ramq_data, categorie=categorie, sous_categorie=sous_categorie, sous_sous_categorie=sous_sous_categorie, chapter_num=chapter_num, chapter_name=chapter_name, column_list=column_list))
                tuples.append((categorie, sous_categorie, sous_sous_categorie))
        elif results[i][0][2]!='' :
            sous_sous_categorie = results[i][0][2]   
            arbo_df = arbo_df.append(get_arbo_chap_19_df(ramq_data=ramq_data, categorie=categorie, sous_categorie=sous_categorie, sous_sous_categorie=sous_sous_categorie, chapter_num=chapter_num, chapter_name=chapter_name, column_list=column_list))
            tuples.append((categorie, sous_categorie, sous_sous_categorie))
    arbo_df.to_csv(f'{saving_data_path}/CIM_10_Chapitre_19.csv')
    
def main_processing(process=0):
    """
    Main to execute the preprocessing
    Args:
        process, int, design which processing to run
        0 -> process_scientific_words_data
        1 -> process_abreviation_words_data
        2 -> extract_data_from_website
        3 -> main_extract_diagnocstic
        4 -> extract_arborescence_cim10
    """
    if process == 0:
        process_scientific_words_data(data_path=scientific_path, saving_file_name='result_scientific_words_only.txt')
    if process == 1:
        process_abreviation_words_data(data_path=scientific_path, saving_file_name='result_abreviation.txt')
    if process == 2:
        extract_data_from_website()
    if process == 3:
        main_extract_diagnostic(data=data, saving_file='new_processing_tuples_result')
        cleaned_diagnostic_data(data_path=f'{saving_data_path}/new_processing_tuples_result.pck')
    if process == 4:
        extract_arborescence_cim10(data_path=chapitre_19_path)

if __name__ == "__main__":
    # main_processing(process=0)
    # main_processing(process=1)
    # main_processing(process=2)
    main_processing(process=3)
    #main_processing(process=4)
    