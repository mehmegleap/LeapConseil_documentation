import nltk
import os
import string
import pickle
import re
import unidecode
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import Counter
from num2words import num2words

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.snowball import FrenchStemmer

from tqdm import tqdm

# Path definitions
saving_data_path = 'processed_datasets'
scientific_path = 'datasets/raw_scientificwords_dataset.txt'
abreviation_path = 'dataset/Abréviation.txt'
diagnostic_data_path = 'datasets/diagnostics.xlsx'
chapitre_19_path = 'datasets/chapitre19.txt'
chapitre_19_RAMQ_path = 'datasets/CIM_19_RAMQ.xlsx'
chapitre_13_19_path = 'datasets/CIM_10_arbo_Chapitre_13_19.xlsx'
processed_data_for_tuple = 'processed_datasets/data_colones_membres_cotes_var_glob.pck'

def convert_data_to_lower_case(d):
    return d.lower()

def remove_punctuation(d):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n,\xa0®"    
    for s in range(len(symbols)):
        d = d.replace(symbols[s],' ')      
    return d

def remove_numbers(d):
    remove_digits = str.maketrans('', '', '0123456789')
    d = d.translate(remove_digits)
    d = d.replace('  ',' ')
    return d

def remove_apostrophe(d):
    d = d.replace("'", " ")
    d= d.replace("’"," ")
    return d

def remove_stop_words(d):
    stop_words = stopwords.words('french')
    words = word_tokenize(str(d))
    new_text = ""
    for w in tqdm(words):
        if w not in stop_words and len(w) > 2:            
            new_text = new_text + w + " "
    return new_text

def stemming(d):
    stemmer= FrenchStemmer(ignore_stopwords=True)    
    words = word_tokenize(str(d))
    new_text = ""
    for w in tqdm(words):
        new_text = new_text + stemmer.stem(w) + " "
    return new_text

def remove_duplicates(d):
    words = word_tokenize(str(d))
    new_text = ""    
    for w in tqdm(set(words)):
        new_text = new_text + w + " "  
    return new_text

def load_data(data_path=diagnostic_data_path):
    """
    Load diagnocstic data
    """
    data = pd.read_excel(data_path)
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    descriptions = data['DescriptionDiagnostic']
    descriptions_nan_idx = list(np.where(descriptions.isna() == True)[0])
    data.drop(data.index[descriptions_nan_idx], inplace=True)
    colones = data.columns.values[1:]
    return data, colones

def load_pickle_dataset(data_path):
    """
    Use this to load all the pickle dataset
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_members_for_regex(data):
    membres = data.columns.values[1:]
    membres = [el.split('IndLesion')[-1] for el in membres]
    membres = [el.split('Gauche')[0] for el in membres]
    membres = [el.split('Droit')[0] for el in membres] # va faire droite et droit
    #membres.extend(['avant bras', 'avant-bras', 'doigts', 'orteils'])
    #membres.remove('DoigtsMain')
    #membres.remove('OrteilsPied')
    membres.remove('Cuissse')
    membres.remove('AutrePartieCorps')
    membres.remove('PrecisionLesionAutrePartieCorps')
    membres = [convert_data_to_lower_case(el) for el in membres]
    membres = np.unique(membres)
    cote_gauche = ['\\bgauche\\b', '\\bg\\b', '\\bgch\\b', '\\bgche\\b', '\\bleft\\b',] 
    cote_droit = ['\\bdroit\\b', '\\bdroite\\b', '\\bd\\b', '\\bdt\\b', '\\bdte\\b', '\\bright\\b',]
    # cotes = [r'\bgauche\b', r'\bdroit(e)?\b', r'\bg\b', r'\bd\b', r'\bdrte\b', r'\bgche\b', r'\bdte\b', 
    #          r'\bgch\b', r'\bright\b', r'\bleft\b']
    return list(membres), cote_gauche, cote_droit

def extract_membre_and_position_from_columns(col_name, line):
    """
    Recoit juste les columns names et retourne les membres et le cote quand c'est disponible
    Args:
        col_name, str,
        line, str
    """
    col_name = col_name.lower()
    col_name = col_name.split('indlesion')[-1]
    if col_name.find('gauche') != -1:
        return col_name.split('gauche')[0], 'gauche'
    elif col_name.find('droit') != -1:
        return col_name.split('droit')[0], 'droit'
    elif col_name == 'autrepartiecorps':
        # pour retrouver l'index suivant np.where(colones=='PrecisionLesionAutrePartieCorps')[0] == 13 + 1 == 14
        return f'Autre partie: {line[14]}', 'inconnu'
    else:
        return col_name, 'inconnu'

def remove_tag_from_words(mot):
    """
    Remove the \\b from the words if they still exist
    """
    if mot.find('\\b') != -1:
        mot = mot.replace('\\b', '')
    return mot

def extract_variables_for_diagnostic_detection():
    data, colones = load_data()
    membres_var_glob, cote_gauche_var_glob, cote_droit_var_glob = extract_members_for_regex(data=data)
    # for liste in all_variations_list:
    #     membres_var_glob.extend(liste)
    saving_dict = {'data': data, 
                   'colones': colones, 
                   'membres_var_glob': membres_var_glob, 
                   'cote_gauche_var_glob': cote_gauche_var_glob,
                   'cote_droit_var_glob': cote_droit_var_glob}
    with open(f'{saving_data_path}/data_colones_membres_cotes_var_glob.pck', 'wb') as f:
        pickle.dump(saving_dict, f)
    return data, colones, membres_var_glob, cote_gauche_var_glob, cote_droit_var_glob

def regex_type_1(line, mot):
    """ Match pour juste les cas ou on a un seul diagnostic par phrase 
    Returns:
        (sentence, membre, cote) if available
        (inconnu, membre , inconnu) if not available
        type of return is a tuple
    """
    replacement_word = ''
    if mot == 'abdomen':
        mot = '|'.join(variation_abdomen)
        replacement_word = 'abdomen'    
    if mot == 'avantbras':
        mot = '|'.join(variation_avant_bras)
        replacement_word = 'avant bras'
    if mot == 'bassin':
        mot = '|'.join(variation_bassin)
        replacement_word = 'bassin'
    if mot  == 'bras':
        mot = '|'.join(variation_bras)
        replacement_word = 'bras'    
    if mot  == 'cervical':
        mot = '|'.join(variation_cervical)
        replacement_word = 'cervical'    
    if mot == 'cheville':
        mot = '|'.join(variation_cheville)
        replacement_word = 'cheville'
    if mot  == 'coude':
        mot = '|'.join(variation_coude)
        replacement_word = 'coude'    
    if mot == 'crane':
        mot = '|'.join(variation_crane)
        replacement_word = 'crane'
    if mot == 'cuisse':
        mot = '|'.join(variation_cuisse)
        replacement_word = 'cuisse'    
    if mot == 'dents':
        mot = '|'.join(variation_dents)
        replacement_word = 'dents'   
    if mot  == 'doigtsmain':
        mot = '|'.join(variation_doigt)
        replacement_word = 'doigt'  
    if mot  == 'dorsal':
        mot = '|'.join(variation_dorsal)
        replacement_word = 'dorsal'  
    if mot == 'epaule':
        mot = '|'.join(variation_epaule)
        replacement_word = 'epaule'
    if mot == 'genou':
        mot = '|'.join(variation_genou)
        replacement_word = 'genou'
    if mot == 'bassin':
        mot = '|'.join(variation_hanche)
        replacement_word = 'hanche'
    if mot == 'jambe':
        mot = '|'.join(variation_jambe)
        replacement_word = 'jambe'
    if mot == 'lombaire':
        mot = '|'.join(variation_lombaire)
        replacement_word = 'lombaire'
    if mot == 'main':
        mot = '|'.join(variation_main)
        replacement_word = 'main'
    if mot  == 'oreilles':
        mot = '|'.join(variation_oreille)
        replacement_word = 'oreille'
    if mot == 'orteilspied':
        mot = '|'.join(variation_orteil)
        replacement_word = 'orteil'
    if mot == 'pied':
        mot = '|'.join(variation_pied)
        replacement_word = 'pied'
    if mot == 'poignet':
        mot = '|'.join(variation_poignet)
        replacement_word = 'poignet'
    if mot == 'sacre':
        mot = '|'.join(variation_sacre)
        replacement_word = 'sacre'
    if mot == 'thorax':
        mot = '|'.join(variation_thorax)
        replacement_word = 'thorax'
    if mot == 'visage':
        mot = '|'.join(variation_visage)
        replacement_word = 'visage'
    if mot == 'yeux':
        mot = '|'.join(variation_yeux)
        replacement_word = 'yeux'
        
    cotes_droit = '|'.join(cote_droit_var_glob)
    cotes_gauche = '|'.join(cote_gauche_var_glob)
    pattern_mot = f'(.*[a-zA-Zèéùêëçîïûôœö\d]+\s*({mot})\s*)'
    match = re.search(pattern_mot, line, re.U|re.M|re.I)
    if match is None:
        return ('inconnu', mot, 'inconnu')
    else: # le mot existe dans la phrase on va chercher le cote maintenant
        match = list(match.groups())
        pattern_mot_cote_gauche = f'(.*[a-zA-Zèéùêëçîïûôœö\d]+\s*({mot})\s*({cotes_gauche}))'
        pattern_mot_cote_droit = f'(.*[a-zA-Zèéùêëçîïûôœö\d]+\s*({mot})\s*({cotes_droit}))'
        match_gauche = re.search(pattern_mot_cote_gauche, line, re.U|re.M|re.I)
        match_droit = re.search(pattern_mot_cote_droit, line, re.U|re.M|re.I)
        if match_gauche is not None: match_gauche = list(match_gauche.groups())
        else: match_gauche = [None]
        if match_droit is not None: match_droit = list(match_droit.groups())
        else: match_droit = [None]
        el_in_match_gauche = [el for el in match_gauche[2:] if el is not None]
        el_in_match_droit = [el for el in match_droit[2:] if el is not None]
        temp_mot_gauche = False
        temp_mot_droit = False
        if el_in_match_gauche != []:
            for el in el_in_match_gauche:
                if el in ['gauche', 'g', 'gch', 'gche', 'left']: temp_mot_gauche = True
        if el_in_match_droit != []:
            for el in el_in_match_droit:
                if el in ['droit', 'droite',  'd', 'dt', 'dte', 'right']: temp_mot_droit = True
                    
        if temp_mot_gauche == False and temp_mot_droit == False:# on a pas retrouvé les cotes avec nos regex
            if replacement_word != '':
                return (match[0], replacement_word , 'inconnu')
            else:
                #mot = remove_tag_from_words(mot)
                return (match[0], mot , 'inconnu')
        if temp_mot_gauche == True and temp_mot_droit == False: # on a match le cote gauche
            if replacement_word != '':
                return (match_gauche[0], replacement_word , 'gauche')
            else:
                #mot = remove_tag_from_words(mot)
                return (match_gauche[0], mot , 'gauche')
        if temp_mot_gauche == False and temp_mot_droit == True: # on a match le cote gauche
            if replacement_word != '':
                return (match_droit[0], replacement_word , 'droit')
            else:
                #mot = remove_tag_from_words(mot)
                return (match_droit[0], mot , 'droit')
        if temp_mot_gauche == True and temp_mot_droit == True: # on a match le cote gauche
            if replacement_word != '':
                return (match[0], replacement_word , 'droit/gauche')
            else:
                #mot = remove_tag_from_words(mot)
                return (match[0], mot , 'droit/gauche')
            
def regex_type_2(line, mots):
    """ Match pour juste les cas ou on a plus d'un diagnostic par phrase
    Args:
        line: str, the sentence 
        mots: list of str, of members to find in the sentence
    Returns:
        (sentence, membre, cote) if available
        (inconnu, membre , inconnu) if not available
        type of return is a list of tuple
    """
    original_line = deepcopy(line)
    temp_list_solutions = []
    for idx, mot in enumerate(mots):
        temp_list_solutions.append(regex_type_1(line=line, mot=mot))
        # eliminer le premier match de la phrase comme ca on peut aller chercher le 2eme
        line = line.replace(list(temp_list_solutions[idx])[0], '') 
        if line == '':
            temp_list_solutions.append(regex_type_1(line=original_line, mot=mot))
    return temp_list_solutions

def regex_type_3(line):
    """ Match pour les cas ou on ne sait rien au préalable
    Args:
        line: str, the sentence 
        mots: list of str, of members to find in the sentence
    Returns:
        (sentence, membre, cote) if available
        (sentence, inconnu , inconnu) if not available
        type of return is a list of tuple
    """
    temp_list_solutions = []
    for membre in membres_var_glob:
        temp_list_solutions.append(regex_type_1(line=line, mot=membre))    
    return temp_list_solutions 

variation_abdomen = ['\\babdo\\b', '\\babdomen\\b', '\\babdominal\\b', '\\babdominaux\\b', '\\binguinal hernia\\b', 
                     '\\bhernie(s)? inguinale(s)?\\b', '\\bgastro(-| )?enterite(s)?\\b', '\\benterite(s)?\\b', '\\bgastro\\b', 
                     '\\bcostale(s)?\\b', '\\bombilicale(s)?\\b']
variation_avant_bras = ['\\bavant(s| |-)?bras\\b', '\\bav bras\\b', '\\bmembre(s)? superieur(s)?\\b', '\\bsupinator(-| )?muscle(s)?\\b', 
                        '\\bsupinateur\\b', '\\bepicondylite(s)?\\b', '\\bcervico(-| )?brachialgie(s)?\\b', '\\bbrachialgie\\b', '\\bepicondylien(s)?\\b', 
                        '\\bforearm\\b']
variation_bassin = ['\\bbassin(s)?\\b', '\\bcoccyx\\b', '\\bfesse(s)?\\b', '\\bischion(s)?\\b', '\\bsacro(-| )?iliaque(s)?\\b', '\\bsacro-ileite(s)?\\b',
                    '\\bsacrum\\b', '\\bpiriforme\\b', '\\bpsoas\\b']
variation_bras = ['\\bbras\\b', '\\bbrachialgi(e)?(s)?\\b', '\\btrapez(e)?(s)?\\b', '\\bbrachial(e)?(s)?\\b', '\\btriceps\\b', 
                  '\\bbiceps\\b', '\\bmembre(s)? superieur(s)?\\b', '\\bcoiffe(des| |-)?rotateur(s)?\\b']
variation_cervical = ['\\bcervical(e)?(s)?\\b', '\\bcervico(s)?\\b', '\\bcervico(-| )?dorsale(s)?\\b', '\\btcc(l)?\\b', '\\bcervicalgie(s)?\\b', 
                      '\\bcervico(-| )?brachialgie(s)?\\b', '\\bcephale(e)(s)?\\b', '\\bcervicogene(s)?\\b']
variation_cheville = ['\\bcheville(s)?\\b', '\\bchevlle\\b', '\\bmalleole(s)?\\b', '\\btallon(s)?\\b', '\\btendon(s| )?achille(s)?\\b']
variation_coude = ['\\bocude\\b', '\\bcoude(s)?\\b', '\\bepicondylite(s)?\\b', '\\bepicondilitis\\b']
variation_crane = ['\\bcrane(s)?\\b', '\\bcerebral(e)?\\b', '\\bcephale(e)(s)?\\b', '\\bfront(al)?(e)?(s)?\\b', '\\borbite(s)?\\b', '\\bhead\\b', '\\btete(s)?\\b', 
                   '\\bcranien(ne)?(nes)?(s)?\\b', '\\bcranio\\b', '\\btcc(l)?\\b', '\\bmenton(s)?\\b', '\\bcommotion(s| )?cerebrale(s)?\\b'] 
variation_cuisse = ['\\bcuisse(s)?\\b', 'ischio(-| )?jambier(s)?\\b', '\\bmembre(s)?(-| )?inferieur(s)?\\b', 
                    '\\btrochanterien(ne)?(s)?\\b', '\\bfemur(s)?\\b', '\\bhamstring(s)?\\b', '\\bquadricep(s)?\\b', 
                    '\\bsciatalgie(s)?\\b', '\\bsciatique(s)?\\b', '\\blombo(-| )?sciatique(s)?\\b', '\\blombo(-| )?sciatalgie(s)?\\b']
variation_dents = ['\\bdent(s)?\\b', '\\bincisive(s)?\\b', '\\bmachoire(s)?\\b', '\\bmollaire(s)?\\b', '\\bdentaire(s)?\\b', '\\bcanine(s)?\\b']
variation_doigt = ['\\bdoigt(s)?\\b', '\\bongle(s)?\\b', '\\bdoigtsmain\\b', '\\bpouce(s)?\\b', '\\bindex\\b', '\\bannulaire(s)?\\b', '\\bauriculaire(s)?\\b', 
                   '\\bfinger(s)?\\b']
variation_dorsal = ['\\bdorsal(e)?(s)?', '\\bdorso\\b', '\\bdorsalgie(s)?\\b', '\\bdorso(-| )?lombaire(s)?\\b', '\\brhomboide(s)?\\b', 
                    '\\bcervico(-| )?dorsale(s)?\\b', '\\btrapeze(s)?\\b', '\\bparadorsale(s)?\\b', '\\bdorsal(-| )?lumbar\\b']
variation_epaule = ['\\bepaule(s)?\\b', '\\btrapeze(s)?\\b', '\\bplexus(-| )?brachial\\b', '\\bcervico(-| )?brachialgie\\b', '\\brotateur(s)?\\b', 
                    '\\bshoulder(s)?\\b', '\\bdeltoide(s)?\\b']
variation_genou = ['\\bgenou(x)?\\b']
variation_hanche = ['\\bhanche(s)?\\b', '\\badducteur(s)?\\b', '\\binguinal(-| )?hernia\\b', '\\bhernie(s)?(-| )?inguinal(e)?(s)?\\b', '\\bligament(s)?(-| )?inguinal(e)?(s)?\\b',
                   '\\btrochanterien(ne)?(s)?\\b', '\\baine(s)?\\b']
variation_jambe = ['\\bjambe(s)?\\b', '\\bperone(s)?\\b', '\\btibia(le)?(s)?\\b', '\\bmollet(s)?\\b', '\\bgastrocnemien(s)?\\b',
                   '\\bsolaire(s)?\\b', '\\bplantaire(s| )?grele(s)?\\b']
variation_lombaire = ['\\bdorso(-| )?lombaire\\b', '\\bdorsal(-| )?lumbar\\b', '\\blombaire(s)?\\b', '\\blumbar(s)?\\b', 
                      '\\bbas(du| )?dos\\b', '\\bhernie(s| )?discale(s)?\\b'] # 'hernie(s)?'
variation_main = ['\\btunnel(s)? carpien(s)?\\b', '\\bmain\\b', '\\bphalange(s)?\\b', '\\bphalangienne(s)?\\b', '\\bphalange(s)? distale(s)?\\b', 
                  '\\bphalange(s)? intermediaire(s)?\\b', '\\bphalange(s)? proximal(e)?(s)?\\b', '\\bmetacarpe(s)?\\b', 
                  '\\bmetacarpo(s)?\\b', '\\bcarpe(s)?\\b', '\\bcarpo(s)?\\b'] # NB:tant que poignet ou doigt sont touche main aussi
variation_oreille = ['\\boreille(s)?\\b', '\\bsurdite(s)?\\b', '\\bacouphene(s)?\\b', '\\bperte(s)? auditive(s)?\\b', '\\batteinte(s)? auditive(s)?\\b']
variation_orteil = ['\\borteil(s)?\\b', '\\borteilspied(s)?\\b']
variation_pied = ['\\bpied(s)?\\b', 'talon(s)?', '\\bplantaire(s)?\\b']
variation_poignet = ['\\b(de)?quervain\\b', '\\bpoignet(s)?\\b', '\\bstyloide(s)? cubitale(s)?\\b', '\\bcubitale(s)?\\b']
variation_sacre = ['\\blombo(-| )?sacre(e)?(s)?\\b', '\\bsacro\\b', '\\bsacre(e)?\\b', '\\bsacrum\\b']
variation_thorax = ['\\bthorax\\b', '\\bthoracique(s)?\\b', '\\bhemithorax\\b', '\\bcostale(s)?\\b']
variation_visage = ['\\bvisage\\b', '\\bfacial\\b', '\\bnez\\b', '\\barcade(s)? sourcilliere(s)?\\b', '\\barcade(s)? sourcil(s)?\\b\\b',
                    '\\bjoue(s)?\\b', '\\bsourcil(s)?\\b', '\\bnasale(s)?\\b']
variation_yeux = ['\\byeux\\b', '\\boeil\\b', '\\bocculaire(s)?\\b', '\\beye(s)?\\b']
all_variations_list = [variation_abdomen, variation_avant_bras, variation_bassin, variation_bras, 
                       variation_cervical, variation_cheville, variation_coude, variation_crane, 
                       variation_cuisse, variation_dents, variation_doigt, variation_dorsal, variation_epaule, 
                       variation_genou, variation_hanche, variation_jambe, variation_lombaire, variation_main, 
                       variation_oreille, variation_orteil, variation_pied, variation_poignet, variation_sacre, 
                       variation_thorax, variation_visage, variation_yeux]
# data, colones, membres_var_glob, cote_gauche_var_glob, cote_droit_var_glob = extract_variables_for_diagnostic_detection()

all_data = load_pickle_dataset(data_path=processed_data_for_tuple)
data = all_data['data']
colones = all_data['colones']
membres_var_glob = all_data['membres_var_glob']
cote_gauche_var_glob = all_data['cote_gauche_var_glob']
cote_droit_var_glob = all_data['cote_droit_var_glob']

def main_extract_diagnostic(data=data, saving_file='processing_tuples_result'):
    """
    Main function that compile the extraction of the diagnostic into the tuple
    """
    saving_list = []
    for line in tqdm(data.values):
        diagnostic_sentences = line[0]
        # je retrouve des phrase qui ont juste '1' comme str dans la description: on va juste faire ceci pour l'instant
        if len(word_tokenize(str(diagnostic_sentences))) <= 1: continue
        diagnostic_sentences = unidecode.unidecode(diagnostic_sentences).lower()
        diagnostic_sentences = remove_punctuation(d=diagnostic_sentences)
        membres_idx = list(np.where(line[1:] == True)[0])
        if len(membres_idx) == 1:
            #print(f'Sentence enterring Pattern 1 : {diagnostic_sentences}')
            membre_touche = colones[membres_idx[0]]
            membre_extrait, cote_extrait = extract_membre_and_position_from_columns(col_name=membre_touche, line=line)
            if membre_extrait.startswith('Autre partie:'):
                saving_list.append((diagnostic_sentences, membre_extrait, cote_extrait))
            else:
                #print(f'Sortie Pattern 1 : {regex_type_1(line=diagnostic_sentences, mot=membre_extrait)}')
                saving_list.append(regex_type_1(line=diagnostic_sentences, mot=membre_extrait))
        elif len(membres_idx) > 1:
            #print(f'Sentence enterring Pattern 2 : {diagnostic_sentences}')
            membres_mots = []
            for pos in membres_idx:
                membre_touche = colones[pos]
                membre, _ = extract_membre_and_position_from_columns(col_name=membre_touche, line=line)
                membres_mots.append(membre)
            membres_mots = [el for el in membres_mots if el.startswith('Autre partie') is False]
            #print(f'Sortie Pattern 2 : {regex_type_2(line=diagnostic_sentences, mots=membres_mots)}')
            saving_list.extend(regex_type_2(line=diagnostic_sentences, mots=membres_mots))
        else:
            print(f'Sentence enterring Pattern 3 : {diagnostic_sentences}')
            print(f'Sortie Pattern 3 : {regex_type_3(line=diagnostic_sentences)}')
            saving_list.extend(regex_type_3(line=diagnostic_sentences))
            
    saving_list = [el for el in saving_list if list(el)[0] != 'inconnu']
    with open(f'processed_datasets/{saving_file}.pck', 'wb') as f:
        pickle.dump(saving_list, f)
    return saving_list

def cleaned_diagnostic_data(data_path=f'{saving_data_path}/processing_tuples_result.pck'):
    tuples = load_pickle_dataset(data_path=data_path)
    with open(f'{saving_data_path}/scientificWords_corpus.txt', encoding='utf-8') as f:
        scientificWords = word_tokenize(f.readline())
    new_tuple = []
    stemmer= FrenchStemmer(ignore_stopwords=False)
    for tup in tqdm(tuples):
        new_discription = ''
        words_in_discription = word_tokenize(tup[0])    
        for word in words_in_discription:
            word.encode('utf-8')
            if stemmer.stem(word.lower()) in scientificWords:
                new_discription = new_discription + word + " "
        new_tuple.append((new_discription, tup[1], tup[2]))
    with open(f'{saving_data_path}/cleaned_tuples.pck' , 'wb') as f:
        pickle.dump(new_tuple, f)
        
def get_arbo_chap_19_df(ramq_data, categorie, sous_categorie, sous_sous_categorie, chapter_num, chapter_name, column_list):
    """
    Get arborescence for CIM10 chapter 19
    Args:
        ramq_data, pd.dataframe
        categorie, str,
        sous_categorie, str,
        sous_sous_categorie,  str,
    Returns:
        arbo_df_tmp, pd.dataframe
    """
    arbo_df_tmp = pd.DataFrame(columns = column_list)
    find_result = None    
    if sous_sous_categorie != '' :   
        find_result = ramq_data[ramq_data['diagnostic_num'].str.contains(sous_sous_categorie.split(' ')[0])].reset_index(drop=True)
    else :        
        find_result = ramq_data[ramq_data['diagnostic_num'].str.contains(sous_categorie.split(' ')[0])].reset_index(drop=True)
    for i in range(len(find_result)):        
        tuple_list = [chapter_num , chapter_name , categorie , '', sous_categorie, sous_sous_categorie, '' , '']
        tuple_list[6] = find_result.loc[i][0]
        tuple_list[7] = find_result.loc[i][1]
        tmp_df = pd.DataFrame([tuple_list], columns = column_list)
        arbo_df_tmp = arbo_df_tmp.append(tmp_df)
    return arbo_df_tmp


def process_string(d, with_stemming=True):
    """
    Args:
        d, str
        with_stemming, bool true if we want to apply the stemming, false otherwise
    """
    d = unidecode.unidecode(d) 
    d = remove_numbers(d)
    d = convert_data_to_lower_case(d)
    d = remove_punctuation(d)
    d = remove_stop_words(d)
    d = remove_apostrophe(d)
    if with_stemming:
        d = stemming(d)
    d = remove_stop_words(d)
    return d

def process_string_with_duplicates_elimination(d, with_stemming=True):
    """
    Args:
        d, str
        with_stemming, bool true if we want to apply the stemming, false otherwise
    """
    d = unidecode.unidecode(d)
    d = remove_numbers(d)
    d = convert_data_to_lower_case(d)
    d = remove_punctuation(d)
    d = remove_stop_words(d)
    d = remove_apostrophe(d)
    if with_stemming:
        d = stemming(d)
    d = remove_duplicates(d)
    d = remove_stop_words(d)
    return d


def load_cim10_chap_13_19(data_path=chapitre_13_19_path, remove_duplicate=False, with_stemming=True):
    """
    Load the cim10 chap 13 and 19
    Args:
        data_path, str, path to the cim10 chap 13 and 19
        remove_duplicate, bool if false stay like that if true remove duplicate words
    """
    df_tmp = pd.read_excel(data_path)
    columns_list = df_tmp.columns
    df_tmp = df_tmp[columns_list[1]].map(str) + " " + df_tmp[columns_list[2]].map(str) + " " + df_tmp[columns_list[3]].map(str) + " " + df_tmp[columns_list[4]].map(str) + " " + \
              df_tmp[columns_list[5]].map(str) + " " + df_tmp[columns_list[5]].map(str) + " " + df_tmp[columns_list[6]].map(str) + " " + df_tmp[columns_list[7]].map(str)
    data_tmp = []    
    for i in tqdm(df_tmp):    
        doc = process_string(d=i, with_stemming=with_stemming)
        if remove_duplicate:
            doc = process_string_with_duplicates_elimination(d=doc, with_stemming=with_stemming)
        data_tmp.append(doc)
    return data_tmp

def tf_idf_return_k_result_for_a_given_query(query,k):
    """ Takes a dataframe of diagnostics and returns a list of the most similar CIM10s
    Args:
        query, k
        query is a dataframe containing diagnostics, k represents the length of the CIM10 to return
    """
    list_ = []        
    #get tuples generated from the query (json)
    tuples = generate_tuples_from_query(query)
    # Clean the tuples description so it contains only scientific words
    cleaned_tuples = clean_tuples_list(tuples)        
    for tuple_ in cleaned_tuples :
        res = []
        query_text = ""        
        for text in tuple_ :            
            if text != "inconnu" :                
                query_text += str(text) + " "
       
        cleaned_query = process_string_with_duplicates_elimination(query_text)        
        queryTFIDF = TfidfVectorizer().fit(tf_idf_words)
        queryTFIDF = queryTFIDF.transform([cleaned_query])
        cosine_similarities = cosine_similarity(queryTFIDF, tf_idf_matrix).flatten()
        results = cosine_similarities.argsort()[::-1][:k]
        for i in results:
            text = CIM10_arboresence['diagnostic_num'][i] + " - " + CIM10_arboresence['diagnostic'][i]
            res.append(text)        
        dict_ = {"Diagnostic" : tuple_[0], "CIM10s" : res}        
        list_.append(dict_)
    
    return list_

def tf_idf_return_k_result_for_a_given_sub_query(sub_query,k):    
    """ Takes a sub diagnostic as a String and returns a list of the most similar CIM10s
    Args:
        sub_query, k
        sub_query is a String containing sub diagnostic, k represents the length of the CIM10 to return
    """    
    list_ = [] 
    res = []    
    stemmer= FrenchStemmer(ignore_stopwords=False)
    new_sub_query = ''
    words_in_sub_query = word_tokenize(sub_query)    
    for word in words_in_sub_query:
        word.encode('utf-8')
        if stemmer.stem(word.lower()) in scientificWords:
            new_sub_query += word + " "
    
    cleaned_query = process_string_with_duplicates_elimination(new_sub_query)        
    queryTFIDF = TfidfVectorizer().fit(tf_idf_words)
    queryTFIDF = queryTFIDF.transform([cleaned_query])
    cosine_similarities = cosine_similarity(queryTFIDF, tf_idf_matrix).flatten()
    results = cosine_similarities.argsort()[::-1][:k]
    for i in results:
        text = CIM10_arboresence['diagnostic_num'][i] + " - " + CIM10_arboresence['diagnostic'][i]
        res.append(text)        
    dict_ = {"Diagnostic" : sub_query, "CIM10s" : res}        
    list_.append(dict_)
    
    return list_


def clean_tuples_list(tuples):
    """ Takes a list of tuples and clean every first element from every non scienitic word (in scientificWords) then return a list of the new cleaned tuples
    Args:
        tuples
        tuples is a list of tuple containing each one three elements
    """   
    new_tuple = []
    stemmer= FrenchStemmer(ignore_stopwords=False)
    for tup in tuples:
        new_discription = ''
        words_in_discription = word_tokenize(tup[0])    
        for word in words_in_discription:
            word.encode('utf-8')
            if stemmer.stem(word.lower()) in scientificWords:
                new_discription = new_discription + word + " "
        new_tuple.append((new_discription, tup[1], tup[2]))   
    
    return new_tuple

def get_necessary_data():
    """ 
    returns all necessary data for the cnesst project by extracting them from the pickle necessary_data.pck
    """   
    with open('necessary_data.pck' , 'rb') as file :
        necessary_data = pickle.load(file)
    colones = necessary_data['columns']
    scientificWords = necessary_data['ScientificWords']
    membres_var_glob = necessary_data['membres_var_glob']
    cote_gauche_var_glob = necessary_data['cote_gauche_var_glob']
    cote_droit_var_glob = necessary_data['cote_droit_var_glob']
    tf_idf_matrix = necessary_data['TF_IDF_Matrix']
    tf_idf_words = necessary_data['TF_IDF_Words']
    CIM10_arboresence = necessary_data['arbo_CIM10']
    
    return colones,scientificWords,membres_var_glob,cote_gauche_var_glob,cote_droit_var_glob,tf_idf_matrix,tf_idf_words,CIM10_arboresence


