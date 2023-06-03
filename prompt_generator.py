class SemAnnPrompt:
    def __init__(self, preamble, c1sentence, question):
        self.preamble = preamble
        self.c1sentence = c1sentence
        self.question = question
        self.lang = 'english'
    
    def get_prompt(self, candidate1):
        full_st = self.preamble + ' ' + self.c1sentence + candidate1 + ' ' + self.question
        return full_st
    
    def get_prompt_nospaces(self, candidate1):
        full_st = self.preamble + self.c1sentence + candidate1 + self.question
        return full_st

'''
Purpose: This file contains the semantic annotation prompt generation functions.
'''

label_lst = ['city', 'state', 'category', 'class', 'name', 'description', 
             'type', 'address', 'age', 'club', 'day', 'location', 'status', 
             'result', 'year', 'album', 'code', 'position', 'company', 'symbol', 
             'rank', 'country', 'team', 'county', 'weight', 'language', 'origin', 
             'genre', 'gender', 'artist', 'collection', 'ranking', 'notes', 
             'isbn', 'format', 'owner', 'nationality', 'order', 'area', 
             'publisher', 'region', 'family', 'sex', 'component', 'elevation', 
             'capacity', 'range', 'duration', 'affiliation', 'teamName', 'plays', 
             'director', 'credit', 'brand', 'jockey', 'product', 'grades', 
             'requirement', 'command', 'education', 'birthPlace', 'currency', 
             'continent', 'depth', 'service', 'industry', 'birthDate', 'sales', 
             'creator', 'person', 'operator', 'species', 'classification', 
             'manufacturer', 'fileSize', 'affiliate']

def gen_veryplain():
    preamble = 'Choose the labels from the following list that best represent the columns in the following table:\n\n'
    preamble += 'List:\n'
    preamble += str(label_lst) + '\n\n'
    c1sentence = 'Table:\n'
    question = '\n\nGive the answer as a python list whose ith element corresponds to the label for the ith column.'
    prompt_tmp = SemAnnPrompt(preamble, c1sentence, question)
    return prompt_tmp

def gen_game():
    preamble = 'Suppose we are playing a game where I show you instances of multiple concepts in the form of a table,'
    preamble += 'and you choose the one that fits best for each column.\n\n'
    preamble += 'Here are the concepts to choose from:\n'
    preamble += str(label_lst) + '\n\n'
    c1sentence = 'And here are instances of some set of concepts, as a table:\n'
    question = '\n\nWhat concept best describes each column? '
    question += 'Give the answer as a python list whose ith element corresponds to the concept for the ith column.'
    prompt_tmp = SemAnnPrompt(preamble, c1sentence, question)
    return prompt_tmp

def gen_analyst():
    preamble = 'I am a business analyst trying to figure out the schema of a table I found '
    preamble += 'in our company\'s data lake.\n\n'
    preamble += 'I\'m using our company\'s business dictionary, which has the following terms:\n'
    preamble += str(label_lst) + '\n\n'
    c1sentence = 'And the table I found is the following:\n'
    question = '\n\nWhat terms best describe each column of this table?'
    question += ' Give the answer as a python list whose ith element corresponds to the concept for the ith column.'
    prompt_tmp = SemAnnPrompt(preamble, c1sentence, question)
    return prompt_tmp

def gen_lost():
    preamble = 'I seem to have lost a spreadsheet I was working with yesterday.'
    preamble += 'I recovered it, but I can no longer tell what each column means.\n\n'
    preamble += 'I brainstormed the following list of candidate labels for each column:\n'
    preamble += str(label_lst) + '\n\n'
    c1sentence = 'And here\'s the spreadsheet I recovered:\n'
    question = '\n\nWhat are the best labels for each spreadsheet column?'
    question += ' Give the answer as a python list whose ith element corresponds to the label for the ith column.'
    prompt_tmp = SemAnnPrompt(preamble, c1sentence, question)
    return prompt_tmp

TEMPLATES = {
    'plain': gen_veryplain(),
    'game': gen_game(),
    'analyst': gen_analyst(),
    'lost': gen_lost(),
}
