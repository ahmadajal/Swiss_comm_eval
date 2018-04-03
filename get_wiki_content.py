#script for getting the raw content of wikipedia pages in each category
import wikipedia
import re
import pandas as pd
import time

politics = pd.read_csv("Politics.csv")
sports = pd.read_csv("Sports.csv")
history = pd.read_csv("History.csv")
culture = pd.read_csv("Culture.csv")
comp_science = pd.read_csv("Computer_science.csv")

politics.drop(['number', 'namespace', 'touched'], axis = 1, inplace=True)
sports.drop(['number', 'namespace', 'touched'], axis = 1, inplace=True)
history.drop(['number', 'namespace', 'touched'], axis = 1, inplace=True)
culture.drop(['number', 'namespace', 'touched'], axis = 1, inplace=True)
comp_science.drop(['number', 'namespace', 'touched'], axis = 1, inplace=True)

# N = 1000
politics = politics.sort_values('length', ascending=False).reset_index().drop('index', axis=1)
sports = sports.sort_values('length', ascending=False).reset_index().drop('index', axis=1)
history = history.sort_values('length', ascending=False).reset_index().drop('index', axis=1)
culture = culture.sort_values('length', ascending=False).reset_index().drop('index', axis=1)
comp_science = comp_science.sort_values('length', ascending=False).reset_index().drop('index', axis=1)

def correct_title(category):
    category['title'] = category['title'].map(lambda x: re.sub('_', ' ', x))
    
correct_title(politics)
correct_title(sports)
correct_title(history)
correct_title(culture)
correct_title(comp_science)

def add_contents(category):
    content = []
    indicies = []
    counter = 0
    ind = 0
    while len(content) < 1000 or ind >= len(list(category['title'])):
        counter += 1
        title = list(category['title'])[ind]
        try:
            p = wikipedia.page(title)
            content.append(p.content)
            indicies.append(ind)
            ind += 1
            if counter == 10:
                time.sleep(10)
                counter = 0
                print(ind)
        except wikipedia.exceptions.PageError: 
            ind += 1
        except wikipedia.exceptions.DisambiguationError:
            ind += 1
        except wikipedia.exceptions.RedirectError:
            ind += 1
        except Exception:
            time.sleep(10)
    result = category.loc[indicies]
    result['content'] = content
    return result



#---------------
res = add_contents(sports)
print("done sports")
res.to_csv("sports_full.csv")
time.sleep(50)
#---------------
res = add_contents(history)
print("done history")
res.to_csv("history_full.csv")
time.sleep(50)
#---------------
res = add_contents(culture)
print("done culture")
res.to_csv("culture_full.csv")
time.sleep(50)
#---------------
res = add_contents(comp_science)
print("done comp_science")
res.to_csv("comp_science_full.csv")
time.sleep(50)
#---------------
res = add_contents(politics)
res.to_csv("politics_full.csv")
print("done politics")

