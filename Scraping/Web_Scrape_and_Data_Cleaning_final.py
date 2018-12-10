
# coding: utf-8

# # Web Scraping

### Web Scraping From IMDB
!pip install selenium
#from imdb import IMDb
import json
from selenium import webdriver
from tqdm import tqdm
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import numpy as np

# get the total movie list
# select filter url.
url_all = 'https://www.imdb.com/search/title?' +\
          'title_type=feature&release_date=2014-01-01,2018-06-30' +\
          '&num_votes=1000,&languages=en'
# open the page
driver = webdriver.Chrome('./chromedriver')
# display the wedbpage
driver.get(url_all)

film_ls = driver.find_elements_by_class_name('lister-item-content')
for film_link in film_ls:
        # update the dictionary
        process_lister(film_dict, film_link)

# turn next page
page =  1
while True:
    if page % 5 == 0 or page == 1:
            print('Process at page {}.'.format(page))
    try:
        next_page = driver.find_elements_by_class_name('desc')[-1]
        next_page = next_page.find_elements_by_tag_name('a')[-1]
        try:
            next_page.click()
            time.sleep(0.5)
        except:
            continue
        film_ls = driver.find_elements_by_class_name('lister-item-content')
        for film_link in film_ls:
            # update the dictionary
            process_lister(film_dict, film_link)
        page += 1
    except:
        break
df_film_ls = pd.DataFrame(film_dict).T
#df_film_ls.to_csv('film_list.csv')
df_film_ls = pd.read_csv('film_list.csv', index_col = 0)
# display the film data
df_film_ls.head()
def review_scrape(driver, film_url, MAX_NUM  = 100):
    '''
    core func. to scrape the review for given film
    Args:
        film_url - respective film link 
                   (format: https://www.imdb.com/title/tt0113703)
        MAX_NUM - largest number of review to collected, be default set up to 100
    '''
    # open the page with chromedriver path 
    # display the wedbpage
    driver.get(film_url+'/reviews?ref_=tt_urv')
    ## @press the load button
    try:
        elem = driver.find_element_by_id('load-more-trigger')
        load_count = 0
        while load_count < MAX_NUM // 20 + 1:
            try:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                elem.click()
                load_count += 1
                continue
            except:
                break
    except:
        print('no need to load')
    ## @open the warning button to release the full content
    reviews = driver.find_elements_by_class_name('review-container')
    for rev in reviews:
        try:
            # catch warning
            button = ind_review.find_element_by_class_name('ipl-expander')
            button.click()
            time.sleep(1)
        except:
            continue
    ## collect the review 
    # maximum MAX_NUM reviews
    content_dict = generate_content(driver, MAX_NUM)
    
    return content_dict
ind_url = 'https://www.imdb.com/title/tt4123430/reviews?ref_=tt_urv'
beast_review = review_scrape(scrape_driver, ind_url, MAX_NUM= 1000)

# chromedriver path may need to specified specifically
scrape_driver = webdriver.Chrome('./chromedriver')

store the updated review dictionary
with open('beast_review.json', 'w') as fp:
    json.dump(beast_review, fp)

# Help Function
# access the content function
def generate_content(driver, MAX_NUM):
    '''
    generate the content dictionary with user name as key & review detail as values
    Args: 
        driver - active website driver for scraping
        MAX_NUM - largest number of reviews to collected for the movie
    '''
    review_dict = dict()
    COUNT = 0
    
    review_ls = driver.find_elements_by_class_name('review-container')
    for i,rev in enumerate(review_ls):
        if COUNT > MAX_NUM:
            break
        try:
            COUNT += 1
            name, rating, date, content = access_feature(rev)
            review_dict[name] = {'rating':rating, 'date':date, 'content':content}
        except:
            continue
    return review_dict

# extract the specific review data
def access_feature(ind_review):
    rating = ind_review.find_element_by_class_name('rating-other-user-rating').text
    name = ind_review.find_element_by_class_name('display-name-link').text
    date = ind_review.find_element_by_class_name('review-date').text
    content = ind_review.find_element_by_class_name('content').text
        
    return name, rating, date, content

## Scrape budget, gross revenue, open weekend revenue and release date information from IMDB
import requests
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
data = pd.read_csv("film_list.csv")
data.rename(columns={data.columns[0]:'name'},inplace = True)
# get the budget, open weekend revenye and studio information on IMDB from our basic dataframe
def get_info(data):
    import requests
    import pandas as pd
    import requests
    import re
    from bs4 import BeautifulSoup
    Name = data['name']
    Budget = []
    Open = []
    Studio = []
    url_list = data['link']
    for link in url_list:
        response = requests.get(link)
        results_page = BeautifulSoup(response.content,'lxml')
        div_tag = results_page.find_all('div',{'class':'txt-block'})
        flag1 = False
        flag2 = False
        flag3 = False
        for item in div_tag:
            a = item.get_text()
            b = re.search(r'(Budget:)([\$,0-9]+)',a)
            c = re.search(r'(Opening Weekend USA: )([\$,0-9]+)',a)
            d = re.search(r'(Production Co:)(([\s\S]*))(See more)',a)
            if b:
                Budget.append(b.group(2))  
                flag1 = True
            if c:
                Open.append(c.group(2).strip(','))
                flag2 = True
            if d:
                Studio.append(d.group(3))
                flag3 = True
            if flag1 and flag2 and flag3:
                break
        if not flag1:
            Budget.append('NaN')
        if not flag2:
            Open.append('NaN')
        if not flag3:
            Studio.append('NaN')
    df = pd.DataFrame({'Name':Name, 'Budget':Budget,'Open':Open,'Studio':Studio})    
    return df

import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import datetime

# We extent the info of movie in film_list by scraping the box office.
data = pd.read_csv("film_list.csv")
# Check the data in film_list.csv
data.info()
data.rename(columns={ data.columns[0]: "name" }, inplace=True)
# In order to get the info in box office, generate a function as get_boxinfo
def get_boxinfo(data):
    
    Source = list(data['name'])
    Name = []
    Page_name = []
    Num = []
    Url = []
    Studio = []
    Theaters =[]
    Open_revenue =[]
    Release_date = []
    for i in tqdm(range(len(Source))):
#         print(i)
        source = Source[i]
        if re.search(r'\'', source):
            source = source.replace("'", '%27')
        if re.search(r' ', source):
            source = source.replace(' ', '%20')
        url = "https://www.boxofficemojo.com/search/?q=%s" %(source)
        response = requests.get(url)
        results_page = BeautifulSoup(response.content,'lxml')
        num = 0
        
        if results_page.find('tr',{'bgcolor':'#FFFF99'}):
            num += 1
            Name.append(Source[i])
            td_tag = results_page.find('tr',{'bgcolor':'#FFFF99'}).find_all('td')
            tag = results_page.find('tr',{'bgcolor':'#FFFF99'})
            url = 'https://www.boxofficemojo.com' + tag.find('a').get('href')
            Url.append(url)
            page_name = tag.find('a').get_text()
            Page_name.append(page_name)
            studio = td_tag[1].get_text()
            Studio.append(studio)
            theaters = td_tag[3].get_text()
            Theaters.append(theaters)
            open_revenue = td_tag[4].get_text()
            Open_revenue.append(open_revenue)
            release_date = td_tag[6].get_text()
            Release_date.append(release_date)
            Num.append(num)
            
        if results_page.find('tr',{'bgcolor':'#FFFFFF'}):
            num += 1
            Name.append(Source[i])
            td_tag = results_page.find('tr',{'bgcolor':'#FFFFFF'}).find_all('td')
            tag = results_page.find('tr',{'bgcolor':'#FFFFFF'})
            url = 'https://www.boxofficemojo.com' + tag.find('a').get('href')
            Url.append(url)
            page_name = tag.find('a').get_text()
            Page_name.append(page_name)
            studio = td_tag[1].get_text()
            Studio.append(studio)
            theaters = td_tag[3].get_text()
            Theaters.append(theaters)
            open_revenue = td_tag[4].get_text()
            Open_revenue.append(open_revenue)
            release_date = td_tag[6].get_text()
            Release_date.append(release_date)
            Num.append(num)
        
        if num == 0:
            Name.append(Source[i])
            Url.append(None)
            Studio.append(None)
            Theaters.append(None)
            Open_revenue.append(None)
            Release_date.append(None)
            Num.append(num)
            
    df = pd.DataFrame({'Name':Name, 'Page_name': Page_Name, 'Num': Num, 'Url':Url,'Studio':Studio,'Theaters':Theaters,'Open_revenue':Open_revenue,'Release_date':Release_date})    
    return df
box_info = get_boxinfo(df)
# Save for use
# box_info.to_csv("box_info.csv")
# box_info = pd.read_csv("box_info.csv", index_col = 0)

# Change the format of the date
for i in range(len(box_info['Release_date'])):
    try:
        box_info['Release_date'][i] = datetime.datetime.strptime(box_info['Release_date'][i], '%m/%d/%Y')
    except:
        pass
# Get the Release_year
box_info['Release_year'] = [None]*len(box_info)
for i in range(len(box_info)):
    try:
        box_info['Release_year'][i] = datetime.datetime.strftime(box_info['Release_date'][i], '%Y')
    except:
        pass
# Get the Release_month
box_info['Release_month'] = [None]*len(box_info)
for i in range(len(box_info)):
    try:
        box_info['Release_month'][i] = datetime.datetime.strftime(box_info['Release_date'][i], '%m')
    except:
        pass
# Get the Release_day
box_info['Release_day'] = [None]*len(box_info)
for i in range(len(box_info)):
    try:
        box_info['Release_day'][i] = datetime.datetime.strftime(box_info['Release_date'][i], '%d')
    except:
        pass
# Delete Duplicate films
for i in range(len(box_info)):
    if box_info_1['Num'][i] == 2:
        if not np.isnan(box_info_1['Release_year'][i-1]):
            if box_info_1['Release_year'][i-1] < 2014:
                box_info_1['Num'][i-1] = -1
                
        if not np.isnan(box_info_1['Release_year'][i]):
            if box_info_1['Release_year'][i] < 2014:
                box_info_1['Num'][i] = -1
box_info = box_info[box_info['Num'] != -1]
box_info = box_info.reset_index(drop=True)
for i in range(len(box_info) - 1):
    if box_info['Name'][i+1] == box_info['Name'][i]:
        box_info['Num'][i+1] = -2
box_info = box_info[box_info['Num'] != -2]
# Save for use
# box_info.to_csv("box_data_clean.csv")
# We find that the info of some films cannot match the name list of IMDb
# Thus, we return back to IMDb to get the matched info for that list of movies

# Similar date data can be scarp from IMDB (but less)
# Get the release_date
def get_release_date(df):
    Name = list()
    Release_Date = list()
    pattern = r'(\d+)\s(\w+)\s(\d{4})'
    for i in tqdm(range(len(df))):
        name = df.index[i]
        Name.append(name)
        url = df['link'][i]
        response = requests.get(url)
        results_page = BeautifulSoup(response.content,'lxml')
        response = requests.get(url)
        results_page = BeautifulSoup(response.content,'lxml')
        table = results_page.find('div', class_ = "article", id = "titleDetails").find_all('div', class_ = "txt-block")
        k = -1
        for j in range(len(table)):
            if table[j].find('h4'):
                if table[j].find('h4').get_text() == 'Release Date:':
                    k = j
                    break
        if k != -1: 
            raw_info = table[k].get_text()
            res = re.findall(pattern, raw_info)
        else:
            Day.append(None)
            Month.append(None)
            Year.append(None)
            Date.append(None)
    data = pd.DataFrame({'Name':Name, 'Release_Date': Release_Date})    
    return data
dff = get_release_date(df)

# Get Year, Month, Day
dff['Release_Year'] = [None]*len(dff)
dff['Release_Month'] = [None]*len(dff)
dff['Release_Day'] = [None]*len(dff)
for i in tqdm(range(len(dff))):
    if isinstance(dff['Release'][i], str):
        temp = dff['Release'][i].split('(')[0].strip().split( )
        if len(temp) == 3:
            dff['Release_Year'][i] = temp[0]
            dff['Release_Month'][i] = temp[1][0:3]
            dff['Release_Day'][i] = temp[2]
        elif len(temp) == 1:
            dff['Release_Year'][i] = temp[0]
for i in tqdm(range(len(dff))):
    try:
        dff['Release_Month'][i] = datetime.datetime.strptime(dff['Release_Month'][i], '%b')
        dff['Release_Month'][i] = datetime.datetime.strftime(dff['Release_Month'][i], '%m')
    except:
        pass
# dff['Release_Month'] = dff['Release_Month'].apply(lambda x: datetime.datetime.strptime(x, '%b'))
dff.to_csv("Release_Date.csv")


# Scraping user reviews
# store the review into dicitonary along movie
review_dict = {} 

# set up the start and stop movie index for scraping
start_id = 680

stop_id = 999

assert 0 <= start_id <= 2846
assert 0 <= stop_id <= 2846

# load the json data for processing
with open('review_data.json', 'r') as fp:
    review_dict = json.load(fp)
    
# open the driver
# chromedriver path may need to specified specifically
scrape_driver = webdriver.Chrome('./chromedriver')

for i in tqdm(num_ls):
    try:
        # get the film name
        name = df_film_ls.index[i]
        url = df_film_ls.iloc[i, -4]
        # scrape the review text (max 100)
        val_dict = review_scrape(scrape_driver, url.split('/?')[0])
        review_dict[name] = val_dict
    except:
        print('Oops, scraping for movie {}-{} was just halted at the index {} movie {}'\
              .format(start_id, stop_id, i, name))
    if i % 25 == 0:
        print('process at {}th film {}'.format(i, name))

# check for empty reviews
movie_ls = list(review_dict.keys())
bug_ls = []
# missing movie index
num_ls = []

for j, mov in tqdm(enumerate(movie_ls)):
    check_val = review_dict[mov]
    
    if len(check_val) == 0:
        print('movie: {} with index {} is empty!'.format(mov, j))
        bug_ls.append(mov)
        num_ls.append(j)
                
with open('review_data_xxx.json', 'r') as fp:
    review_dict_xxx = json.load(fp)

with open('review_data_ql.json', 'r') as fp1:
    review_dict_ql = json.load(fp1)
    
with open('review_data_ql2.json', 'r') as fp2:
    review_dict_ql2 = json.load(fp2)
    
with open('review_data_ql3.json', 'r') as fp3:
    review_dict_ql3 = json.load(fp3)

    
# update the revirw dict together
#review_dict.update(review_dict_ql3)

# store the updated review dictionary
with open('review_data.json', 'w') as fp:
    json.dump(review_dict, fp)

# Help Function
def process_lister(film_dict, indiv_review, return_obj = False):
    # header part
    header = indiv_review.find_element_by_class_name('lister-item-header').find_element_by_tag_name('a')

    url = header.get_property('href')
    name = header.text

    # category
    try:
        certificate = indiv_review.find_element_by_class_name('certificate').text
    except:
        certificate = None
    try:
        runtime = indiv_review.find_element_by_class_name('runtime').text
    except:
        runtime = None
    genre = indiv_review.find_element_by_class_name('genre').text

    # rating
    rate_text = indiv_review.find_element_by_class_name('ratings-bar').text
    rate = rate_text.split('\n')[0].split(' ')[0]
    try:
        meta_score = rate_text.split('\n')[1].split(' ')[0]
    except:
        meta_score = None
    # description
    description = [x.text for x in indiv_review.find_elements_by_class_name('text-muted')][2]

    # cast
    cast = indiv_review.find_elements_by_tag_name('p')[2]
    cast_url = ' | '.join([x.get_attribute('href') for x in cast.find_elements_by_tag_name('a')])
    cast_str = cast.text

    # gross
    gross = indiv_review.find_element_by_class_name('sort-num_votes-visible').text
    
    film_dict[name] = {'link': url, 'certificate':certificate, 'runtime':runtime,
                         'genre':genre, 'rate':rate, 'meta_score':meta_score, 'description':description,
                         'cast_str':cast_str, 'cast_url':cast_url, 'gross':gross }
    if return_obj:
        return film_dict
    return 

# Scraping and cleaning critic reviews
import datetime
import json
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
from tqdm import tqdm


# In[17]:


def get_movie_url(keyword, year):
    url = "https://www.rottentomatoes.com/search/?search=%s"  % (keyword)
    response = requests.get(url)
    if not response.status_code == 200:
        return None
    try:
        results_page = BeautifulSoup(response.content,'lxml')
        #print(results_page.prettify())
        div_tag = results_page.find('div',{'id':"main_container"})
        data_string = div_tag.find('script').get_text().strip()
        pattern = re.compile(r'(\"year\"):(%s),(\"url\"):(\".*?\")'%(str(year)))
        movie_url = re.search(pattern, data_string)
        return movie_url.group(4).strip('"')
    except:
        return None


# In[18]:


def get_first_week_critic_review(name, in_theater_date):
    single_film = dict()
    if not isinstance(in_theater_date, str):
        single_film['bug'] = 'invalid release date'
        return single_film
    try:
        year = int(in_theater_date[-4:])
        ddl = datetime.datetime.strptime(in_theater_date,'%b %d, %Y')+datetime.timedelta(days=7)
    except:
        single_film['bug'] = 'invalid release date and year'
        return single_film
    part_url = get_movie_url(name, year)
    if not part_url:
        single_film['bug'] = 'invalid homepage url'
        return single_film
    mark = 0
    count = 1
    test_url = "https://www.rottentomatoes.com%s/reviews/?page=%s&sort=" % (part_url, str(1))
    test_response = requests.get(test_url,allow_redirects=False)
    if not test_response.status_code == 200:
        single_film['bug'] = 'invalid homepage url'
        return single_film
    test_results_page = BeautifulSoup(test_response.content,'html')
    try:
        page_info = test_results_page.find('span',{'class':'pageInfo'}).get_text()
        page_match = re.search(r'\d+$',page_info)
        if not page_match:
            max_page = 50
        max_page = int(page_info[page_match.span()[0]:])
    except:
        max_page = 50
    for n in range(max_page):    
        if mark == 1:
            break
        url = "https://www.rottentomatoes.com%s/reviews/?page=%s&sort=" % (part_url, str(max_page-n))
        response = requests.get(url,allow_redirects=False)
        if not response.status_code == 200:
            continue
        results_page = BeautifulSoup(response.content,'html')
        list_ = results_page.find_all('div', {'class':'review_area'})
        for item in list_: 
            single_review = dict()
            review_date = item.find('div',{'class':'review_date subtle small'}).get_text()
            review_date = datetime.datetime.strptime(review_date,' %B %d, %Y')
            if review_date > ddl:
                mark = 1
                continue
            string_date = datetime.datetime.strftime(review_date, '%Y/%m/%d')
            review = item.find('div', {'class':'the_review'}).get_text()
            score = item.find('div',{'class':'small subtle'}).get_text()
            match = re.search(r'[\d.]*\d+/\d+$',score)
            if match:
                start = match.span()[0]
                end = match.span()[1]
                score = score[start:end]
                bound = re.search(r'/',score).span()[0]
                try:
                    score = float(score[:bound])/float(score[bound+1:])
                except:
                    score = ''
            else:
                score = ''
            single_review['score'] = score
            single_review['date'] = string_date
            single_review['content'] = review
            single_film[str(count)] = single_review
            count += 1
    if not single_film:
        single_film['bug'] = 'No critics review'
    return single_film


# In[5]:


def total_critic_review(name_df):
    total_films = dict()
    num = 1
    try:
        for n in range(len(name_df['name'])):
            name = name_df['name'][n]
            try:
                in_theater_date = name_df['Release_date'][n]
                total_films[name] = get_first_week_critic_review(name, in_theater_date)
            except:
                total_films[name] = dict()
            if not total_films[name]:
                print("No.%s is empty."%str(num))
            #For convenience after every 5 filmswe make a saving
            if (num % 5 == 0)|(n == len(name_df['name'])-1):
                with open('total_critic_review.json', 'w') as fp:
                    json.dump(total_films, fp)
            print(num)
            num += 1
    except:
        print("Something went wrong. Completed %s films." % str(len(total_films)))
        with open('temp_total_critic_review.json', 'w') as fp:
            json.dump(total_films, fp)
        print("Saved.")
        raise
    return len(total_films)


# In[ ]:


#Load df to get namelist
df = pd.read_csv('final_data_v1.csv')


# In[ ]:


total_critic_review(df)


# In[19]:


#If in total_critic_review any key has an empty content, it is a bug
#We record all bug names, and try to figure out the reason before deciding how to deal with them
bug_df = pd.read_csv('e:/DA/bug_data.csv')
bug_df.info()


# In[ ]:


checked_films = dict()
bug_count = dict()
name_df = bug_df
for n in range(len(name_df['name'])):
    name = name_df['name'][n]
    in_theater_date = name_df['Release_date'][n]
    checked_films[name] = get_first_week_critic_review(name, in_theater_date)
    if 'bug' in checked_films[name]:
        bug = checked_films[name]['bug']
        if bug in bug_count:
            bug_count[bug] += 1
        else:
            bug_count[bug] = 1
        print(n+1,name,bug)


# # Data Cleaning

# In[ ]:


import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
film_list = pd.read_csv('film_list.csv')
df = film_list

# get directors from "cast_str"
def get_director_actor(string):
    try:
        string.split('|')
        director = string.split('|')[0].split(':')[1]
        actor = string.split('|')[1].split(':')[1]
        return (director,actor)
    except:
        return (None,None)
df['director'] = df.apply(lambda x: get_director_actor(x['cast_str'])[0],axis = 1)
df['actor'] = df.apply(lambda x: get_director_actor(x['cast_str'])[1],axis = 1)

# get votes and gross 
def get_votes_gross(string):
    try:
        string.split('|')
        votes = string.split('|')[0].split(':')[1].strip()
        gross = string.split('|')[1].split(':')[1].strip()[1:-1]
        return (votes,gross)
    except:
        return (None,None)
df['Votes'] = df.apply(lambda x: get_votes_gross(x['gross'])[0],axis = 1)
df['Gross'] = df.apply(lambda x: get_votes_gross(x['gross'])[1],axis = 1)

# get runtime
def get_runtime(string):
    try:
        return int(string.split(' ')[0])
    except:
        return None
df['Runtime'] = df.apply(lambda x :get_runtime(x['runtime']),axis = 1)

# get all genre in dataset, and lable each movie with its genre
import re
def get_genre(string,genre_string):
    if genre_string in re.split('[, ]',string):
        return 1
    else:
        return 0
def all_genre_set(data):
    movie_genre = data['genre']
    genre = set()   
    genre_set = set()
    for item in movie_genre:
        genre.update(str(item).strip().split(','))
    for item in genre:
        genre_set.add(str(item).strip())
    return genre_set
genre_list = list(all_genre_set(df))
for ge in genre_list:
    df[ge] = df.apply(lambda x:get_genre(x['genre'],ge),axis = 1)

# scrap top 250 directors from web
def get_famous_directors(url,famous_directors = []):
    response = requests.get(url)
    response.status_code
    results_page = BeautifulSoup(response.content,'lxml')
    td_tag = results_page.find('div',{'class':'lister-list'}).find_all('div',{'class':'lister-item mode-detail'})
    for i in range(len(td_tag)):
        director = td_tag[i].find('h3',{'class':'lister-item-header'}).find('a').get_text().strip()
        famous_directors.append(director)
    return famous_directors
get_famous_directors("https://www.imdb.com/list/ls054420501/" )
get_famous_directors("https://www.imdb.com/list/ls054420501/?sort=list_order,asc&mode=detail&page=2" )
get_famous_directors("https://www.imdb.com/list/ls054420501/?sort=list_order,asc&mode=detail&page=3" )

# find if director to certain movie is famous
def if_famous_director(string):
    try:
        for i in string.split(','):
            if i.strip() in famous_directors:
                return 1
        return 0
    except:
        return None
df['famous_director'] = df.apply(lambda x :if_famous_director(x['director']),axis = 1)

# scrap top 500 actors from web
def get_famous_actor(url,famous_actors = []):
    response = requests.get(url)
    response.status_code
    results_page = BeautifulSoup(response.content,'lxml')
    td_tag = results_page.find('div',{'class':'lister-list'}).find_all('div',{'class':'lister-item mode-detail'})
    for i in range(100):
        director = td_tag[i].find('h3',{'class':'lister-item-header'}).find('a').get_text().strip()
        famous_actors.append(director)
get_famous_actor("https://www.imdb.com/list/ls004521485/" )
for i in range(1,5):
    get_famous_actor(f"https://www.imdb.com/list/ls004521485/?sort=list_order,asc&mode=detail&page={i}")
    
# find if actors to certain movie is famous
def if_famous_actor(string):
    try:
        for i in string.split(','):
            if i.strip() in famous_actors:
                return 1
        return 0
    except:
        return None
df['famous_actor'] = df.apply(lambda x :if_famous_director(x['actor']),axis = 1)

df.to_csv('cleaned1.csv')
df1 = pd.read_csv('budget_open_studio_clean.csv')

# get all studios in dataset, and lable each movie with its studio
def all_Studio_set(data):
    studio_genre = data['Studio']
    studio = set()   
    studio_set = set()
    for item in studio_genre:
        studio.update(str(item).strip().split(','))
    for item in studio:
        studio_set.add(str(item).strip())
    studio_set.remove('')
    return studio_set
all_Studio_set(df1)

# get a all_studio list
all_studio = list()
for i in range(len(df1)):
    all_studio.extend(str(df1.iloc[i]['Studio']).split(','))
    
# find the top 20 biggest(famous) studios
import collections
c = collections.Counter(all_studio)
studio_common = c.most_common(20)
studio = list(dict(studio_common).keys())

# find if studio to certain movie is famous
def if_famous_studio(string):
    try:
        for i in string.split(','):
            if i.strip() in studio:
                return 1
        return 0
    except:
        return None
df1['famous_studio'] = df1.apply(lambda x :if_famous_studio(x['Studio']),axis = 1)
df1.to_csv('cleaned3.csv')

