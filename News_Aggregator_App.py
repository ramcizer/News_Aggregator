import psycopg2
import re 
import pandas as pd
from datetime import datetime
from google_news_feed import GoogleNewsFeed
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from bertopic import BERTopic
import plotly.graph_objs as go
from plotly.graph_objs import Figure
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from wordcloud import WordCloud
import plotly.io as pio
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import streamlit as st
import schedule
import time
import os

def database_connection(): 
# The connection to the PostGre database
    try: 
        conn = psycopg2.connect(
            host="data-sandbox.c1tykfvfhpit.eu-west-2.rds.amazonaws.com",
            dbname="pagila",
            user="de_raka",
            password="vacso",
            port="5432"    
        )
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error:", error)
    return conn


def database_table_creation(connection, cursor):

    sql_create_table = """
        CREATE TABLE IF NOT EXISTS student.rck_news_agg (
            pub_date date,
            title varchar(255),
            publisher varchar(255),
            link TEXT CHECK (LENGTH(link) <= 1000),
            polarity float
        )
    """

    sql_create_table = """
        CREATE TABLE IF NOT EXISTS student.rck_news_entities (
            pub_date date,
            entity varchar(255),
            entity_type varchar(255),
            polarity float
        )
    """
    cursor = connection.cursor()
    cursor.execute(sql_create_table)
    conn.commit()

def update_reprsentative_items_dict(dictionary):
    for key, value in dictionary.items():        
        match = re.match(r'^-?\d+_(.*?)$', value[0])
        # match = re.match(r'(^-?\d+_)', value[0])
        if match:
            new_value = match.group(1)
            value[0] = new_value
    return dictionary


def main_sql_insert_and_check(): 
# Definition of SQL statements

    sql_ins_for_news_agg = "INSERT INTO student.rck_news_agg (pub_date, title, publisher, link, polarity) VALUES (%s, %s, %s, %s, %s)"
    # Insert into a new table for people, orgs and associated poloarity
    sql_ins_for_entities = "INSERT INTO student.rck_news_entities (pub_date, entity, entity_type, polarity) VALUES (%s, %s, %s, %s)"
    # Query Select in order to check whether the news item has already been stored in the databse
    sql_duplicate_check = "SELECT COUNT(*) FROM student.rck_news_agg WHERE title = %s"

    return sql_ins_for_news_agg, sql_ins_for_entities, sql_duplicate_check

def sql_queries_for_BERTopic(): 

    sql_query_for_titles = "SELECT title FROM student.rck_news_agg WHERE pub_date  > current_date - interval '7 days'"
    top_representative_sql = f"SELECT rna.link FROM student.rck_news_agg rna WHERE rna.title = %s AND pub_date  > current_date - interval '7 days'"

    return sql_query_for_titles, top_representative_sql

def sql_for_top_entity_polarity(): 
    top_org_sql = "SELECT entity, avg(polarity) FROM student.rck_news_entities rne WHERE entity_type = 'ORG' AND pub_date  > current_date - interval '7 days' GROUP BY entity ORDER BY count(*) DESC LIMIT 2"
    top_people_sql = "SELECT entity, avg(polarity) FROM student.rck_news_entities rne WHERE entity_type = 'PERSON' AND pub_date  > current_date - interval '7 days' GROUP BY entity ORDER BY count(*) DESC LIMIT 2"
    top_GPE_sql = "SELECT entity, avg(polarity) FROM student.rck_news_entities rne WHERE entity_type = 'GPE' AND pub_date  > current_date - interval '7 days' GROUP BY entity ORDER BY count(*) DESC LIMIT 2"
    NORP_sql = "SELECT entity, avg(polarity) FROM student.rck_news_entities rne WHERE entity_type = 'NORP' AND pub_date  > current_date - interval '7 days' GROUP BY entity ORDER BY count(*) DESC LIMIT 2"
    poduct_sql = "SELECT entity, avg(polarity) FROM student.rck_news_entities rne WHERE entity_type = 'PRODUCT'AND pub_date  > current_date - interval '7 days'  GROUP BY entity ORDER BY count(*) DESC LIMIT 2"

    return top_org_sql, top_people_sql, top_GPE_sql, NORP_sql, poduct_sql


def spacey_load(query1, query2, query3, google_news_feed): 

    # setting the English pipeline
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('spacytextblob')

    # Adding the data
    for headlines in gnf.top_headlines(): 
        description = headlines.description
        match = re.search(r'continue=(.*?)$', headlines.link)
        link_ = match.group(1)
        # link_ = re.match(r'(?<=continue).*', headlines.link)
        cursor.execute(query3, (description,))
        try: 
            record_count = cursor.fetchone()[0]
        except Exception: 
            return 0
        if record_count == 0: 
            doc = nlp(description)
            polarity = doc._.blob.polarity        
            news_data = (headlines.pubDate, description, headlines.source, link_, polarity) 
            cursor.execute(query1, news_data)
            conn.commit()
            for ent in doc.ents:
                entity_data = (headlines.pubDate, ent.text, ent.label_, polarity)
                print(headlines.pubDate, ent.text, ent.label_, polarity)
                cursor.execute(query2, entity_data)        
                conn.commit()


def bertopic_load_query_output(cursor):

    ### Section for BERTopic 

    sql_query_for_titles, top_representative_sql = sql_queries_for_BERTopic()   

    cursor.execute(sql_query_for_titles)
    titles = [title[0] for title in cursor.fetchall()]
    # titles = cursor.fetchall()

    vectorizer_model = CountVectorizer(stop_words="english")

    # The visualisation and topic relations seem to work ok with vectorised model
    topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True)

    topics, probabilities = topic_model.fit_transform(titles)

    print(topic_model.get_topics())

    topic_df = topic_model.get_topic_info()
    print(topic_df)

    # The links to the right of intertopic
    representative_topics_dict = {}
    for i in range(len(topic_df)): 
        # top_representative_sql = f"SELECT rna.link FROM student.rck_news_agg rna WHERE rna.title = %s"
        cursor.execute(top_representative_sql, (topic_df.Representative_Docs[i][0],))
        top_representative = cursor.fetchall()
        representative_topics_dict[i] = [f'{topic_df.loc[i, "Name"]}', top_representative[0][0]]

    # print(representative_topics_dict)

    representative_topics_dict = update_reprsentative_items_dict(representative_topics_dict)

    # print(representative_topics_dict)

    fig = topic_model.visualize_topics()
    # fig.add_candlestick()
    # fig.write_image("visualisation.png")

    # Show works to open a browser. 
    # fig.show()
    fig.write_html("visualisation.html")

    return fig, titles, representative_topics_dict

def wordcloud_load_and_output(title_list): 
    title_text = ' '.join(title_list)
    stopwords = nltk.corpus.stopwords.words('english')

    # Adding stopwords
    stopwords.extend(['say', 'says', 'new', 'day', 'man', 'woman', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'])

    wc = WordCloud(background_color='white', colormap = 'binary',
    stopwords = stopwords, width = 800, height = 500).generate(title_text)

    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation='bilinear')
    # plt.title('NewsCloud')
    plt.axis("off")  # Hide axes

    return plt


def top_entity_polarity(cursor):

    top_org_sql, top_people_sql, top_GPE_sql, NORP_sql, poduct_sql = sql_for_top_entity_polarity()
    cursor.execute(top_org_sql)
    top_org = cursor.fetchall()

    cursor.execute(top_people_sql)
    top_people = cursor.fetchall()

    cursor.execute(top_GPE_sql)
    top_GPE = cursor.fetchall()

    cursor.execute(NORP_sql)
    top_NORP = cursor.fetchall()

    cursor.execute(poduct_sql)
    top_product = cursor.fetchall()

    return top_org, top_people, top_GPE, top_NORP, top_product

st.set_page_config()

st.header('Weekly News Aggregated Summary', divider="blue") 
st.markdown('<link rel="stylesheet" href="custom.css">', unsafe_allow_html=True)

# with st.sidebar:
#     add_radio = st.(
#         "Choose a shipping method",
#         ("Standard (5-15 days)", "Express (2-5 days)")
#     )

with st.container(border=True): 

    st_pyplot = st.empty()

with st.container(border=True): 
    col1, col2 = st.columns([3, 2], gap="small")   
    
    with col1: 
        st.markdown('<link rel="stylesheet" href="custom.css">', unsafe_allow_html=True)
        
        topics_headline = st.empty()
        topics1 = st.empty() 
        topics2 = st.empty()
        topics3 = st.empty() 
        topics4 = st.empty()
        topics5 = st.empty()

        topics6 = st.empty()
        topics7 = st.empty()
        topics8 = st.empty()
        topics9 = st.empty()
        topics10 = st.empty()
        topics11 = st.empty()
        topics12 = st.empty()
    with col2: 
        with st.container(border=True): 
            st.write(f'**Top Mentioned Organisation**', unsafe_allow_html=True) 
            top_mentioned_orgs1 = st.empty()
        with st.container(border=True): 
            st.write(f'**Top Mentioned Person**', unsafe_allow_html=True) 
            top_mentioned_people1 = st.empty() 
        with st.container(border=True):
            st.write(f'**Top Mentioned GPEs**', unsafe_allow_html=True)                
            top_mentioned_GPEs1 = st.empty()
        with st.container(border=True):
            st.write(f'**Top Mentioned NORPs**', unsafe_allow_html=True)                
            top_mentioned_NORPS1 = st.empty()
        with st.container(border=True):
            st.write(f'**Top Mentioned Products**', unsafe_allow_html=True)                
            top_mentioned_products1 = st.empty()
with st.container(): 
    intertopic_chart = st.empty()

def my_task(connection, cursor, news_feed):
    
    sql1, sql2, sql3 = main_sql_insert_and_check()
    spacey_load(sql1, sql2, sql3, gnf)
    fig, titles, representative_topics = bertopic_load_query_output(cursor=cursor)
    original_figure_data = fig['data']
    original_figure_layout = fig['layout']
    # Creating a new figure using plotly.graph_objs.Figure constructor
    fig = Figure(data=original_figure_data, layout=original_figure_layout)
    plt = wordcloud_load_and_output(titles)
    plt2 = wordtopic__good_news()
    orgs, people, GPEs, NORPs, products = top_entity_polarity(cursor=cursor)
    return fig, plt, representative_topics, orgs, people, GPEs, NORPs, products


def frontpage_update(plt, representative_topics):
    st_pyplot.pyplot(plt)
    topics_headline.write(f'**Top Intertopic Indicative Links**', unsafe_allow_html=True)
    topics1.write(f'<span class="custom-line">1. [{representative_topics[1][0]}]({representative_topics[1][1]})', unsafe_allow_html=True)
    topics2.write(f'<span class="custom-line">2. [{representative_topics[2][0]}]({representative_topics[2][1]})', unsafe_allow_html=True)
    topics3.write(f'<span class="custom-line">3. [{representative_topics[3][0]}]({representative_topics[3][1]})', unsafe_allow_html=True)
    topics4.write(f'<span class="custom-line">4. [{representative_topics[4][0]}]({representative_topics[4][1]})', unsafe_allow_html=True)
    topics5.write(f'<span class="custom-line">5. [{representative_topics[5][0]}]({representative_topics[5][1]})', unsafe_allow_html=True)
    topics6.write(f'<span class="custom-line">6. [{representative_topics[6][0]}]({representative_topics[6][1]})', unsafe_allow_html=True)
    topics7.write(f'<span class="custom-line">7. [{representative_topics[7][0]}]({representative_topics[7][1]})', unsafe_allow_html=True)
    topics8.write(f'<span class="custom-line">8. [{representative_topics[8][0]}]({representative_topics[8][1]})', unsafe_allow_html=True)
    topics9.write(f'<span class="custom-line">9. [{representative_topics[9][0]}]({representative_topics[9][1]})', unsafe_allow_html=True)
    topics10.write(f'<span class="custom-line">10. [{representative_topics[10][0]}]({representative_topics[10][1]})', unsafe_allow_html=True)
    topics11.write(f'<span class="custom-line">11. [{representative_topics[11][0]}]({representative_topics[11][1]})', unsafe_allow_html=True)
    topics12.write(f'<span class="custom-line">12. [{representative_topics[12][0]}]({representative_topics[12][1]})', unsafe_allow_html=True)
    top_mentioned_orgs1.write(f'{orgs[0][0]} | Polarity {round(orgs[0][1], 3)}')
    # top_mentioned_orgs2.write(f'{orgs[1][0]} | Polarity {round(orgs[1][1],3)}')
    top_mentioned_people1.write(f'{people[0][0]} | Polarity {round(people[0][1],3)}')
    # top_mentioned_people2.write(f'{people[1][0]} | Polarity {round(people[1][1], 3)}')
    top_mentioned_GPEs1.write(f'{GPEs[0][0]} | Polarity {round(GPEs[0][1], 3)}')
    # top_mentioned_GPEs2.write(f'{GPEs[1][0]} | Polarity {round(GPEs[1][1],3)}')
    top_mentioned_NORPS1.write(f'{NORPs[0][0]} | Polarity {round(NORPs[0][1], 3)}')
    # top_mentioned_NORPS2.write(f'{NORPs[1][0]} | Polarity {round(NORPs[1][1], 3)}')
    top_mentioned_products1.write(f'{products[0][0]} | Polarity {round(products[0][1],3)}')
    # top_mentioned_products2.write(f'{products[1][0]} | Polarity {round(products[1][1],3)}')
    intertopic_chart.plotly_chart(fig, use_container_width=True)


date = datetime.now().strftime('%Y-%m-%d')
conn = database_connection()    
cursor = conn.cursor()
gnf = GoogleNewsFeed(language='en',country='UK')

pio.templates.default = 'plotly'

fig, plt, representative_topics, orgs, people, GPEs, NORPs, products =  my_task(connection=conn, cursor=cursor, news_feed=gnf)

schedule.every(15).minutes.do(my_task, connection=conn, cursor=cursor, news_feed=gnf)

while True: 
    schedule.run_pending()
    frontpage_update(plt=plt, representative_topics=representative_topics)
    time.sleep(1)

