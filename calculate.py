from main import GetUserPersonalityRequest, GetUserPersonalityResponse, getUserPersonality
from post_stats import get_post_words, get_posts_word_stat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import sqlite3
import pandas as pd
import numpy as np

ei_model=joblib.load("./ei_model.joblib")
as_model=joblib.load("./as_model.joblib")
gr_model=joblib.load("./gr_model.joblib")

con = sqlite3.connect("./test.sqlite")
cursor = con.cursor()
data =  cursor.execute("""--sql--sql
    SELECT
        u.id,
        u.vk_id,
        u.age,
        IFNULL(p.post_count, 0) AS post_count,
        IFNULL(p.repost_count, 0) AS repost_count,
        IFNULL(p.avg_post_length, 0) AS avg_post_length,
        p.date_diff AS post_date_diff,
        IFNULL(ph.photo_count, 0) AS photo_count,
        IFNULL(ph.total_likes, 0) AS total_likes_on_photos,
        IFNULL(fr.friend_count, 0) AS friend_count,
        IFNULL(gr.group_count, 0) AS group_count,
        IFNULL(bio.has_activity, 0) AS has_activity,
        IFNULL(bio.has_interests, 0) AS has_interests,
        IFNULL(bio.has_music, 0) AS has_music,
        IFNULL(bio.has_films, 0) AS has_films,
        IFNULL(bio.has_tv, 0) AS has_tv,
        IFNULL(bio.has_books, 0) AS has_books,
        IFNULL(bio.has_games, 0) AS has_games,
        IFNULL(bio.has_quotes, 0) AS has_quotes,
        IFNULL(bio.has_about, 0) AS has_about,
        IFNULL(gr.depression_group_count, 0) as depression_group_count,
        IFNULL(gr.minecraft_group_count, 0) as minecraft_group_count,
        all_posts_text
    FROM
        user u
    LEFT JOIN (
        SELECT
            user_id,
            COUNT(*) AS post_count,
            AVG(LENGTH(text)) AS avg_post_length,
            (MAX(date) - MIN(date)) / (1000 * 3600 * 24) AS date_diff,
            SUM(CASE WHEN isowner = 0 THEN 1 ELSE 0 END) AS repost_count,
            GROUP_CONCAT(post.text, '|||') AS all_posts_text
        FROM
            post
        GROUP BY
            user_id
    ) p ON u.id = p.user_id
    LEFT JOIN (
        SELECT
            user_id,
            COUNT(*) AS photo_count,
            SUM(like_count) AS total_likes
        FROM
            photo
        GROUP BY
            user_id
    ) ph ON u.id = ph.user_id
    LEFT JOIN (
        SELECT
            user_id,
            COUNT(*) AS friend_count
        FROM
            friend
        GROUP BY
            user_id
    ) fr ON u.id = fr.user_id
    LEFT JOIN (
        SELECT
            user_id,
            COUNT(*) AS group_count,
            SUM(CASE WHEN lower(name) like '%депресси%' then 1 else 0 END) as depression_group_count,
            SUM(CASE WHEN lower(name) like '%майнкрафт%' or lower(name) like '%minecraft%'  then 1 else 0 END) as minecraft_group_count
        FROM
            group_table
        GROUP BY
            user_id
    ) gr ON u.id = gr.user_id
    LEFT JOIN (
        SELECT
            user_id,
            MAX(CASE WHEN activity != '' THEN 1 ELSE 0 END) AS has_activity,
            MAX(CASE WHEN interests != '' THEN 1 ELSE 0 END) AS has_interests,
            MAX(CASE WHEN music != '' THEN 1 ELSE 0 END) AS has_music,
            MAX(CASE WHEN films != '' THEN 1 ELSE 0  END) AS has_films,
            MAX(CASE WHEN tv != '' THEN 1 ELSE 0  END) AS has_tv,
            MAX(CASE WHEN books != '' THEN 1 ELSE 0  END) AS has_books,
            MAX(CASE WHEN games != '' THEN 1 ELSE 0  END) AS has_games,
            MAX(CASE WHEN quotes != '' THEN 1 ELSE 0  END) AS has_quotes,
            MAX(CASE WHEN about != '' THEN 1 ELSE 0  END) AS has_about
        FROM
            biography
        GROUP BY
            user_id
    ) bio ON u.id = bio.user_id
    ;
""").fetchall()
column_names = [
    'id',
    'vk_id',
    'age',
    'post_count',
    'repost_count',
    'avg_post_length',
    'post_date_diff',
    'photo_count',
    'total_likes',
    'friend_count',
    'group_count',
    'has_activity',
    'has_interests',
    'has_music',
    'has_films',
    'has_tv',
    'has_books',
    'has_games',
    'has_quotes',
    'has_about',
    'depression_group_count',
    'minecraft_group_count',
    'all_posts_text'
]

df = pd.DataFrame(np.array(data))
df = df.rename({i: name for i, name in enumerate(column_names)}, axis=1)
df.set_index('id')
df['profile_fullness'] = (
    df['has_activity'] * 5 +
    df['has_interests'] * 5 +
    df['has_music'] * 3 +
    df['has_films'] * 3 +
    df['has_tv'] * 3 +
    df['has_books'] * 3 +
    df['has_games'] * 3 +
    df['has_quotes'] * 5 +
    df['has_about'] * 5
)
df = df.fillna(0)
df['word_stat'] = df.all_posts_text.apply(lambda x: get_post_words(x))
df['unique_word_stat'] = df.word_stat.apply(len)
df['word_sens_stat'] = df.word_stat.apply(lambda x: get_posts_word_stat(x))
df['post_len_std'] = df['all_posts_text'].apply(lambda x: np.array(list(map(len, x.split('|||'))) if x != 0 else np.array([])).std())

df['sad'] = df.word_sens_stat.apply(lambda x: x['sad'][0] if 'sad' in x else 0)
df['fun'] = df.word_sens_stat.apply(lambda x: x['fun'][0] if 'fun' in x else 0)
df['angry'] = df.word_sens_stat.apply(lambda x: x['angry'][0] if 'angry' in x else 0)
df['fear'] = df.word_sens_stat.apply(lambda x: x['fear'][0] if 'fear' in x else 0)
df['disgust'] = df.word_sens_stat.apply(lambda x: x['disgust'][0] if 'disgust' in x else 0)
df['mat'] = df.word_sens_stat.apply(lambda x: x['mat'][0] if 'mat' in x else 0)

df['sad_unique'] = df.word_sens_stat.apply(lambda x: x['sad'][1] if 'sad' in x else 0)
df['fun_unique'] = df.word_sens_stat.apply(lambda x: x['fun'][1] if 'fun' in x else 0)
df['angry_unique'] = df.word_sens_stat.apply(lambda x: x['angry'][1] if 'angry' in x else 0)
df['fear_unique'] = df.word_sens_stat.apply(lambda x: x['fear'][1] if 'fear' in x else 0)
df['disgust_unique'] = df.word_sens_stat.apply(lambda x: x['disgust'][1] if 'disgust' in x else 0)
df['mat_unique'] = df.word_sens_stat.apply(lambda x: x['mat'][1] if 'mat' in x else 0)


df2 = df[[
    'friend_count',
    'repost_count',
    'post_count',
    'profile_fullness',
    'total_likes',
    'photo_count',
    'group_count',
    'post_date_diff'
]]

df2['posts_per_day'] = df2.apply(lambda row: row['post_count'] / row['post_date_diff'] if row['post_date_diff'] != 0 else 0, axis=1)

df2 = df2[[
    'friend_count',
    'repost_count',
    'post_count',
    'profile_fullness',
    'total_likes',
    'photo_count',
    'group_count',
    'posts_per_day'
]]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df2)

df_scaled[:, 0] *= 1.5
df['ei_cluster'] = ei_model.predict(df_scaled)

df2 = df[[
    'repost_count',
    'sad',
    'fun',
    'angry',
    'fear',
    'disgust',
    'mat',
    'sad_unique',
    'fun_unique',
    'angry_unique',
    'fear_unique',
    'disgust_unique',
    'mat_unique',
]].dropna()


scaler = StandardScaler()
df_scaled = scaler.fit_transform(df2)

df_scaled[:, 0] *= 0.94
# df_scaled[:, 1] *= 1.1
df['as_cluster'] = as_model.predict(df_scaled)

df2 = df[[
    'unique_word_stat',
    'post_len_std'
]].fillna(0)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df2)

df['gr_cluster'] = gr_model.predict(df_scaled)

print(
    df['ei_cluster'].describe(),
    df['as_cluster'].describe(),
    df['gr_cluster'].describe()
)
df['result'] = df.apply(lambda row: f"{((1 if (row['age'] == 14 or row['age'] == 15) else (2 if (row['age'] == row['age'] == 16 or row['age'] == 17) else 3)))}{'E' if row['ei_cluster'] == 1 else 'I'}{'A' if row['as_cluster'] == 0 else 'S'}{'G' if row['gr_cluster'] == 1 else 'R'}", axis=1)

print(df['result'])
df[['vk_id','result']].to_csv('out.csv')
# users = list(map(lambda x: GetUserPersonalityRequest(user_id=x[1]), users))
#users_responses = list(map(getUserPersonality, users))
#print(users_responses)
