import vk
from post_stats import get_post_style, get_count_of_style_changes
from datetime import datetime
import numpy as np

def profile_fullness_count(t: dict) -> int:
    c = 0
    for k, v in {
        'activities': 5,
        'about': 5,
        'books': 3,
        'career': 1,
        'city': 1,
        'education': 1,
        'has_photo': 1,
        'site': 1,
        'schools': 1,
        'games': 3,
        'interests': 5,
        'military': 1,
        'movies': 3,
        'music': 3,
        'occupation': 1,
        'personal': 3,
        'quotes': 5,
        'tv': 3,
        'universities': 1
    }.items():
        if k not in t:
            continue
        
        if t[k] and t[k] != '' and t[k] != []:
            c += v

    if 'personal' in t.keys():
        for k, v in t['personal'].items():
            if v and v != [] and v != '':
                c += 0.5

    return c

def get_user_stat(user_id: str, token) -> dict:
    api = vk.API(access_token=token)

    person_info = api.users.get(user_ids=user_id
        , v=5.199
        , fields='activities,about,books,bdate,career,connections,contacts,city,domain,education,exports,followers_count,has_photo,has_mobile,sex,site,schools,screen_name,status,games,interests,military,movies,music,nickname,occupation,personal,quotes,relation,tv,universities,is_closed'
    )[0]

    if person_info['is_closed']:
        return {
            'profile_fullness': 0.0,
            'is_closed': person_info['is_closed'],
        }
    return {
        'profile_fullness': profile_fullness_count(person_info),
        'is_closed': person_info['is_closed'],
        'info': person_info,
    }


def get_posts_features(user_id: int, token: str) -> dict:
    api = vk.API(access_token=token)

    t = api.wall.get(owner_id=user_id, count=100, v=5.199)

    a = []

    for elem in t['items']:
        if 'views' not in elem:
            a.append({
                'post_type': elem['post_type'],
                # 'views_count': elem['views'],
                'likes_count': elem['likes']['count'],
                'comments_count': elem['comments']['count'],
                'reposts_count': elem['reposts']['count'],
                'text': elem['text'],
                'date': datetime.utcfromtimestamp(elem['date'])
            })
            continue
        
        a.append({
                'post_type': elem['post_type'],
                'views_count': elem['views'],
                'likes_count': elem['likes']['count'],
                'comments_count': elem['comments']['count'],
                'reposts_count': elem['reposts']['count'],
                'text': elem['text'],
                'date': datetime.utcfromtimestamp(elem['date'])
            })

    post_texts = list(map(lambda x: x['text'], a))
    post_sentiments = list(map(lambda x: get_post_style(x)[0], post_texts))
    sentiment_changes = get_count_of_style_changes(post_sentiments)
    post_len_std = np.array(list(map(len, post_texts))).std()

    return {
        'total_posts_count': t['count'],
        'posts': a,
        'post_sentiments': {
            'negative': post_sentiments.count('negative'),
            'neutral': post_sentiments.count('neutral'),
            'positive': post_sentiments.count('positive'),
            'skip': post_sentiments.count('skip'),
            'speech': post_sentiments.count('speech'),
        },
        'sentiment_changes': sentiment_changes,
        'post_length_std': post_len_std,
    }
