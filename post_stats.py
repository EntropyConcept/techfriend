from typing import Union, List, Dict, Tuple
import re
from collections import Counter
import yaml
from transformers import pipeline
import time


pipe = pipeline("text-classification", model="sismetanin/rubert-ru-sentiment-rusentiment")

def get_post_style(text: str) -> Tuple[str, float]:
    """
    Возвращает информацию о стиле текста

    Args: 
        text (str): Текст поста

    Returns: 
        str: тип стиля (принимает значения *positive*, *negative*, *neutral*, *skip*, *speech*)
    """

    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
    sentiments = pipe(text, **tokenizer_kwargs)

    d = {
        'LABEL_0': 'negative', 
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
        'LABEL_3': 'skip',
        'LABEL_4': 'speech'
    }

    return (d[sentiments[0]['label']], sentiments[0]['score'])

def get_count_of_style_changes(styles: List[str]) -> int:
    if styles == []:
        return 0

    changes_count = 0
    current_style = styles[0]

    for style in styles[1:]:
        if style == 'skip':
            continue
        
        if style != current_style:  # Если стиль изменился
            changes_count += 1
            current_style = style  # Обновляем текущий стиль

    return changes_count

def get_post_words(text: Union[List[str], str]) -> Dict[str, int]:
    """
    Возвращает статистику по кол-ву каждого слова в тексте или массиве текстов,
    игнорируя знаки препинания.
    
    Args:
        text (Union[List[str], str]): Текст поста или массив текстов постов.
    
    Returns:
        Dict[str, int]: ключ - слово, значение - кол-во упоминаний этого слова.
    """
    
    if isinstance(text, list):
        text = ' '.join(text)
    
    text = re.sub(r'[^\w\s]', '', str(text)).lower()
    words = text.split()

    word_count = Counter(words)
    
    return dict(word_count)


def get_posts_word_stat(words: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    """
    Возвращает статистику по всем словам, разделяя их на категории (зумерские слова, маты и тп)
    Args: 
        words (Dict[str, int]): Статистика каждого слова (можно получить в get_post_words)

    Returns: 
        Dict[str, Tuple[int, int]]: ключ - категория слов, значение - кол-во таких слов в тексте и кол-во уникальных таких слов
    """

    with open('./word_dicts.yaml', 'r', encoding='utf-8') as file:
        categories = yaml.safe_load(file)

    stats = {category: (0, 0) for category in categories}
    total_count = 0
    unique_count = 0
    other_count = 0
    other_unique = 0

    for word, count in words.items():
        found_in_category = False
        for category, category_words in categories.items():
            # print(category_words)
            if word in category_words:
                found_in_category = True
                stats[category] = (stats[category][0] + count, stats[category][1] + 1)
                break

        if not found_in_category:
            other_count += count
            other_unique += 1

        total_count += count
        unique_count += 1

    stats['other'] = (other_count, other_unique)
    stats['total'] = (total_count, unique_count)


    return stats
