"""
Скрипт первичного анализа Russian Sentiment Dataset
Курсовая работа: Раздел 4 — Первичный анализ набора текстовых данных

Структура датасета:
    sentiment_dataset.csv (~290 000+ строк, 181 МБ)
    Колонки:
        text  — текст отзыва (русский язык)
        label — тональность: 0=негативный, 1=нейтральный, 2=позитивный
        src   — источник: rureviews, geo, и другие

Установка зависимостей:
    pip install pandas matplotlib seaborn scikit-learn nltk pymorphy2

Запуск:
    python analyze_sentiment.py --dataset_path D:\\Kursivaia\\Glava4

Все графики сохраняются в папку ./output/
"""

import os, re, sys, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

plt.rcParams['font.family']       = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('./output')
OUTPUT_DIR.mkdir(exist_ok=True)

LABEL_MAP = {0: 'Негативный', 1: 'Нейтральный', 2: 'Позитивный'}
COLORS    = ['#C44E52', '#8172B2', '#4C72B0']

RUSSIAN_STOPS = set(stopwords.words('russian'))
EXTRA_STOPS   = {'это','так','вот','быть','как','в','и','к','на','не','но',
                 'а','что','он','она','они','мы','вы','я','за','из','по',
                 'до','от','со','её','его','их','все','для','же','ли','нет',
                 'да','ну','уже','ещё','тоже','при','очень','просто','если',
                 'можно','когда','бы','тут','там','был','была','были','буду',
                 'будет','которые','который','которая','свой','своя','своих'}
ALL_STOPS = RUSSIAN_STOPS | EXTRA_STOPS

try:
    import pymorphy2
    _morph    = pymorphy2.MorphAnalyzer()
    lemmatize = lambda w: _morph.parse(w)[0].normal_form
    HAVE_MORPH = True
    print('[OK] pymorphy2 — лемматизация включена')
except ImportError:
    lemmatize  = lambda w: w
    HAVE_MORPH = False
    print('[WARN] pymorphy2 не найден. Установи: pip install pymorphy2')


def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text):       return [w for w in text.split() if len(w) > 2]
def remove_stops(tokens): return [t for t in tokens if t not in ALL_STOPS]
def lemma(tokens):        return [lemmatize(t) for t in tokens]


# ── ШАГ 0: Загрузка ──────────────────────────────────────────────────────────
def step0_load(folder):
    print('\n' + '='*60 + '\nШАГ 0: Загрузка датасета\n' + '='*60)
    csvs = list(Path(folder).glob('*.csv'))
    if not csvs: print('[ERROR] CSV не найден'); sys.exit(1)
    csv_path = csvs[0]
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    print(f'Файл    : {csv_path.name}')
    print(f'Строк   : {len(df):,}')
    print(f'Колонки : {list(df.columns)}')
    print(f'Уникальных label: {sorted(df["label"].unique())}')
    print(f'\nТоп источников:\n{df["src"].value_counts().head(8).to_string()}')
    return df


# ── ШАГ 1: Классы, источники, длины ──────────────────────────────────────────
def step1_class_and_length(df):
    print('\n' + '='*60 + '\nШАГ 1: Классы, источники, длины текстов\n' + '='*60)
    lc = df['label'].value_counts().sort_index()
    for lbl, cnt in lc.items():
        print(f'  label={lbl} {LABEL_MAP.get(lbl,"?"):12s}: {cnt:>8,} ({cnt/len(df)*100:.1f}%)')

    df['word_len'] = df['text'].fillna('').apply(lambda x: len(str(x).split()))
    df['char_len'] = df['text'].fillna('').apply(len)
    print('\nСтатистика длины (слов):')
    print(df['word_len'].describe().round(2).to_string())

    lnames = [LABEL_MAP.get(i, str(i)) for i in lc.index]

    # Рис 1 — классы + длины
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].pie(lc.values, labels=lnames, autopct='%1.1f%%',
                colors=COLORS, startangle=130)
    axes[0].set_title('Распределение классов тональности', fontsize=12)

    axes[1].bar(lnames, lc.values, color=COLORS, edgecolor='white')
    axes[1].set_title('Количество примеров по классам', fontsize=12)
    axes[1].set_ylabel('Количество примеров')
    for i, v in enumerate(lc.values):
        axes[1].text(i, v + len(df)*0.003, f'{v:,}', ha='center', fontsize=9)

    clip99 = df['word_len'].quantile(0.99)
    axes[2].hist(df['word_len'].clip(upper=clip99), bins=50,
                 color='steelblue', edgecolor='white')
    axes[2].axvline(df['word_len'].median(), color='red', linestyle='--',
                    label=f'Медиана: {df["word_len"].median():.0f}')
    axes[2].set_title('Распределение длины текстов (слов)', fontsize=12)
    axes[2].set_xlabel('Количество слов'); axes[2].set_ylabel('Текстов')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_class_distribution.png', dpi=150); plt.close()
    print('[Сохранено] fig1_class_distribution.png')

    # Рис 2 — boxplot по классам
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    groups = [df.loc[df['label'] == lbl, 'word_len'].clip(upper=300).values
              for lbl in sorted(lc.index)]
    bp = ax2.boxplot(groups, labels=lnames, patch_artist=True)
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax2.set_title('Длина текстов по классам тональности', fontsize=12)
    ax2.set_xlabel('Класс'); ax2.set_ylabel('Слов (до 300)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_length_by_class.png', dpi=150); plt.close()
    print('[Сохранено] fig2_length_by_class.png')

    # Рис 3 — источники
    src_counts = df['src'].value_counts()
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    top_src = src_counts.head(10)
    axes3[0].barh(top_src.index[::-1], top_src.values[::-1], color='#4C72B0')
    axes3[0].set_title('Источники данных (топ-10)', fontsize=12)
    axes3[0].set_xlabel('Количество записей')
    for i, v in enumerate(top_src.values[::-1]):
        axes3[0].text(v + len(df)*0.001, i, f'{v:,}', va='center', fontsize=8)

    top5_src = src_counts.head(5).index
    src_by_cls = df[df['src'].isin(top5_src)].groupby(['src', 'label']).size().unstack(fill_value=0)
    src_by_cls.columns = [LABEL_MAP.get(c, str(c)) for c in src_by_cls.columns]
    src_by_cls.plot(kind='bar', stacked=True, ax=axes3[1], color=COLORS, edgecolor='white')
    axes3[1].set_title('Распределение классов по топ-5 источникам', fontsize=12)
    axes3[1].set_xlabel('Источник'); axes3[1].set_ylabel('Записей')
    axes3[1].tick_params(axis='x', rotation=30); axes3[1].legend(title='Тональность')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_sources.png', dpi=150); plt.close()
    print('[Сохранено] fig3_sources.png')

    return df, lc, src_counts


# ── ШАГ 2: Очистка и лемматизация (выборка) ──────────────────────────────────
SAMPLE_SIZE = 30_000

def step2_clean(df):
    print('\n' + '='*60 + f'\nШАГ 2: Очистка + лемматизация (выборка {SAMPLE_SIZE:,})\n' + '='*60)
    sample = df.sample(SAMPLE_SIZE, random_state=42).copy()
    print('[...] Очистка...')
    sample['text_clean'] = sample['text'].apply(clean_text)
    print('[...] Токенизация + стоп-слова...')
    sample['tokens'] = sample['text_clean'].apply(lambda t: remove_stops(tokenize(t)))
    if HAVE_MORPH:
        print('[...] Лемматизация (3–7 минут)...')
        sample['tokens_lemma'] = sample['tokens'].apply(lemma)
    else:
        sample['tokens_lemma'] = sample['tokens']
    print('\nПример:')
    row = sample.sample(1, random_state=5).iloc[0]
    print(f'  Исходный : {str(row["text"])[:80]}')
    print(f'  Леммы    : {row["tokens_lemma"][:10]}')
    return sample


# ── ШАГ 3: Частотный анализ ───────────────────────────────────────────────────
def step3_freq(sample):
    print('\n' + '='*60 + '\nШАГ 3: Частотный анализ\n' + '='*60)
    all_tok = [t for toks in sample['tokens_lemma'] for t in toks]
    freq = Counter(all_tok)
    print('Топ-20 слов:')
    for w, c in freq.most_common(20):
        print(f'  {w:20s}: {c:,}')

    w20, c20 = zip(*freq.most_common(20))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(list(w20)[::-1], list(c20)[::-1], color='steelblue')
    ax.set_title(f'Топ-20 слов (выборка {SAMPLE_SIZE:,} записей)', fontsize=12)
    ax.set_xlabel('Частота')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_top_words.png', dpi=150); plt.close()
    print('[Сохранено] fig4_top_words.png')

    labels = sorted(sample['label'].unique())
    fig2, axes = plt.subplots(1, len(labels), figsize=(7*len(labels), 6))
    if len(labels) == 1: axes = [axes]
    for i, lbl in enumerate(labels):
        sub  = sample[sample['label'] == lbl]
        toks = [t for ts in sub['tokens_lemma'] for t in ts]
        f    = Counter(toks).most_common(15)
        if not f: continue
        ws, cs = zip(*f)
        axes[i].barh(list(ws)[::-1], list(cs)[::-1], color=COLORS[i % 3])
        axes[i].set_title(LABEL_MAP.get(lbl, str(lbl)), fontsize=11)
        axes[i].set_xlabel('Частота')
    plt.suptitle('Топ-15 слов по классам тональности', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_words_by_class.png', dpi=150); plt.close()
    print('[Сохранено] fig5_words_by_class.png')
    return freq


# ── ШАГ 4: TF-IDF ─────────────────────────────────────────────────────────────
def step4_tfidf(sample):
    print('\n' + '='*60 + '\nШАГ 4: TF-IDF\n' + '='*60)
    corpus = sample['tokens_lemma'].apply(lambda x: ' '.join(x))
    corpus = corpus[corpus.str.strip() != '']

    tfidf = TfidfVectorizer(max_features=2000, min_df=5, ngram_range=(1, 2))
    X     = tfidf.fit_transform(corpus)
    feat  = np.array(tfidf.get_feature_names_out())

    mean_v  = np.asarray(X.mean(axis=0)).flatten()
    top20_i = mean_v.argsort()[-20:][::-1]
    print('Топ-20 слов по TF-IDF:')
    for w, v in zip(feat[top20_i], mean_v[top20_i]):
        print(f'  {w:25s}: {v:.5f}')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(feat[top20_i][::-1], mean_v[top20_i][::-1], color='darkorange')
    ax.set_title('Топ-20 слов/биграмм по среднему TF-IDF', fontsize=12)
    ax.set_xlabel('Средний TF-IDF')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_tfidf_top.png', dpi=150); plt.close()
    print('[Сохранено] fig6_tfidf_top.png')

    labels = sorted(sample.loc[corpus.index, 'label'].unique())
    fig2, axes = plt.subplots(1, len(labels), figsize=(7*len(labels), 6))
    if len(labels) == 1: axes = [axes]
    for i, lbl in enumerate(labels):
        mask = sample.loc[corpus.index, 'label'] == lbl
        mc   = np.asarray(X[mask.values].mean(axis=0)).flatten()
        t10  = mc.argsort()[-10:][::-1]
        axes[i].barh(feat[t10][::-1], mc[t10][::-1], color=COLORS[i % 3])
        axes[i].set_title(LABEL_MAP.get(lbl, str(lbl)), fontsize=11)
        axes[i].set_xlabel('Средний TF-IDF')
    plt.suptitle('Топ-10 ключевых слов/биграмм по классам (TF-IDF)', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_tfidf_by_class.png', dpi=150); plt.close()
    print('[Сохранено] fig7_tfidf_by_class.png')
    return tfidf, X, corpus


# ── ШАГ 5: Качество ───────────────────────────────────────────────────────────
def step5_quality(df):
    print('\n' + '='*60 + '\nШАГ 5: Качество данных\n' + '='*60)
    total      = len(df)
    miss_text  = df['text'].isnull().sum()
    miss_lbl   = df['label'].isnull().sum()
    empty      = (df['text'].fillna('').str.strip() == '').sum()
    dups       = df.duplicated(subset=['text']).sum()
    short      = (df.get('word_len', pd.Series(dtype=int)) <= 2).sum()

    print(f'Всего         : {total:,}')
    print(f'Пропуски text : {miss_text} ({miss_text/total*100:.3f}%)')
    print(f'Пропуски label: {miss_lbl}')
    print(f'Пустые тексты : {empty}')
    print(f'Дубликаты     : {dups:,} ({dups/total*100:.2f}%)')
    print(f'Тексты ≤2 сл  : {short:,} ({short/total*100:.2f}%)')

    probs = {'Пропуски\ntext': miss_text, 'Пропуски\nlabel': miss_lbl,
             'Пустые': empty, 'Дубликаты': dups, 'Очень\nкороткие': int(short)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].pie([total - dups, dups], labels=['Уникальные', 'Дубликаты'],
                autopct='%1.2f%%', colors=['#55A868', '#C44E52'], startangle=140)
    axes[0].set_title('Доля дубликатов', fontsize=12)
    axes[1].bar(probs.keys(), probs.values(),
                color=['#DD8452','#C44E52','#8172B2','#937860','#4C72B0'])
    axes[1].set_title('Проблемы качества данных', fontsize=12)
    axes[1].set_ylabel('Количество записей')
    for i, (k, v) in enumerate(probs.items()):
        axes[1].text(i, v + total*0.001, f'{v:,}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig8_quality.png', dpi=150); plt.close()
    print('[Сохранено] fig8_quality.png')
    return miss_text, miss_lbl, dups, int(short)


# ── ШАГ 6: Информационный поиск ──────────────────────────────────────────────
def step6_retrieval(sample, tfidf, X, corpus):
    print('\n' + '='*60 + '\nШАГ 6: Информационный поиск\n' + '='*60)
    queries = [
        'хорошее качество быстрая доставка',
        'плохо не рекомендую разочарован',
        'нормально ничего особенного',
    ]
    texts_list = sample['text'].tolist()
    fig, axes  = plt.subplots(1, 3, figsize=(21, 5))
    for idx, query in enumerate(queries):
        q_tok = remove_stops(tokenize(clean_text(query)))
        if HAVE_MORPH: q_tok = lemma(q_tok)
        q_vec = tfidf.transform([' '.join(q_tok)])
        sims  = cosine_similarity(q_vec, X).flatten()
        top5  = sims.argsort()[-5:][::-1]
        print(f'\nЗапрос: «{query}»')
        for rank, doc_i in enumerate(top5, 1):
            lbl = LABEL_MAP.get(int(sample['label'].iloc[doc_i]), '?')
            txt = str(texts_list[doc_i])[:60]
            print(f'  Топ-{rank} (sim={sims[doc_i]:.4f}, {lbl}): {txt}')
        axes[idx].barh([f'Топ-{r+1}' for r in range(5)],
                       sims[top5][::-1], color='steelblue')
        axes[idx].set_xlim(0, max(sims[top5]) * 1.4 + 0.01)
        axes[idx].set_title(f'«{query[:28]}»', fontsize=10)
        axes[idx].set_xlabel('Cosine Similarity')
    plt.suptitle('Информационный поиск (TF-IDF + cosine similarity)', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9_retrieval.png', dpi=150); plt.close()
    print('[Сохранено] fig9_retrieval.png')


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    args   = parser.parse_args()
    folder = Path(args.dataset_path)
    if not folder.exists(): print(f'[ERROR] {folder}'); sys.exit(1)

    df                    = step0_load(folder)
    df, lc, src_cnts      = step1_class_and_length(df)
    sample                = step2_clean(df)
    freq                  = step3_freq(sample)
    tfidf, X, corpus      = step4_tfidf(sample)
    step5_quality(df)
    step6_retrieval(sample, tfidf, X, corpus)

    print('\n' + '='*60)
    print(f'Готово! Графики: {OUTPUT_DIR.resolve()}')
    print('='*60)

if __name__ == '__main__':
    main()
