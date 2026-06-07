import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter

# ─── Настройки отображения ───────────────────────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
COLORS = plt.cm.tab20.colors
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Вспомогательные функции ─────────────────────────────────────────────────

def load_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_images(folder):
    """Найти все изображения в папке."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    return sorted([p for p in Path(folder).rglob('*') if p.suffix.lower() in exts])


def find_labels(folder):
    """Найти все файлы аннотаций (.txt) в папке."""
    return sorted(Path(folder).rglob('*.txt'))


def read_label_file(label_path):
    """
    Читает YOLO-файл аннотаций.
    Формат: class_id  cx  cy  w  h
    Возвращает список (class_id, cx, cy, w, h).
    """
    annotations = []
    if not Path(label_path).exists():
        return annotations
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                annotations.append((cls, cx, cy, bw, bh))
    return annotations


def collect_all_annotations(dataset_path, splits=('train', 'valid', 'test')):
    """
    Собирает все аннотации по сплитам.
    Возвращает DataFrame: split, image, class_id, cx, cy, bw, bh
    """
    records = []
    image_info = {}  # img_stem -> (split, w, h)

    for split in splits:
        img_dir  = Path(dataset_path) / split / 'images'
        lbl_dir  = Path(dataset_path) / split / 'labels'
        if not img_dir.exists():
            print(f"[WARNING] Папка не найдена: {img_dir}")
            continue

        imgs = find_images(img_dir)
        for img_path in imgs:
            stem = img_path.stem
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                w, h = 0, 0
            image_info[stem] = (split, w, h)

            lbl_path = lbl_dir / (stem + '.txt')
            anns = read_label_file(lbl_path)
            if not anns:  # изображение без аннотаций
                records.append({'split': split, 'image': stem,
                                'class_id': -1, 'cx': None, 'cy': None,
                                'bw': None, 'bh': None,
                                'img_w': w, 'img_h': h})
            for cls, cx, cy, bw, bh in anns:
                records.append({'split': split, 'image': stem,
                                'class_id': cls, 'cx': cx, 'cy': cy,
                                'bw': bw, 'bh': bh,
                                'img_w': w, 'img_h': h})

    return pd.DataFrame(records), image_info


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 0 — Структура датасета
# ═════════════════════════════════════════════════════════════════════════════

def step0_dataset_structure(df, image_info, class_names):
    split_counts = df.groupby('split')['image'].nunique()
    ann_counts   = df[df['class_id'] >= 0].groupby('split').size()

    print("\n" + "="*60)
    print("ШАГ 0: Структура датасета")
    print("="*60)
    print(f"Всего уникальных классов : {len(class_names)}")
    print(f"Всего изображений        : {df['image'].nunique()}")
    print(f"Всего аннотаций          : {(df['class_id'] >= 0).sum()}")
    for sp in ('train', 'valid', 'test'):
        n_img = split_counts.get(sp, 0)
        n_ann = ann_counts.get(sp, 0)
        print(f"  {sp:6s}: {n_img:4d} изображений, {n_ann:5d} аннотаций")

    # Диаграмма разбиения
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [sp for sp in ('train', 'valid', 'test') if sp in split_counts.index]
    img_vals = [split_counts.get(sp, 0) for sp in labels]
    ann_vals = [ann_counts.get(sp, 0) for sp in labels]

    axes[0].pie(img_vals, labels=labels, autopct='%1.1f%%',
                colors=['#4C72B0', '#DD8452', '#55A868'], startangle=140)
    axes[0].set_title('Распределение изображений\nпо сплитам', fontsize=13)

    axes[1].bar(labels, ann_vals, color=['#4C72B0', '#DD8452', '#55A868'])
    axes[1].set_title('Количество аннотаций по сплитам', fontsize=13)
    axes[1].set_ylabel('Количество аннотаций')
    for i, v in enumerate(ann_vals):
        axes[1].text(i, v + 20, str(v), ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_dataset_split.png', dpi=150)
    plt.close()
    print("[Сохранено] fig1_dataset_split.png")


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 1 — Анализ количества и баланса классов
# ═════════════════════════════════════════════════════════════════════════════

def step1_class_balance(df, class_names):
    print("\n" + "="*60)
    print("ШАГ 1: Анализ количества и баланса классов")
    print("="*60)

    ann_df = df[df['class_id'] >= 0].copy()
    class_counts = ann_df['class_id'].value_counts().sort_values(ascending=False)

    # Сводная таблица
    table = pd.DataFrame({
        'Класс': [class_names[i] if i < len(class_names) else f'class_{i}'
                  for i in class_counts.index],
        'ID': class_counts.index,
        'Кол-во объектов': class_counts.values,
        'Доля, %': (class_counts.values / class_counts.sum() * 100).round(2)
    })
    print(table.to_string(index=False))

    # Топ-15 классов (горизонтальный bar chart)
    top_n = min(15, len(class_counts))
    top_ids    = class_counts.index[:top_n]
    top_labels = [class_names[i] if i < len(class_names) else f'class_{i}' for i in top_ids]
    top_vals   = class_counts.values[:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(top_n), top_vals[::-1],
                   color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n)))
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_labels[::-1], fontsize=9)
    ax.set_xlabel('Количество аннотаций', fontsize=11)
    ax.set_title(f'Топ-{top_n} классов по числу аннотаций', fontsize=13)
    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_class_balance.png', dpi=150)
    plt.close()
    print("[Сохранено] fig2_class_balance.png")

    # Все классы (полный bar chart)
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    all_labels = [class_names[i] if i < len(class_names) else f'c{i}'
                  for i in class_counts.index]
    ax2.bar(range(len(class_counts)), class_counts.values,
            color='steelblue', edgecolor='none')
    ax2.set_xticks(range(len(class_counts)))
    ax2.set_xticklabels(all_labels, rotation=90, fontsize=7)
    ax2.set_ylabel('Количество аннотаций')
    ax2.set_title('Распределение всех классов по количеству аннотаций', fontsize=13)
    ax2.axhline(class_counts.mean(), color='red', linestyle='--',
                label=f'Среднее: {class_counts.mean():.0f}')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_all_classes.png', dpi=150)
    plt.close()
    print("[Сохранено] fig3_all_classes.png")

    return table


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 2 — Примеры типичных изображений с аннотациями
# ═════════════════════════════════════════════════════════════════════════════

def step2_sample_images(df, dataset_path, class_names, n=9):
    print("\n" + "="*60)
    print("ШАГ 2: Примеры типичных изображений с аннотациями")
    print("="*60)

    # Берём изображения из train с наибольшим числом аннотаций
    ann_df = df[df['class_id'] >= 0].copy()
    train_df = ann_df[ann_df['split'] == 'train']
    top_imgs = train_df.groupby('image').size().nlargest(n).index.tolist()

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    found = 0
    for img_stem in top_imgs:
        if found >= n:
            break
        img_rows = df[df['image'] == img_stem]
        split = img_rows['split'].iloc[0]

        # Найти изображение
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            p = Path(dataset_path) / split / 'images' / (img_stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            iw, ih = img.size
        except Exception:
            continue

        ax = axes[found]
        ax.imshow(img)

        anns = img_rows[img_rows['class_id'] >= 0]
        colors_per_class = {}
        for _, row in anns.iterrows():
            cid = int(row['class_id'])
            if cid not in colors_per_class:
                colors_per_class[cid] = COLORS[cid % len(COLORS)]
            cx, cy, bw, bh = row['cx'], row['cy'], row['bw'], row['bh']
            x1 = (cx - bw/2) * iw
            y1 = (cy - bh/2) * ih
            rect = patches.Rectangle((x1, y1), bw*iw, bh*ih,
                                      linewidth=2, edgecolor=colors_per_class[cid],
                                      facecolor='none')
            ax.add_patch(rect)
            label = class_names[cid] if cid < len(class_names) else f'class_{cid}'
            ax.text(x1, y1 - 3, label[:15], color='white', fontsize=6,
                    bbox=dict(facecolor=colors_per_class[cid], alpha=0.7, pad=1))

        ax.set_title(f'{img_stem[:20]} ({len(anns)} obj.)', fontsize=8)
        ax.axis('off')
        found += 1

    for i in range(found, n):
        axes[i].axis('off')

    plt.suptitle('Примеры типичных изображений датасета TACO с аннотациями',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_sample_images.png', dpi=120)
    plt.close()
    print("[Сохранено] fig4_sample_images.png")


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 3 — Оценка качества изображений
# ═════════════════════════════════════════════════════════════════════════════

def step3_image_quality(df, image_info):
    print("\n" + "="*60)
    print("ШАГ 3: Оценка качества изображений")
    print("="*60)

    widths  = [v[1] for v in image_info.values() if v[1] > 0]
    heights = [v[2] for v in image_info.values() if v[2] > 0]
    aspects = [w/h for w, h in zip(widths, heights) if h > 0]

    print(f"Ширина  — min: {min(widths)}, max: {max(widths)}, median: {int(np.median(widths))}")
    print(f"Высота  — min: {min(heights)}, max: {max(heights)}, median: {int(np.median(heights))}")
    print(f"Соотношение сторон — min: {min(aspects):.2f}, max: {max(aspects):.2f}, median: {np.median(aspects):.2f}")

    below_512 = sum(1 for w, h in zip(widths, heights) if w < 512 or h < 512)
    print(f"Изображений меньше 512×512 пикселей: {below_512} ({below_512/len(widths)*100:.1f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(widths, bins=30, color='steelblue', edgecolor='white')
    axes[0].set_title('Распределение ширины изображений', fontsize=11)
    axes[0].set_xlabel('Ширина, пикселей')
    axes[0].set_ylabel('Количество')

    axes[1].hist(heights, bins=30, color='darkorange', edgecolor='white')
    axes[1].set_title('Распределение высоты изображений', fontsize=11)
    axes[1].set_xlabel('Высота, пикселей')

    axes[2].hist(aspects, bins=30, color='seagreen', edgecolor='white')
    axes[2].set_title('Распределение соотношения сторон (W/H)', fontsize=11)
    axes[2].set_xlabel('Соотношение сторон')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_image_quality.png', dpi=150)
    plt.close()
    print("[Сохранено] fig5_image_quality.png")

    return widths, heights, aspects


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 4 — Анализ аннотаций
# ═════════════════════════════════════════════════════════════════════════════

def step4_annotation_analysis(df):
    print("\n" + "="*60)
    print("ШАГ 4: Анализ аннотаций")
    print("="*60)

    ann_df = df[df['class_id'] >= 0].copy()

    # Кол-во объектов на изображении
    objs_per_img = ann_df.groupby('image').size()
    print(f"Аннотаций на изображение — min: {objs_per_img.min()}, "
          f"max: {objs_per_img.max()}, "
          f"mean: {objs_per_img.mean():.2f}, "
          f"median: {objs_per_img.median():.1f}")

    # Размеры bounding box (нормализованные)
    bw_vals = ann_df['bw'].dropna().values
    bh_vals = ann_df['bh'].dropna().values
    area    = bw_vals * bh_vals

    print(f"Нормализованная ширина BB  — mean: {bw_vals.mean():.4f}, median: {np.median(bw_vals):.4f}")
    print(f"Нормализованная высота BB  — mean: {bh_vals.mean():.4f}, median: {np.median(bh_vals):.4f}")
    print(f"Площадь BB (нормализованная) — mean: {area.mean():.5f}, median: {np.median(area):.5f}")

    small_objs = (area < 0.01).sum()
    print(f"Малые объекты (площадь < 1% кадра): {small_objs} ({small_objs/len(area)*100:.1f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Кол-во объектов на изображение
    axes[0].hist(objs_per_img.values, bins=range(1, min(objs_per_img.max()+2, 25)),
                 color='steelblue', edgecolor='white', align='left')
    axes[0].set_title('Количество объектов на изображении', fontsize=11)
    axes[0].set_xlabel('Кол-во объектов')
    axes[0].set_ylabel('Кол-во изображений')

    # Размер BB (ширина)
    axes[1].hist(bw_vals, bins=40, color='darkorange', edgecolor='white')
    axes[1].set_title('Нормализованная ширина BB', fontsize=11)
    axes[1].set_xlabel('Нормализованная ширина')

    # Scatter: ширина × высота BB
    sample_mask = np.random.choice(len(bw_vals), min(2000, len(bw_vals)), replace=False)
    axes[2].scatter(bw_vals[sample_mask], bh_vals[sample_mask],
                    alpha=0.3, s=8, color='seagreen')
    axes[2].set_title('Соотношение ширины и высоты BB\n(выборка 2000 объектов)', fontsize=11)
    axes[2].set_xlabel('Нормализованная ширина')
    axes[2].set_ylabel('Нормализованная высота')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_annotation_analysis.png', dpi=150)
    plt.close()
    print("[Сохранено] fig6_annotation_analysis.png")

    # Распределение центров BB по 9 зонам кадра
    zone_df = ann_df[['cx', 'cy']].dropna().copy()
    zone_df['x_zone'] = pd.cut(
        zone_df['cx'],
        bins=[0, 1/3, 2/3, 1],
        labels=['левая', 'центральная', 'правая'],
        include_lowest=True
    )
    zone_df['y_zone'] = pd.cut(
        zone_df['cy'],
        bins=[0, 1/3, 2/3, 1],
        labels=['верхняя', 'средняя', 'нижняя'],
        include_lowest=True
    )
    zone_df['zone'] = zone_df['y_zone'].astype(str) + ' ' + zone_df['x_zone'].astype(str)

    zone_order = [
        'верхняя левая', 'верхняя центральная', 'верхняя правая',
        'средняя левая', 'средняя центральная', 'средняя правая',
        'нижняя левая', 'нижняя центральная', 'нижняя правая',
    ]
    zone_labels = [
        'Верхняя левая', 'Верхняя центральная', 'Верхняя правая',
        'Средняя левая', 'Центральная', 'Средняя правая',
        'Нижняя левая', 'Нижняя центральная', 'Нижняя правая',
    ]
    zone_counts = zone_df['zone'].value_counts().reindex(zone_order, fill_value=0)
    zone_pct = zone_counts / len(zone_df) * 100

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    colors = ['#4C78A8'] * len(zone_order)
    colors[4] = '#2F855A'
    bars = ax2.barh(zone_labels, zone_pct.values, color=colors, edgecolor='#26384D', linewidth=0.5)
    ax2.invert_yaxis()
    ax2.set_xlabel('Доля аннотаций, %')
    ax2.set_title('Распределение центров ограничивающих рамок по зонам изображения', fontsize=12)
    ax2.grid(axis='x', color='#E0E0E0', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.spines[['top', 'right']].set_visible(False)

    for bar, count, pct in zip(bars, zone_counts.values, zone_pct.values):
        ax2.text(
            pct + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f'{count} ({pct:.1f}%)',
            va='center',
            fontsize=9
        )

    center_pct = zone_pct.loc['средняя центральная']
    ax2.text(
        0,
        len(zone_order) + 0.25,
        f'Всего аннотаций: {len(zone_df)}. Центральная зона содержит {center_pct:.1f}% объектов.',
        fontsize=9,
        color='#555555'
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_bbox_centers_heatmap.png', dpi=150)
    plt.close()
    print("[Сохранено] fig7_bbox_centers_heatmap.png")

    return objs_per_img, bw_vals, bh_vals, area


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 5 — Оценка качества разметки
# ═════════════════════════════════════════════════════════════════════════════

def step5_markup_quality(df, image_info, dataset_path):
    print("\n" + "="*60)
    print("ШАГ 5: Оценка качества разметки")
    print("="*60)

    total_images = df['image'].nunique()
    unannotated  = df[df['class_id'] == -1]['image'].nunique()
    annotated    = total_images - unannotated
    ann_df = df[df['class_id'] >= 0].copy()

    # Подозрительные аннотации: BB выходит за пределы [0,1]
    bad_cx = ((ann_df['cx'] < 0) | (ann_df['cx'] > 1)).sum()
    bad_cy = ((ann_df['cy'] < 0) | (ann_df['cy'] > 1)).sum()
    bad_bw = ((ann_df['bw'] <= 0) | (ann_df['bw'] > 1)).sum()
    bad_bh = ((ann_df['bh'] <= 0) | (ann_df['bh'] > 1)).sum()

    # Очень маленькие BB (< 0.1% площади кадра)
    ann_df['area'] = ann_df['bw'] * ann_df['bh']
    tiny = (ann_df['area'] < 0.001).sum()

    # Дубликаты (одинаковые bbox в одном изображении)
    dup_check = ann_df.groupby(['image', 'class_id', 'cx', 'cy', 'bw', 'bh']).size()
    duplicates = (dup_check > 1).sum()

    print(f"Всего изображений          : {total_images}")
    print(f"С аннотациями              : {annotated} ({annotated/total_images*100:.1f}%)")
    print(f"Без аннотаций (фон)        : {unannotated} ({unannotated/total_images*100:.1f}%)")
    print(f"Подозрительные cx          : {bad_cx}")
    print(f"Подозрительные cy          : {bad_cy}")
    print(f"Подозрительные bw          : {bad_bw}")
    print(f"Подозрительные bh          : {bad_bh}")
    print(f"Очень маленькие BB (<0.1%) : {tiny}")
    print(f"Дублирующие аннотации      : {duplicates}")

    # Круговая диаграмма аннотированных vs. неаннотированных
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].pie([annotated, unannotated],
                labels=['С аннотациями', 'Без аннотаций (фон)'],
                autopct='%1.1f%%', colors=['#4C72B0', '#DD8452'])
    axes[0].set_title('Доля изображений с аннотациями', fontsize=12)

    # Статистика проблем разметки
    problems = {
        'Без аннотаций': unannotated,
        'Ошибки cx/cy': bad_cx + bad_cy,
        'Ошибки bw/bh': bad_bw + bad_bh,
        'Микро-объекты\n(<0.1% кадра)': tiny,
        'Дубликаты': duplicates,
    }
    axes[1].bar(problems.keys(), problems.values(),
                color=['#DD8452', '#C44E52', '#C44E52', '#937860', '#8172B2'])
    axes[1].set_title('Потенциальные проблемы разметки', fontsize=12)
    axes[1].set_ylabel('Количество')
    for i, (k, v) in enumerate(problems.items()):
        axes[1].text(i, v + 0.5, str(v), ha='center', fontsize=10)
    plt.xticks(rotation=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig8_markup_quality.png', dpi=150)
    plt.close()
    print("[Сохранено] fig8_markup_quality.png")

    return annotated, unannotated, tiny, duplicates


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Анализ датасета TACO (YOLO-формат)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Путь к папке с датасетом (содержит data.yaml)')
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"[ERROR] Папка не существует: {dataset_path}")
        sys.exit(1)

    # Загрузка метаданных
    yaml_candidates = list(dataset_path.glob('*.yaml')) + list(dataset_path.glob('*.yml'))
    if not yaml_candidates:
        print("[WARNING] data.yaml не найден. Попытка продолжить без имён классов.")
        class_names = [f'class_{i}' for i in range(100)]
    else:
        meta = load_yaml(yaml_candidates[0])
        class_names = meta.get('names', [])
        if isinstance(class_names, dict):
            class_names = [class_names[k] for k in sorted(class_names.keys())]
        print(f"[OK] Загружено {len(class_names)} классов из {yaml_candidates[0].name}")

    # Сбор аннотаций
    print("\n[...] Сканирование датасета...")
    df, image_info = collect_all_annotations(dataset_path)
    print(f"[OK] Собрано {len(df)} записей, {len(image_info)} изображений")

    # Шаги анализа
    step0_dataset_structure(df, image_info, class_names)
    class_table = step1_class_balance(df, class_names)
    step2_sample_images(df, dataset_path, class_names)
    widths, heights, aspects = step3_image_quality(df, image_info)
    objs_per_img, bw_vals, bh_vals, area = step4_annotation_analysis(df)
    step5_markup_quality(df, image_info, dataset_path)

    print("\n" + "="*60)
    print(f"Готово! Все графики сохранены в папку: {OUTPUT_DIR.resolve()}")
    print("="*60)


if __name__ == '__main__':
    main()
