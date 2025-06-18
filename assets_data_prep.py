import pandas as pd
import numpy as np


def prepare_data(df, mode='train', expected_columns=None):
    """
    מכין נתונים למודל - תיקון אולטימטיבי
    """
    df = df.copy()
    print(f"🔄 [ULTIMATE] מעבד נתונים במצב: {mode}")
    print(f"📊 צורת נתונים התחלתית: {df.shape}")

    # הסרת שורות עם מחיר חסר (רק באימון)
    if mode == 'train' and 'price' in df.columns:
        df = df.dropna(subset=['price'])

    # טיפול במספר חדרים
    if 'room_num' in df.columns:
        df['room_num'] = df['room_num'].replace(0, np.nan)
        if 'neighborhood' in df.columns and mode == 'train':
            df['room_num'] = df.groupby('neighborhood')['room_num'].transform(lambda x: x.fillna(x.mean()))
        df['room_num'] = df['room_num'].fillna(df['room_num'].median())

    # המרה של עמודות מספריות
    numeric_cols = ['area', 'num_of_payments', 'monthly_arnona', 'building_tax',
                    'total_floors', 'num_of_images', 'distance_from_center']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # טיפול בקומה
    if 'floor' in df.columns:
        df['floor'] = df['floor'].astype(str)
        df['floor'] = df['floor'].apply(lambda x: x.split(' מתוך')[0] if 'מתוך' in x else x)
        floor_map = {'ground': 0, 'basement': -1, 'missing': np.nan, 'קומת קרקע': 0, 'קרקע': 0, 'מרתף': -1}
        df['floor'] = df['floor'].replace(floor_map)
        df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
        df['floor'] = df['floor'].fillna(df['floor'].median())

    # סינון חריגים (רק באימון)
    if mode == 'train':
        if 'area' in df.columns:
            df = df[(df['area'] > 10) & (df['area'] < 300)]
        if 'price' in df.columns:
            df = df[(df['price'] > 1000) & (df['price'] < 30000)]
        if 'distance_from_center' in df.columns:
            df = df[df['distance_from_center'] < 20000]
        if 'floor' in df.columns:
            df = df[df['floor'] < 100]

    # יצירת פיצ'רים מתיאור (אם קיים)
    if 'description' in df.columns:
        desc = df['description'].astype(str)
        df['has_furniture_words'] = desc.str.contains("furnished|furniture", case=False).astype(int)
        df['description_length'] = desc.apply(len)
        df['has_renovated_word'] = desc.str.contains("משופצת", case=False).astype(int)
        df['has_balcony_word'] = desc.str.contains("מרפסת", case=False).astype(int)
        df = df.drop(columns=['description'])

    # יצירת פיצ'רים מורכבים
    if 'area' in df.columns and 'room_num' in df.columns:
        df['area_per_room'] = df['area'] / df['room_num'].replace(0, np.nan)

    if 'garden_area' in df.columns:
        df['has_garden'] = (df['garden_area'].fillna(0) > 0).astype(int)

    if 'num_of_images' in df.columns:
        df['image_count_level'] = pd.cut(df['num_of_images'], bins=[-1, 0, 3, 6, 100],
                                         labels=['none', 'low', 'med', 'high'])

    if 'monthly_arnona' in df.columns and 'area' in df.columns:
        df['arnona_per_sqm'] = df['monthly_arnona'] / df['area'].replace(0, np.nan)

    if 'building_tax' in df.columns and 'area' in df.columns:
        df['tax_per_sqm'] = df['building_tax'] / df['area'].replace(0, np.nan)

    if 'floor' in df.columns and 'total_floors' in df.columns:
        df['floor_ratio'] = df['floor'] / df['total_floors'].replace(0, np.nan)

    if 'property_type' in df.columns:
        df['is_luxury_type'] = df['property_type'].isin(['גג/פנטהאוז', "פרטי/קוטג'", 'דירת גן']).astype(int)

    if 'is_furnished' in df.columns and 'has_parking' in df.columns:
        df['furnished_and_parking'] = df['is_furnished'] * df['has_parking']

    print(f"📊 [ULTIMATE] לפני המרת קטגוריות: {df.shape}")
    print(f"📋 [ULTIMATE] עמודות קיימות: {list(df.columns)}")

    # שמירת ערכים מספריים לפני המרת קטגוריות
    numeric_data = {}
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and col != 'price':
            numeric_data[col] = df[col].iloc[0]

    print(f"💾 [ULTIMATE] שמרתי נתונים מספריים: {numeric_data}")

    # המרה לדמי משתנים
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        print(f"🏷️ [ULTIMATE] מעבד עמודות קטגוריאליות: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"📊 [ULTIMATE] אחרי המרת קטגוריות: {df.shape}")

    # הסרת עמודות מיותרות
    cols_to_drop = ['days_to_enter', 'garden_area', 'ac']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)

    # הסרת עמודות עם שונות נמוכה
    low_var_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if low_var_cols:
        df = df.drop(columns=low_var_cols, errors='ignore')

    # הסרת עמודות נוספות רק באימון
    if mode == 'train':
        print("🗑️ [ULTIMATE] מסיר עמודות נוספות רק באימון")
        cols_to_drop_pps = [
            'floor', 'num_of_payments', 'is_renovated',
            'address', 'num_of_images', 'distance_from_center',
            'description_length', 'has_bars', 'has_furniture_words'
        ]
        existing_cols_to_drop_pps = [col for col in cols_to_drop_pps if col in df.columns]
        if existing_cols_to_drop_pps:
            df = df.drop(columns=existing_cols_to_drop_pps)

    # טיפול בערכים אינסופיים
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # הסרת חריגים על בסיס z-score (רק באימון)
    if mode == 'train':
        try:
            from scipy.stats import zscore
            z_cols = [col for col in ['price', 'area', 'arnona_per_sqm'] if col in df.columns]
            if z_cols:
                z_scores = np.abs(zscore(df[z_cols]))
                outlier_mask = (z_scores < 3).all(axis=1)
                df = df[outlier_mask]
        except ImportError:
            pass

    # הסרת עמודת המחיר במצב test
    if mode == 'test' and 'price' in df.columns:
        df = df.drop(columns=['price'])

    print(f"📊 [ULTIMATE] לפני התאמת עמודות: {df.shape}")

    # התאמה לעמודות הצפויות במצב test
    if mode == 'test' and expected_columns is not None:
        print(f"🎯 [ULTIMATE] מתאים ל-{len(expected_columns)} עמודות צפויות")

        # שמירת העמודות הקיימות עם ערכים לא-אפס
        preserved_data = {}
        for col in df.columns:
            if col in expected_columns and df[col].iloc[0] != 0:
                preserved_data[col] = df[col].iloc[0]

        # גם שמירת הנתונים המספריים שלא הלכו לאיבוד
        for col_name, value in numeric_data.items():
            if col_name in expected_columns and value != 0:
                preserved_data[col_name] = value

        print(f"💾 [ULTIMATE] שומר עמודות עם ערכים: {preserved_data}")

        # יצירת DataFrame חדש עם כל העמודות הצפויות
        new_df = pd.DataFrame(0, index=df.index, columns=expected_columns)

        # החזרת הערכים השמורים
        for col, value in preserved_data.items():
            new_df[col] = value

        df = new_df

    # בדיקה סופית
    non_zero_count = (df != 0).sum().sum()
    total_values = df.shape[0] * df.shape[1]
    print(f"✅ [ULTIMATE] סיים עיבוד: {df.shape}, ערכים לא-אפס: {non_zero_count}/{total_values}")

    if mode == 'test' and non_zero_count > 0:
        print("🎉 [ULTIMATE] יש ערכים לא-אפס!")
        non_zero_cols = []
        for col in df.columns:
            if df[col].iloc[0] != 0:
                non_zero_cols.append(f"{col}: {df[col].iloc[0]}")
        print(f"📋 [ULTIMATE] עמודות עם ערכים: {non_zero_cols[:10]}")  # רק 10 הראשונות

    return df