from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# השתמש בקובץ המתוקן האולטימטיבי
from assets_data_prep import prepare_data

app = Flask(__name__)

# טעינת המודל
try:
    model_data = joblib.load('trained_model.pkl')

    if isinstance(model_data, dict):
        model = model_data['model']
        scaler = model_data.get('scaler', None)
        expected_columns = model_data['feature_names']
        print("✅ נטען מודל עם חבילה מלאה")
    else:
        model = model_data
        scaler = None
        expected_columns = model.feature_names_in_
        print("✅ נטען מודל בסיסי")

    print(f"✅ המודל מצפה ל-{len(expected_columns)} פיצ'רים")

except Exception as e:
    print(f"❌ שגיאה בטעינת המודל: {e}")
    exit(1)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None

    if request.method == 'POST':
        try:
            # איסוף הנתונים מהטופס
            data = {
                'area': float(request.form['area']),
                'room_num': float(request.form['room_num']),
                'monthly_arnona': float(request.form['monthly_arnona']),
                'floor': float(request.form['floor']),
                'property_type': request.form['property_type'],
                'is_furnished': int(request.form['is_furnished']),
                'has_parking': int(request.form['has_parking']),
                'neighborhood': request.form['neighborhood']
            }

            print(">>> [API] נתוני הטופס:", data)

            # בדיקות תקינות בסיסיות
            if data['area'] <= 0 or data['area'] > 500:
                raise ValueError("שטח הדירה חייב להיות בין 1-500 מ\"ר")

            if data['room_num'] <= 0 or data['room_num'] > 10:
                raise ValueError("מספר החדרים חייב להיות בין 1-10")

            if data['monthly_arnona'] < 0 or data['monthly_arnona'] > 5000:
                raise ValueError("ארנונה חודשית חייבת להיות בין 0-5000 ש\"ח")

            # יצירת DataFrame
            df = pd.DataFrame([data])
            print(f">>> [API] DataFrame נוצר: {df.shape}")

            # עיבוד הנתונים עם הקובץ המתוקן
            try:
                processed = prepare_data(df, mode='test', expected_columns=expected_columns)
                print(f">>> [API] נתונים מעובדים: {processed.shape}")

                # בדיקה שיש ערכים לא-אפס
                non_zero_count = (processed != 0).sum().sum()
                print(f">>> [API] ערכים לא-אפס: {non_zero_count}")

                if non_zero_count == 0:
                    raise ValueError("כל הערכים הפכו לאפס אחרי העיבוד")

                # חיזוי
                if scaler:
                    processed_scaled = scaler.transform(processed)
                    prediction = model.predict(processed_scaled)[0]
                else:
                    prediction = model.predict(processed)[0]

                print(f">>> [API] חיזוי: {prediction}")

                # וידוא שהחיזוי הגיוני
                if prediction < 500 or prediction > 50000:
                    print(f"⚠️ [API] אזהרה: חיזוי חריג - {prediction}")

            except Exception as prep_error:
                print(f"❌ [API] שגיאה בעיבוד נתונים: {prep_error}")
                error_message = f"שגיאה בעיבוד הנתונים: {str(prep_error)}"

        except ValueError as ve:
            print(f"❌ [API] שגיאת תקינות: {ve}")
            error_message = str(ve)
        except Exception as e:
            print(f"❌ [API] שגיאה כללית: {e}")
            error_message = "אירעה שגיאה בחיזוי. אנא נסה שוב."

    return render_template('index.html', prediction=prediction, error_message=error_message)


if __name__ == '__main__':
    print("🚀 מפעיל שרת Flask מתוקן...")
    print("📍 היכנס לכתובת: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)