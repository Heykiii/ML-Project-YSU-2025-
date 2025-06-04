# ML-Project-YSU-2025
 FIFA 23 Player Rating Prediction | ML Project with 5 Regression Models | XGBoost R²=0.892 | Python &amp; Google Colab
# FIFA 23 Խաղացողների Գնահատականի Կանխատեսում  
**Machine Learning Project - Regression Analysis**

---

## 📋 Նախագծի Ընդհանուր Նկարագրություն

Այս նախագիծը նպատակ ունի կանխատեսել **FIFA 23** խաղի խաղացողների ընդհանուր գնահատականը (overall rating)՝ կիրառելով տարբեր **Machine Learning** (ML) մոդելներ: Ամբողջ գործընթացը իրականացվել է **Google Colab**-ում և ներառում է տվյալների վերլուծության ամբողջական փուլ՝ նախապատրաստումից մինչև վերջնական գնահատում:

### 🎯 Հիմնական Նպատակ

Կառուցել ճշգրիտ ML մոդել, որը կարող է կանխատեսել խաղացողի overall rating-ը՝ հիմնվելով նրա տարբեր հատկանիշների և վիճակագրական տվյալների վրա:

---

## 🗂️ Dataset-ի Մասին

- **Աղբյուր**: Kaggle - FIFA Player Stats Database
- **Չափս**: 18,000+ խաղացող
- **Հատկանիշներ**: 100+ տարբերատիպ առանձնահատկություններ (attributes, statistics, personal info)
- **Թիրախ**: Overall Rating (0-100)

#### 📊 Dataset-ում ներառված է
- Անձնական տվյալներ
- Տեխնիկական հմտություններ (օր.՝ shooting, passing, dribbling)
- Ֆիզիկական տվյալներ (օր.՝ speed, strength, height)
- Դիրքային վիճակագրություն
- Market value ու contract info

---

## 🔧 Տվյալների Նախապատրաստում

**1. Տվյալների մաքրում**  
- Անպետք սյունակների հեռացում (օր.՝ ID, URL, լուսանկարներ)  
- Բացակա արժեքների հայտնաբերում և լրացում  
- Տվյալների տեսակների ճշգրտում

**2. Կատեգորիկ տվյալների փոխակերպում**  
- Տեքստային արժեքները փոխարկվել են թվային (LabelEncoder՝ օրինակ՝ Position, Club, Nationality)

**3. Feature Scaling**  
- Բոլոր հատկանիշները նույն սանդղակում՝ օգտագործելով StandardScaler

**4. Արտաբերցիկների հայտնաբերում ու հեռացում**  
- IQR մեթոդ  
- Միայն ծայրահեղ արժեքների հեռացում (~2-3% տվյալների)

---

## 📊 Exploratory Data Analysis (EDA)

- **Բաշխման վերլուծություն**՝ թիրախի ու հատկանիշների համար (histogram-ներ)
- **Կորելյացիայի վերլուծություն**՝ correlation matrix, heatmap-ներ
- **Արտաբերցիկների վիզուալիզացիա**՝ boxplot-ներով
- **Կարևոր հատկանիշների բացահայտում**՝ bar chart-երով

---

## 🤖 Machine Learning Մոդելներ

Այս նախագծում փորձարկվել են հետևյալ 5 regression մոդելները․

1. **Linear Regression** — պարզ, baseline մոդել, հեշտ բացատրվող
2. **Random Forest Regressor** — ensemble մեթոդ, դիմանում է overfitting-ին, 100 ծառ
3. **Gradient Boosting Regressor** — հաջորդական boosting, բարձր ճշգրտություն
4. **Support Vector Regressor (SVR)** — kernel-based, լավ արդյունք բարդ տվյալների դեպքում
5. **XGBoost Regressor** — առաջադեմ gradient boosting, state-of-the-art արդյունքներ

---

## 📈 Մոդելների Գնահատում

Օգտագործված են հետևյալ գնահատման չափանիշները․

- **R² Score** — ցույց է տալիս, թե որքանով է մոդելը բացատրում թիրախի տատանումը
- **RMSE** — միջին քառակուսային սխալի արմատ (ցածր արժեքն ավելի լավ է)
- **MAE** — միջին բացարձակ սխալ (հեշտ մեկնաբանվող)
- **5-Fold Cross Validation** — յուրաքանչյուր մոդելի կայունության ստուգման համար

---

## 🏆 Արդյունքներ

| Մոդել           | R² Score | RMSE | MAE | Cross-Val R² |
|-----------------|----------|------|-----|--------------|
| XGBoost         | 0.892    | 2.34 | 1.87| 0.885        |
| Random Forest   | 0.879    | 2.48 | 1.92| 0.874        |
| Gradient Boosting|0.864    | 2.63 | 2.05| 0.859        |
| SVR             | 0.721    | 3.76 | 2.89| 0.715        |
| Linear Regression|0.687    | 3.98 | 3.12| 0.682        |

**Լավագույն մոդել**: `XGBoost Regressor`  
- Ամենաբարձր R² (0.892)
- Ամենացածր RMSE (2.34)
- Լավագույն Cross-Validation ցուցանիշ (0.885)
- Կայունություն տարբեր data splits-ների դեպքում

#### Լավագույն 10 հատկանիշները՝ ըստ կարևորության
1. Potential
2. Reactions
3. Composure
4. Ball Control
5. Short Passing
6. Long Passing
7. Dribbling
8. Vision
9. Finishing
10. Shot Power

---

## 💡 Տեխնիկական Մանրամասներ

- **Regression** է ընտրվել, քանի որ overall rating-ը շարունակական արժեք է (0-100)
- **80%-20% Train-Test Split**, stratified, որպեսզի բաշխումը հավասարակշռված լինի
- **Feature engineering**՝ scaling, encoding, missing values-ի լրացում (մեդիան կամ "Unknown")

---

## 🔍 Հաջորդ Քայլեր

- Ավելի լավ feature engineering (օր.՝ նոր հատկանիշներ՝ age group, BMI)
- Ավելի առաջադեմ մոդելներ (օր.՝ նեյրոնային ցանցեր, ensemble-ների կոմբինացիա)
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Դոմենային գիտելիքի ներգրավում՝ ֆուտբոլի մասնագետների հետ քննարկելով

---

## 🛠️ Պահանջվող Տեխնոլոգիաներ

**Python Libraries**
```python
pandas >= 1.3.0  
numpy >= 1.21.0  
scikit-learn >= 1.0.0  
matplotlib >= 3.4.0  
seaborn >= 0.11.0  
xgboost >= 1.5.0  
kagglehub >= 0.1.0  
```

**Միջավայր**
- Google Colab (առաջարկվում է)
- Jupyter Notebook
- Python 3.8+

---


## 🎯 Բիզնես Արժեք

- **Player Scouting** — Նոր տաղանդների հայտնաբերում
- **Team Management** — Թիմի օպտիմալ կազմավորում
- **Transfer Decisions** — Ավելի լավ գնային գնահատում
- **Player Development** — Զարգացման ճիշտ ուղիների որոշում

**ROI**  
- Ավելի ճշգրիտ խաղացողի գնահատում  
- Ռիսկի նվազեցում transfer market-ում  
- Տվյալամետ որոշումների կայացում

---




## 📊 Եզրակացություն

Այս նախագիծը ցույց է տալիս, որ **Machine Learning**-ը կարող է ապահովել բարձր ճշգրտությամբ կանխատեսումներ FIFA խաղացողների overall rating-ի համար:  
**XGBoost**-ը գերազանցեց մյուս մոդելներին 89.2% բացատրված variance-ով։

**Հիմնական գտածոներ**
- Տեխնիկական հմտությունները (passing, dribbling) ամենաշատն են ազդում գնահատականի վրա
- Potential-ը ամենամեծ նշանակությունն ունի
- Ensemble մոդելները զգալիորեն ավելի լավ են աշխատում
- Feature engineering-ը մեծ ազդեցություն ունի արդյունքների վրա

Նախագիծը կարող է ծառայել որպես հիմք իրական football analytics-ի և ավելի բարդ sports prediction համակարգերի համար։
