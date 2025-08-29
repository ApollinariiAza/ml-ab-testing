# ML A/B Testing Framework

Комплексный фреймворк для проведения A/B тестирования с расчётом статистики, доверительных интервалов и анализом мощности тестов.

## Возможности

- **Тесты конверсии** - биномиальные метрики (CTR, конверсия покупки)
- **Тесты непрерывных метрик** - средние значения (выручка, время на сайте)
- **Расчёт размера выборки** - планирование экспериментов
- **Анализ мощности** - оценка вероятности обнаружения эффектов
- **Визуализация результатов** - информативные графики и отчёты
- **Генерация тестовых данных** - для обучения и демонстрации

## 🛠 Технологии

- **Python 3.9+**
- **NumPy** - математические вычисления
- **Pandas** - работа с данными  
- **SciPy** - статистические тесты
- **Matplotlib + Seaborn** - визуализация
- **Jupyter** - интерактивные notebook'ы

## Установка

### Быстрый старт

```bash
# Клонировать репозиторий
git clone <repository-url>
cd ml-ab-testing

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установить зависимости
pip install -r requirements.txt
```

### Альтернативно через conda

```bash
conda create -n ab_testing python=3.9
conda activate ab_testing
pip install -r requirements.txt
```

## Быстрый пример

```python
from ab_testing import ABTesting, TestType

# Инициализация
ab_test = ABTesting(alpha=0.05)

# Генерация тестовых данных
data = ab_test.generate_sample_data(
    n_control=1500,      # Размер контрольной группы
    n_treatment=1500,    # Размер тестовой группы
    test_type=TestType.CONVERSION,
    effect_size=0.02     # Ожидаемый прирост конверсии на 2%
)

# Проведение теста
result = ab_test.conversion_test(data, 'converted')

# Результаты
print(ab_test.generate_report(result))
ab_test.plot_results(data, result)
```

**Результат:**
```
╔═══════════════════════════════════════════╗
║                A/B TEST REPORT            ║
║ Группа A: 0.1013 | Группа B: 0.1213      ║
║ p-value: 0.0034 | Эффект: +19.7%         ║
║ ВЫВОД: Статистически значимое улучшение ║
╚═══════════════════════════════════════════╝
```

## Подробные примеры

### 1. Тест конверсии кнопки покупки

```python
# Сравниваем старую и новую кнопку "Купить"
conversion_data = ab_test.generate_sample_data(
    n_control=2000,    # Старая кнопка
    n_treatment=2000,  # Новая кнопка
    test_type=TestType.CONVERSION,
    effect_size=0.015  # Ожидаем +1.5% к конверсии
)

result = ab_test.conversion_test(conversion_data)

# Детальный анализ
print(f"Конверсия A: {result.group_a_metric:.3f}")
print(f"Конверсия B: {result.group_b_metric:.3f}")
print(f"Относительный прирост: {((result.group_b_metric/result.group_a_metric - 1)*100):+.1f}%")
print(f"p-value: {result.p_value:.6f}")
print(f"Доверительный интервал: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
```

### 2. Тест средней выручки на пользователя

```python
# Сравниваем алгоритмы рекомендаций
revenue_data = ab_test.generate_sample_data(
    n_control=1000,
    n_treatment=1000,
    test_type=TestType.CONTINUOUS,
    effect_size=0.12   # Ожидаем +12% к выручке
)

result = ab_test.continuous_test(revenue_data, 'revenue')

print(f"Средняя выручка A: {result.group_a_metric:.2f} ₽")
print(f"Средняя выручка B: {result.group_b_metric:.2f} ₽")
print(f"Абсолютная разность: {result.group_b_metric - result.group_a_metric:+.2f} ₽")
```

### 3. Планирование размера выборки

```python
# Рассчитываем необходимый размер выборки
sample_size = ab_test.sample_size_calculator(
    baseline_rate=0.08,      # Текущая конверсия 8%
    expected_lift=0.15,      # Хотим обнаружить прирост в 15%
    power=0.8,               # Мощность 80%
    alpha=0.05               # Значимость 5%
)

print(f"Необходимый размер выборки: {sample_size} пользователей в каждой группе")
print(f"Общий размер эксперимента: {sample_size * 2} пользователей")

# При 1000 пользователей в день
days_needed = (sample_size * 2) / 1000
print(f"Время проведения теста: {days_needed:.1f} дней")
```

## Типы анализа

### Тест конверсии (биномиальные метрики)

Используется для метрик типа "да/нет":
- Конверсия покупки
- Клик по кнопке (CTR)
- Регистрация
- Подписка на рассылку

**Статистический метод:** Z-тест для пропорций

### Тест непрерывных метрик

Используется для количественных метрик:
- Выручка на пользователя
- Время на сайте  
- Количество просмотров страниц
- Средний чек

**Статистический метод:** t-тест Уэлча

## Интерпретация результатов

### P-value
- **< 0.05** - статистически значимое различие
- **≥ 0.05** - нет статистически значимых различий

### Размер эффекта (Effect Size)
- **Малый:** < 0.2
- **Средний:** 0.2 - 0.8  
- **Большой:** > 0.8

### Мощность теста (Power)
- **< 0.8** - низкая мощность, высок риск пропустить реальный эффект
- **≥ 0.8** - достаточная мощность для надёжного обнаружения эффектов

### Доверительный интервал
- Если **не содержит 0** - эффект статистически значим
- **Ширина интервала** показывает точность оценки

## Визуализация

Фреймворк создаёт комплексные графики:

1. **Сравнение метрик** между группами
2. **Доверительные интервалы** для разности
3. **Распределения** данных (для непрерывных метрик)
4. **Статистические метрики** (p-value, effect size, power)

```python
# Создание и сохранение графиков
ab_test.plot_results(data, result, save_path='experiment_results.png')
```

## Примеры использования

### Запуск базовых примеров

```bash
python examples.py
```

Выполнит:
- Тест конверсии кнопки покупки
- Тест средней выручки  
- Анализ сегментированных данных
- Расчёт размера выборки
- Анализ мощности теста

### Интерактивный Jupyter Notebook

```bash
jupyter notebook ab_testing_demo.ipynb
```

Содержит пошаговые примеры с объяснениями и визуализацией.

## Тестирование

Запуск unit тестов:

```bash
python test_ab_testing.py
```

Тесты покрывают:
- Корректность статистических вычислений
- Валидацию входных данных
- Граничные случаи
- Производительность на больших данных
- Контроль ошибки I рода (ложные срабатывания)

## Архитектура проекта

```
ml-ab-testing/
├── ab_testing.py           # Основной модуль
├── examples.py             # Примеры использования  
├── test_ab_testing.py      # Unit тесты
├── ab_testing_demo.ipynb   # Jupyter notebook
├── requirements.txt        # Зависимости
└── README.md              # Документация
```

### Основные классы

- **`ABTesting`** - главный класс для проведения тестов
- **`TestType`** - перечисление типов тестов  
- **`TestResult`** - структура результатов теста

### Ключевые методы

- **`conversion_test()`** - тест биномиальных метрик
- **`continuous_test()`** - тест непрерывных метрик
- **`sample_size_calculator()`** - расчёт размера выборки
- **`plot_results()`** - визуализация результатов
- **`generate_report()`** - текстовый отчёт

## Теоретические основы

### Статистические тесты

**Z-тест для пропорций:**
```
z = (p₁ - p₂) / √(p̂(1-p̂)(1/n₁ + 1/n₂))
где p̂ = (x₁ + x₂)/(n₁ + n₂)
```

**t-тест Уэлча:**
```  
t = (μ₁ - μ₂) / √(σ₁²/n₁ + σ₂²/n₂)
df = (σ₁²/n₁ + σ₂²/n₂)² / ((σ₁²/n₁)²/(n₁-1) + (σ₂²/n₂)²/(n₂-1))
```

### Размер эффекта

**Cohen's h (для пропорций):**
```
h = 2(arcsin(√p₁) - arcsin(√p₂))
```

**Cohen's d (для средних):**
```
d = (μ₁ - μ₂) / σₚₒₒₗₑ𝒹
```

### Расчёт мощности

Учитывает:
- Размер эффекта
- Размер выборки  
- Уровень значимости α
- Вариабельность данных

## Важные предупреждения

### Множественное сравнение
При проведении нескольких тестов одновременно применяйте поправки:
- **Поправка Бонферрони:** α' = α/k
- **FDR контроль** для большого числа тестов

### Peeking Problem  
**Не подглядывайте в результаты** во время теста - это увеличивает ошибку I рода.

### Предположения тестов
- **Независимость** наблюдений
- **Случайное распределение** по группам
- **Стабильность** условий эксперимента

### Практическая vs статистическая значимость
Малые эффекты могут быть статистически значимы при больших выборках, но не иметь практической ценности.

## Реальные примеры

### Пример 1: E-commerce конверсия

```python
"""
Задача: Тестируем новый дизайн страницы товара
Метрика: Конверсия в покупку
Гипотеза: Новый дизайн увеличит конверсию на 10%
"""

# Планирование
current_conversion = 0.12  # Текущая конверсия 12%
expected_lift = 0.10       # Ожидаемый прирост 10%

sample_size = ab_test.sample_size_calculator(
    baseline_rate=current_conversion,
    expected_lift=expected_lift,
    power=0.8
)
print(f"Нужно {sample_size} пользователей в каждой группе")

# Результат: Группа A vs Группа B → вывод о значимости
```

### Пример 2: Мобильное приложение

```python
"""
Задача: Новый алгоритм пуш-уведомлений  
Метрика: Средняя сессионное время
Гипотеза: Персонализированные уведомления увеличат время в приложении
"""

session_data = pd.DataFrame({
    'user_id': range(2000),
    'group': ['A'] * 1000 + ['B'] * 1000,
    'session_time': np.concatenate([
        np.random.lognormal(3.5, 0.8, 1000),  # Группа A
        np.random.lognormal(3.65, 0.8, 1000)  # Группа B (+15%)
    ])
})

result = ab_test.continuous_test(session_data, 'session_time')
print(f"Среднее время A: {result.group_a_metric:.1f} мин")
print(f"Среднее время B: {result.group_b_metric:.1f} мин") 
print(f"Прирост: {((result.group_b_metric/result.group_a_metric-1)*100):+.1f}%")
```

### Пример 3: Email маркетинг

```python
"""
Задача: A/B тест темы письма
Метрика: Open Rate (коэффициент открытий)
"""

email_test = {
    'subject_a': 'Скидка 20% на всё!',           # Стандартная тема  
    'subject_b': 'Эксклюзивное предложение для вас' # Персонализированная
}

# Симуляция результатов
email_data = ab_test.generate_sample_data(
    n_control=5000,   # Разослали 5000 писем с темой A
    n_treatment=5000, # Разослали 5000 писем с темой B  
    test_type=TestType.CONVERSION,
    effect_size=0.08  # Ожидаем +8% к открываемости
)

result = ab_test.conversion_test(email_data, 'converted')
ab_test.plot_results(email_data, result, save_path='email_ab_test.png')
```

## Продвинутые техники

### Сегментированный анализ

```python
def segment_analysis(data, segments):
    """Анализ результатов по сегментам пользователей"""
    results = {}
    
    for segment_name, segment_data in segments.items():
        filtered_data = data[data['user_id'].isin(segment_data)]
        result = ab_test.conversion_test(filtered_data)
        results[segment_name] = result
        
        print(f"\n{segment_name}:")
        print(f"  Размер: {len(filtered_data)}")
        print(f"  p-value: {result.p_value:.4f}")
        print(f"  Эффект: {((result.group_b_metric/result.group_a_metric-1)*100):+.1f}%")
    
    return results

# Пример использования
segments = {
    'new_users': new_user_ids,
    'returning_users': returning_user_ids,
    'mobile_users': mobile_user_ids,
    'desktop_users': desktop_user_ids
}

segment_results = segment_analysis(data, segments)
```

### Байесовский A/B тест

```python
def bayesian_ab_test(conversions_a, trials_a, conversions_b, trials_b):
    """Байесовский подход к A/B тестированию"""
    from scipy.stats import beta
    
    # Априорные распределения (неинформативные)
    alpha_prior, beta_prior = 1, 1
    
    # Апостериорные распределения
    alpha_a = alpha_prior + conversions_a
    beta_a = beta_prior + trials_a - conversions_a
    
    alpha_b = alpha_prior + conversions_b  
    beta_b = beta_prior + trials_b - conversions_b
    
    # Вероятность что B > A
    samples_a = beta.rvs(alpha_a, beta_a, size=100000)
    samples_b = beta.rvs(alpha_b, beta_b, size=100000)
    
    prob_b_better = np.mean(samples_b > samples_a)
    
    return prob_b_better

# Пример
prob = bayesian_ab_test(120, 1000, 140, 1000)
print(f"Вероятность что B лучше A: {prob:.3f}")
```

## 🔧 Практические рекомендации

### Планирование эксперимента

1. **Определите основную метрику** заранее
2. **Рассчитайте размер выборки** для желаемого эффекта
3. **Установите критерий успеха** (MDE - минимальный детектируемый эффект)
4. **Запланируйте длительность** теста

### Во время эксперимента

1. **Не подглядывайте** в промежуточные результаты
2. **Следите за стабильностью** внешних условий
3. **Контролируйте качество** трафика
4. **Документируйте** все изменения

### Анализ результатов

1. **Проверьте предположения** тестов
2. **Рассчитайте практическую значимость**
3. **Проанализируйте сегменты** пользователей
4. **Учтите множественные сравнения**

## Частые ошибки

### 1. Недостаточный размер выборки
```python
# НЕПРАВИЛЬНО
small_test = ab_test.generate_sample_data(n_control=50, n_treatment=50)
result = ab_test.conversion_test(small_test)
print(f"Мощность: {result.power:.3f}")  # Скорее всего < 0.8

# ПРАВИЛЬНО  
needed_size = ab_test.sample_size_calculator(0.10, 0.20, power=0.8)
large_test = ab_test.generate_sample_data(n_control=needed_size, n_treatment=needed_size)
```

### 2. Игнорирование практической значимости
```python
# Фокус только на p-value
if result.p_value < 0.05:
    print("Статистически значимо!")

# Анализ практической ценности  
effect_percent = (result.group_b_metric / result.group_a_metric - 1) * 100
if result.p_value < 0.05 and abs(effect_percent) > 5:  # Минимум 5% изменения
    print(f"Практически значимое изменение: {effect_percent:+.1f}%")
```

### 3. Множественные тестирования без корректировок
```python
# Много тестов без корректировок
segments = ['mobile', 'desktop', 'ios', 'android', 'new', 'returning']
for segment in segments:
    # p-hacking риск!

# С поправкой Бонферрони
alpha_corrected = 0.05 / len(segments)
ab_test_corrected = ABTesting(alpha=alpha_corrected)
```

## Метрики для разных областей

### E-commerce
- **Конверсия покупки** - основная метрика
- **Средний чек** - непрерывная метрика
- **Revenue per User** - непрерывная метрика
- **Cart abandonment rate** - конверсионная метрика

### SaaS продукты  
- **Trial to Paid conversion** - конверсионная
- **Monthly Active Users** - подсчёт
- **Feature adoption rate** - конверсионная
- **Churn rate** - конверсионная

### Контент/Медиа
- **Click-through rate (CTR)** - конверсионная
- **Time on page** - непрерывная
- **Bounce rate** - конверсионная  
- **Pages per session** - непрерывная

## Расширенные возможности

Фреймворк можно расширить для:

### Multi-armed Bandit
```python
# Будущая функциональность
bandit = MultiArmedBandit(arms=['A', 'B', 'C'])
bandit.update(arm='B', reward=1)
best_arm = bandit.select_arm()
```

### Sequential Testing
```python  
# Последовательное тестирование
sequential_test = SequentialABTest(alpha=0.05, beta=0.2)
for day in range(14):
    sequential_test.add_data(daily_data[day])
    if sequential_test.should_stop():
        break
```

### Stratified Randomization
```python
# Стратифицированная рандомизация  
stratified_data = stratify_users(users, strata=['country', 'device_type'])
```

## Поддержка и развитие

### Сообщество
- **Issues** - сообщения об ошибках
- **Feature requests** - предложения новых функций
- **Documentation** - улучшение документации

### Roadmap
- [ ] Поддержка категориальных метрик
- [ ] Байесовские методы  
- [ ] Multi-armed bandit алгоритмы
- [ ] Веб-интерфейс для экспериментов
- [ ] Интеграция с системами аналитики

### Вклад в проект
1. Fork репозитория
2. Создайте feature branch  
3. Добавьте тесты для нового функционала
4. Убедитесь что все тесты проходят
5. Создайте Pull Request с описанием изменений

## Лицензия

MIT License - используйте свободно в коммерческих и личных проектах.

## Благодарности

Основано на классических методах статистики и лучших практиках индустрии:
- Fisher, R.A. - основы статистического вывода
- Student (W.S. Gosset) - t-тест  
- Welch, B.L. - модификация t-теста
- Cohen, J. - размеры эффекта

---

**🎯 Успешных экспериментов и data-driven решений!**
