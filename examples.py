#!/usr/bin/env python3
"""
Примеры использования модуля A/B тестирования
"""

from main import ABTesting, TestType
import pandas as pd
import numpy as np

def example_conversion_test():
    """Пример теста конверсии"""
    print("ПРИМЕР 1: Тест конверсии (кнопка покупки)")
    print("=" * 60)
    
    # Инициализация
    ab_test = ABTesting(alpha=0.05)
    
    # Генерируем данные: старая кнопка vs новая кнопка
    data = ab_test.generate_sample_data(
        n_control=2000,      # Группа A - старая кнопка
        n_treatment=2000,    # Группа B - новая кнопка  
        test_type=TestType.CONVERSION,
        effect_size=0.02     # Ожидаемый прирост конверсии на 2%
    )
    
    print(f"Данные сгенерированы:")
    print(f"• Размер выборки: {len(data)} пользователей")
    print(f"• Группа A: {len(data[data['group'] == 'A'])} пользователей")  
    print(f"• Группа B: {len(data[data['group'] == 'B'])} пользователей")
    print()
    
    # Проводим тест
    result = ab_test.conversion_test(data, 'converted')
    
    # Выводим отчёт
    print(ab_test.generate_report(result))
    
    # Строим графики
    ab_test.plot_results(data, result, save_path='conversion_test.png')
    
    return data, result

def example_revenue_test():
    """Пример теста средней выручки"""
    print("\nПРИМЕР 2: Тест средней выручки на пользователя")
    print("=" * 60)
    
    ab_test = ABTesting(alpha=0.05)
    
    # Генерируем данные: старый алгоритм рекомендаций vs новый
    data = ab_test.generate_sample_data(
        n_control=1500,
        n_treatment=1500, 
        test_type=TestType.CONTINUOUS,
        effect_size=0.15    # Ожидаемый прирост выручки
    )
    
    print(f"Данные сгенерированы:")
    print(f"• Размер выборки: {len(data)} пользователей")
    print(f"• Средняя выручка группы A: {data[data['group'] == 'A']['revenue'].mean():.2f} ₽")
    print(f"• Средняя выручка группы B: {data[data['group'] == 'B']['revenue'].mean():.2f} ₽")
    print()
    
    # Проводим тест
    result = ab_test.continuous_test(data, 'revenue')
    
    # Выводим отчёт
    print(ab_test.generate_report(result))
    
    # Строим графики
    ab_test.plot_results(data, result, metric_col='revenue', save_path='revenue_test.png')
    
    return data, result

def example_real_world_data():
    """Пример с реальными данными"""
    print("\nПРИМЕР 3: Анализ реальных данных")
    print("=" * 60)
    
    # Симулируем реальные данные интернет-магазина
    np.random.seed(123)
    
    # Группа A - существующий дизайн страницы оплаты
    # Группа B - упрощённый дизайн страницы оплаты
    
    # Создаём более реалистичные данные с сегментацией
    data = []
    
    # Различные сегменты пользователей
    segments = ['mobile', 'desktop', 'new_users', 'returning_users']
    base_conversions = {'mobile': 0.08, 'desktop': 0.12, 'new_users': 0.06, 'returning_users': 0.15}
    
    user_id = 0
    for segment in segments:
        for group in ['A', 'B']:
            n_users = np.random.randint(800, 1200)  # Случайные размеры групп
            
            base_conv = base_conversions[segment]
            # Группа B показывает улучшение на 15-25% в зависимости от сегмента
            if group == 'B':
                improvement = np.random.uniform(0.15, 0.25)
                conversion_rate = base_conv * (1 + improvement)
            else:
                conversion_rate = base_conv
            
            # Генерируем конверсии
            conversions = np.random.binomial(1, conversion_rate, n_users)
            
            # Генерируем выручку (только для конвертированных)
            revenues = np.zeros(n_users)
            converted_indices = np.where(conversions == 1)[0]
            
            # Средний чек зависит от сегмента
            avg_revenue = {'mobile': 850, 'desktop': 1200, 'new_users': 600, 'returning_users': 1400}[segment]
            revenues[converted_indices] = np.random.lognormal(
                np.log(avg_revenue), 0.5, len(converted_indices)
            )
            
            for i in range(n_users):
                data.append({
                    'user_id': user_id,
                    'group': group,
                    'segment': segment,
                    'converted': conversions[i],
                    'revenue': revenues[i]
                })
                user_id += 1
    
    df = pd.DataFrame(data)
    
    print(f"Реалистичные данные созданы:")
    print(f"• Общий размер: {len(df)} пользователей")
    print(f"• Сегменты: {', '.join(segments)}")
    print(f"• Группы: A (контроль), B (тест)")
    print()
    
    # Общий анализ конверсии
    ab_test = ABTesting(alpha=0.05)
    
    print("ОБЩИЙ АНАЛИЗ КОНВЕРСИИ:")
    result_conversion = ab_test.conversion_test(df, 'converted')
    print(ab_test.generate_report(result_conversion))
    
    # Анализ по сегментам
    print("\nАНАЛИЗ ПО СЕГМЕНТАМ:")
    print("-" * 40)
    
    segment_results = {}
    for segment in segments:
        segment_data = df[df['segment'] == segment]
        result = ab_test.conversion_test(segment_data, 'converted')
        segment_results[segment] = result
        
        print(f"\n{segment.upper()}:")
        print(f"  • Размер выборки: {len(segment_data)}")
        print(f"  • Конверсия A: {result.group_a_metric:.3f}")
        print(f"  • Конверсия B: {result.group_b_metric:.3f}")
        print(f"  • Прирост: {((result.group_b_metric/result.group_a_metric - 1)*100):+.1f}%")
        print(f"  • p-value: {result.p_value:.4f}")
        print(f"  • Значимо: {'Да' if result.is_significant else 'Нет'}")
    
    # Анализ выручки среди конвертированных
    converted_users = df[df['converted'] == 1]
    if len(converted_users) > 0:
        print(f"\nАНАЛИЗ ВЫРУЧКИ СРЕДИ КОНВЕРТИРОВАННЫХ ({len(converted_users)} пользователей):")
        result_revenue = ab_test.continuous_test(converted_users, 'revenue')
        print(ab_test.generate_report(result_revenue))
    
    # Строим графики для общего анализа
    ab_test.plot_results(df, result_conversion, save_path='real_world_conversion.png')
    
    return df, result_conversion, segment_results

def example_sample_size_calculation():
    """Пример расчёта размера выборки"""
    print("\nПРИМЕР 4: Расчёт размера выборки")
    print("=" * 60)
    
    ab_test = ABTesting()
    
    scenarios = [
        {"baseline": 0.10, "lift": 0.1, "desc": "Базовая конверсия 10%, ожидаемый прирост 10%"},
        {"baseline": 0.05, "lift": 0.2, "desc": "Низкая конверсия 5%, ожидаемый прирост 20%"},
        {"baseline": 0.20, "lift": 0.05, "desc": "Высокая конверсия 20%, небольшой прирост 5%"},
    ]
    
    print("Расчёт необходимого размера выборки для различных сценариев:\n")
    
    for i, scenario in enumerate(scenarios, 1):
        sample_size = ab_test.sample_size_calculator(
            baseline_rate=scenario["baseline"],
            expected_lift=scenario["lift"],
            power=0.8,
            alpha=0.05
        )
        
        print(f"Сценарий {i}: {scenario['desc']}")
        print(f"  • Необходимый размер выборки: {sample_size:,} пользователей в каждой группе")
        print(f"  • Общий размер эксперимента: {sample_size * 2:,} пользователей")
        
        # Оценка времени проведения теста
        daily_traffic = 1000  # Предполагаемый дневной трафик
        days_needed = (sample_size * 2) / daily_traffic
        print(f"  • Время проведения теста (при {daily_traffic} пользователей/день): {days_needed:.1f} дней")
        print()

def example_power_analysis():
    """Пример анализа мощности теста"""
    print("\nПРИМЕР 5: Анализ мощности теста")
    print("=" * 60)
    
    ab_test = ABTesting()
    
    # Создаём данные с известным эффектом
    data = ab_test.generate_sample_data(
        n_control=500,   # Небольшая выборка
        n_treatment=500,
        test_type=TestType.CONVERSION,
        effect_size=0.03  # Небольшой эффект
    )
    
    result = ab_test.conversion_test(data)
    
    print("Анализ мощности для небольшой выборки:")
    print(f"• Размер выборки: {result.group_a_size + result.group_b_size}")
    print(f"• Реальная разность: {result.group_b_metric - result.group_a_metric:.4f}")
    print(f"• p-value: {result.p_value:.4f}")
    print(f"• Мощность теста: {result.power:.3f}")
    print(f"• Статистически значимо: {'Да' if result.is_significant else 'Нет'}")
    
    if result.power < 0.8:
        print(f"\nПРЕДУПРЕЖДЕНИЕ: Мощность теста ({result.power:.3f}) ниже рекомендуемой (0.8)")
        print("   Это означает высокий риск не обнаружить реальный эффект (ошибка II рода)")
        
        # Рассчитаем необходимый размер выборки
        needed_size = ab_test.sample_size_calculator(
            baseline_rate=result.group_a_metric,
            expected_lift=(result.group_b_metric - result.group_a_metric) / result.group_a_metric,
            power=0.8
        )
        print(f"   Рекомендуемый размер выборки: {needed_size} в каждой группе")

def main():
    """Запуск всех примеров"""
    print("ПРИМЕРЫ A/B ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    try:
        # Пример 1: Тест конверсии
        data1, result1 = example_conversion_test()
        
        # Пример 2: Тест выручки
        data2, result2 = example_revenue_test()
        
        # Пример 3: Реальные данные
        data3, result3, segments = example_real_world_data()
        
        # Пример 4: Расчёт размера выборки
        example_sample_size_calculation()
        
        # Пример 5: Анализ мощности
        example_power_analysis()
        
        print("\nВсе примеры выполнены успешно!")
        print("Графики сохранены как PNG файлы")
        
    except Exception as e:
        print(f"Ошибка при выполнении примеров: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()