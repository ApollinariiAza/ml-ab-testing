#!/usr/bin/env python3
"""
Unit тесты для модуля A/B тестирования
"""

import unittest
import numpy as np
import pandas as pd
from main import ABTesting, TestType, TestResult

class TestABTesting(unittest.TestCase):
    """Тесты для класса ABTesting"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.ab_test = ABTesting(alpha=0.05)
        np.random.seed(42)  # Для воспроизводимости
    
    def test_initialization(self):
        """Тест инициализации"""
        self.assertEqual(self.ab_test.alpha, 0.05)
        self.assertEqual(self.ab_test.confidence_level, 0.95)
        
        # Тест с другим alpha
        ab_test_01 = ABTesting(alpha=0.01)
        self.assertEqual(ab_test_01.alpha, 0.01)
        self.assertEqual(ab_test_01.confidence_level, 0.99)
    
    def test_generate_sample_data_conversion(self):
        """Тест генерации данных для конверсии"""
        data = self.ab_test.generate_sample_data(
            n_control=100,
            n_treatment=150,
            test_type=TestType.CONVERSION,
            effect_size=0.05
        )
        
        # Проверяем структуру данных
        self.assertEqual(len(data), 250)
        self.assertEqual(list(data.columns), ['user_id', 'group', 'converted'])
        
        # Проверяем размеры групп
        group_counts = data['group'].value_counts()
        self.assertEqual(group_counts['A'], 100)
        self.assertEqual(group_counts['B'], 150)
        
        # Проверяем что конверсии являются 0 или 1
        self.assertTrue(data['converted'].isin([0, 1]).all())
    
    def test_generate_sample_data_continuous(self):
        """Тест генерации непрерывных данных"""
        data = self.ab_test.generate_sample_data(
            n_control=200,
            n_treatment=200,
            test_type=TestType.CONTINUOUS,
            effect_size=0.1
        )
        
        # Проверяем структуру
        self.assertEqual(len(data), 400)
        self.assertEqual(list(data.columns), ['user_id', 'group', 'revenue'])
        
        # Проверяем что выручка положительная (в основном)
        self.assertGreater(data['revenue'].mean(), 0)
    
    def test_conversion_test_significant(self):
        """Тест конверсии с значимым результатом"""
        # Создаём данные с известным большим эффектом
        data = self.ab_test.generate_sample_data(
            n_control=1000,
            n_treatment=1000,
            test_type=TestType.CONVERSION,
            effect_size=0.05  # Большой эффект
        )
        
        result = self.ab_test.conversion_test(data, 'converted')
        
        # Проверяем тип результата
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.test_type, TestType.CONVERSION)
        
        # Проверяем размеры выборок
        self.assertEqual(result.group_a_size, 1000)
        self.assertEqual(result.group_b_size, 1000)
        
        # Проверяем что метрики в разумных пределах
        self.assertTrue(0 <= result.group_a_metric <= 1)
        self.assertTrue(0 <= result.group_b_metric <= 1)
        
        # Проверяем статистические свойства
        self.assertTrue(0 <= result.p_value <= 1)
        self.assertTrue(0 <= result.power <= 1)
        
        # С большой выборкой и эффектом должно быть значимо
        self.assertLess(result.p_value, 0.1)  # Довольно либеральный порог
    
    def test_conversion_test_not_significant(self):
        """Тест конверсии без значимого результата"""
        # Создаём данные без эффекта
        data = self.ab_test.generate_sample_data(
            n_control=100,
            n_treatment=100,
            test_type=TestType.CONVERSION,
            effect_size=0.001  # Очень маленький эффект
        )
        
        result = self.ab_test.conversion_test(data, 'converted')
        
        # С маленькой выборкой и эффектом часто не значимо
        # Но это не гарантировано, поэтому просто проверяем структуру
        self.assertIsInstance(result.is_significant, bool)
        self.assertIsInstance(result.conclusion, str)
    
    def test_continuous_test(self):
        """Тест для непрерывных метрик"""
        data = self.ab_test.generate_sample_data(
            n_control=500,
            n_treatment=500,
            test_type=TestType.CONTINUOUS,
            effect_size=0.2
        )
        
        result = self.ab_test.continuous_test(data, 'revenue')
        
        # Проверяем тип результата
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.test_type, TestType.CONTINUOUS)
        
        # Проверяем размеры выборок
        self.assertEqual(result.group_a_size, 500)
        self.assertEqual(result.group_b_size, 500)
        
        # Проверяем что метрики положительные (для выручки)
        self.assertGreater(result.group_a_metric, 0)
        self.assertGreater(result.group_b_metric, 0)
        
        # Проверяем доверительный интервал
        ci_lower, ci_upper = result.confidence_interval
        self.assertLess(ci_lower, ci_upper)
    
    def test_sample_size_calculator(self):
        """Тест калькулятора размера выборки"""
        # Тест с типичными параметрами
        sample_size = self.ab_test.sample_size_calculator(
            baseline_rate=0.10,
            expected_lift=0.20,
            power=0.8,
            alpha=0.05
        )
        
        # Размер выборки должен быть положительным целым числом
        self.assertIsInstance(sample_size, int)
        self.assertGreater(sample_size, 0)
        self.assertLess(sample_size, 100000)  # Разумный верхний предел
        
        # Тест с более консервативными параметрами (меньший эффект)
        sample_size_small = self.ab_test.sample_size_calculator(
            baseline_rate=0.10,
            expected_lift=0.05,  # Меньший эффект
            power=0.8,
            alpha=0.05
        )
        
        # Для меньшего эффекта нужна большая выборка
        self.assertGreater(sample_size_small, sample_size)
    
    def test_confidence_intervals(self):
        """Тест доверительных интервалов"""
        data = self.ab_test.generate_sample_data(
            n_control=1000,
            n_treatment=1000,
            test_type=TestType.CONVERSION,
            effect_size=0.03
        )
        
        result = self.ab_test.conversion_test(data, 'converted')
        
        ci_lower, ci_upper = result.confidence_interval
        
        # Нижняя граница должна быть меньше верхней
        self.assertLess(ci_lower, ci_upper)
        
        # Реальная разность должна быть близко к интервалу
        actual_diff = result.group_b_metric - result.group_a_metric
        
        # Проверяем что интервал имеет разумную ширину
        interval_width = ci_upper - ci_lower
        self.assertGreater(interval_width, 0)
        self.assertLess(interval_width, 1)  # Не может быть шире 100%
    
    def test_edge_cases(self):
        """Тест граничных случаев"""
        # Тест с очень маленькими выборками
        small_data = pd.DataFrame({
            'user_id': [1, 2, 3, 4],
            'group': ['A', 'A', 'B', 'B'],
            'converted': [1, 0, 1, 1]
        })
        
        result = self.ab_test.conversion_test(small_data, 'converted')
        
        # Должно работать, но с широкими интервалами
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.group_a_size, 2)
        self.assertEqual(result.group_b_size, 2)
        
        # Тест с нулевой конверсией в одной группе
        zero_conv_data = pd.DataFrame({
            'user_id': range(100),
            'group': ['A'] * 50 + ['B'] * 50,
            'converted': [0] * 50 + [1] * 10 + [0] * 40
        })
        
        result_zero = self.ab_test.conversion_test(zero_conv_data, 'converted')
        self.assertEqual(result_zero.group_a_metric, 0.0)
        self.assertGreater(result_zero.group_b_metric, 0.0)
    
    def test_data_validation(self):
        """Тест валидации данных"""
        # Тест с пустыми данными
        empty_data = pd.DataFrame(columns=['user_id', 'group', 'converted'])
        
        with self.assertRaises(Exception):
            self.ab_test.conversion_test(empty_data, 'converted')
        
        # Тест с неправильными названиями групп
        wrong_groups = pd.DataFrame({
            'user_id': [1, 2, 3, 4],
            'group': ['X', 'X', 'Y', 'Y'],  # Не A и B
            'converted': [1, 0, 1, 0]
        })
        
        # Должно работать, но может дать неожиданные результаты
        # В реальной ситуации стоило бы добавить валидацию

class TestStatisticalProperties(unittest.TestCase):
    """Тесты статистических свойств"""
    
    def setUp(self):
        self.ab_test = ABTesting(alpha=0.05)
    
    def test_type_i_error_rate(self):
        """Тест ошибки I рода (ложные срабатывания)"""
        np.random.seed(123)
        false_positives = 0
        n_simulations = 100  # Уменьшено для быстроты тестов
        
        for _ in range(n_simulations):
            # Генерируем данные БЕЗ эффекта
            data = self.ab_test.generate_sample_data(
                n_control=500,
                n_treatment=500,
                test_type=TestType.CONVERSION,
                effect_size=0.0  # НЕТ эффекта
            )
            
            result = self.ab_test.conversion_test(data, 'converted')
            
            if result.is_significant:
                false_positives += 1
        
        false_positive_rate = false_positives / n_simulations
        
        # Ошибка I рода должна быть близка к alpha (5%)
        # Допускаем погрешность из-за случайности
        self.assertLess(false_positive_rate, 0.15)  # Не более 15%
        
        print(f"Наблюдаемая ошибка I рода: {false_positive_rate:.3f} (ожидается ~0.05)")
    
    def test_power_increases_with_sample_size(self):
        """Тест что мощность растёт с размером выборки"""
        effect_size = 0.03
        sample_sizes = [100, 500, 1000]
        powers = []
        
        for n in sample_sizes:
            data = self.ab_test.generate_sample_data(
                n_control=n,
                n_treatment=n,
                test_type=TestType.CONVERSION,
                effect_size=effect_size
            )
            
            result = self.ab_test.conversion_test(data, 'converted')
            powers.append(result.power)
        
        # Мощность должна расти с размером выборки
        self.assertLess(powers[0], powers[1])
        self.assertLess(powers[1], powers[2])
        
        print(f"Мощность для размеров выборки {sample_sizes}: {powers}")

def run_performance_test():
    """Тест производительности"""
    import time
    
    print("\nТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 30)
    
    ab_test = ABTesting()
    
    # Тест с большими данными
    start_time = time.time()
    
    large_data = ab_test.generate_sample_data(
        n_control=10000,
        n_treatment=10000,
        test_type=TestType.CONVERSION,
        effect_size=0.02
    )
    
    generation_time = time.time() - start_time
    
    start_time = time.time()
    result = ab_test.conversion_test(large_data, 'converted')
    test_time = time.time() - start_time
    
    print(f"Генерация 20,000 записей: {generation_time:.3f} сек")
    print(f"Проведение теста: {test_time:.3f} сек")
    print(f"Результат: p-value = {result.p_value:.6f}")
    
    # Проверяем что время разумное
    assert generation_time < 5.0, "Генерация данных слишком медленная"
    assert test_time < 1.0, "Тест слишком медленный"

if __name__ == '__main__':
    # Запускаем основные тесты
    print("ЗАПУСК UNIT ТЕСТОВ")
    print("=" * 30)
    
    # Создаём test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавляем тесты
    suite.addTests(loader.loadTestsFromTestCase(TestABTesting))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalProperties))
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Тест производительности
    run_performance_test()
    
    # Итоговый отчёт
    print("\nИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 30)
    print(f"Тестов запущено: {result.testsRun}")
    print(f"Ошибки: {len(result.errors)}")
    print(f"Неудачи: {len(result.failures)}")
    
    if result.wasSuccessful():
        print("Все тесты прошли успешно!")
    else:
        print("Некоторые тесты не прошли!")
        
        if result.errors:
            print("\nОШИБКИ:")
            for test, error in result.errors:
                print(f"  {test}: {error}")
        
        if result.failures:
            print("\nНЕУДАЧИ:")
            for test, failure in result.failures:
                print(f"  {test}: {failure}")