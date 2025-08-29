import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, chi2_contingency, ttest_ind
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TestType(Enum):
    """Типы A/B тестов"""
    CONVERSION = "conversion"  # Конверсия (биномиальная метрика)
    CONTINUOUS = "continuous"  # Непрерывная метрика (среднее значение)
    COUNT = "count"           # Количественная метрика

@dataclass
class TestResult:
    """Результаты A/B теста"""
    test_type: TestType
    group_a_size: int
    group_b_size: int
    group_a_metric: float
    group_b_metric: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    is_significant: bool
    conclusion: str

class ABTesting:
    """Класс для проведения A/B тестирования"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Инициализация A/B теста
        
        Args:
            alpha: Уровень значимости (по умолчанию 0.05)
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha
        
    def generate_sample_data(self, 
                           n_control: int = 1000, 
                           n_treatment: int = 1000,
                           test_type: TestType = TestType.CONVERSION,
                           effect_size: float = 0.1) -> pd.DataFrame:
        """
        Генерирует синтетические данные для A/B теста
        
        Args:
            n_control: Размер контрольной группы
            n_treatment: Размер тестовой группы  
            test_type: Тип теста
            effect_size: Размер эффекта
            
        Returns:
            DataFrame с данными эксперимента
        """
        np.random.seed(42)
        
        if test_type == TestType.CONVERSION:
            # Биномиальные данные (конверсия)
            p_control = 0.10  # Базовая конверсия 10%
            p_treatment = p_control + effect_size
            
            control_conversions = np.random.binomial(1, p_control, n_control)
            treatment_conversions = np.random.binomial(1, p_treatment, n_treatment)
            
            data = pd.DataFrame({
                'user_id': range(n_control + n_treatment),
                'group': ['A'] * n_control + ['B'] * n_treatment,
                'converted': np.concatenate([control_conversions, treatment_conversions])
            })
            
        elif test_type == TestType.CONTINUOUS:
            # Непрерывные данные (например, выручка)
            mean_control = 100.0
            mean_treatment = mean_control + effect_size * 10  # Эффект в рублях
            std = 25.0
            
            control_values = np.random.normal(mean_control, std, n_control)
            treatment_values = np.random.normal(mean_treatment, std, n_treatment)
            
            data = pd.DataFrame({
                'user_id': range(n_control + n_treatment),
                'group': ['A'] * n_control + ['B'] * n_treatment,
                'revenue': np.concatenate([control_values, treatment_values])
            })
            
        elif test_type == TestType.COUNT:
            # Данные подсчёта (например, количество заказов)
            lambda_control = 2.0
            lambda_treatment = lambda_control + effect_size
            
            control_counts = np.random.poisson(lambda_control, n_control)
            treatment_counts = np.random.poisson(lambda_treatment, n_treatment)
            
            data = pd.DataFrame({
                'user_id': range(n_control + n_treatment),
                'group': ['A'] * n_control + ['B'] * n_treatment,
                'orders_count': np.concatenate([control_counts, treatment_counts])
            })
        
        return data
    
    def conversion_test(self, data: pd.DataFrame, 
                       conversion_col: str = 'converted') -> TestResult:
        """
        Проводит тест на конверсию (биномиальная метрика)
        
        Args:
            data: DataFrame с данными
            conversion_col: Название колонки с конверсиями
            
        Returns:
            Результаты теста
        """
        group_a = data[data['group'] == 'A'][conversion_col]
        group_b = data[data['group'] == 'B'][conversion_col]
        
        # Основные метрики
        n_a, n_b = len(group_a), len(group_b)
        conversions_a = group_a.sum()
        conversions_b = group_b.sum()
        rate_a = conversions_a / n_a
        rate_b = conversions_b / n_b
        
        # Z-тест для пропорций
        pooled_rate = (conversions_a + conversions_b) / (n_a + n_b)
        pooled_se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/n_a + 1/n_b))
        z_score = (rate_b - rate_a) / pooled_se
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        # Доверительный интервал для разности пропорций
        se_diff = np.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)
        z_critical = norm.ppf(1 - self.alpha / 2)
        diff = rate_b - rate_a
        ci_lower = diff - z_critical * se_diff
        ci_upper = diff + z_critical * se_diff
        
        # Размер эффекта (Cohen's h)
        effect_size = 2 * (np.arcsin(np.sqrt(rate_b)) - np.arcsin(np.sqrt(rate_a)))
        
        # Мощность теста (приближённо)
        power = self._calculate_power_proportion(rate_a, rate_b, n_a, n_b)
        
        is_significant = p_value < self.alpha
        
        # Вывод
        if is_significant:
            direction = "выше" if rate_b > rate_a else "ниже"
            conclusion = f"Конверсия группы B статистически значимо {direction} группы A"
        else:
            conclusion = "Нет статистически значимых различий между группами"
        
        return TestResult(
            test_type=TestType.CONVERSION,
            group_a_size=n_a,
            group_b_size=n_b,
            group_a_metric=rate_a,
            group_b_metric=rate_b,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            power=power,
            is_significant=is_significant,
            conclusion=conclusion
        )
    
    def continuous_test(self, data: pd.DataFrame, 
                       metric_col: str = 'revenue') -> TestResult:
        """
        Проводит тест для непрерывной метрики (t-тест)
        
        Args:
            data: DataFrame с данными
            metric_col: Название колонки с метрикой
            
        Returns:
            Результаты теста
        """
        group_a = data[data['group'] == 'A'][metric_col]
        group_b = data[data['group'] == 'B'][metric_col]
        
        # Основные метрики
        n_a, n_b = len(group_a), len(group_b)
        mean_a, mean_b = group_a.mean(), group_b.mean()
        std_a, std_b = group_a.std(ddof=1), group_b.std(ddof=1)
        
        # t-тест Уэлча (не предполагает равенство дисперсий)
        t_stat, p_value = ttest_ind(group_b, group_a, equal_var=False)
        
        # Доверительный интервал для разности средних
        se_diff = np.sqrt(std_a**2 / n_a + std_b**2 / n_b)
        df = (std_a**2 / n_a + std_b**2 / n_b)**2 / (
            (std_a**2 / n_a)**2 / (n_a - 1) + (std_b**2 / n_b)**2 / (n_b - 1)
        )
        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        diff = mean_b - mean_a
        ci_lower = diff - t_critical * se_diff
        ci_upper = diff + t_critical * se_diff
        
        # Размер эффекта (Cohen's d)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        effect_size = diff / pooled_std
        
        # Мощность теста
        power = self._calculate_power_ttest(mean_a, mean_b, std_a, std_b, n_a, n_b)
        
        is_significant = p_value < self.alpha
        
        # Вывод
        if is_significant:
            direction = "выше" if mean_b > mean_a else "ниже"
            conclusion = f"Среднее значение группы B статистически значимо {direction} группы A"
        else:
            conclusion = "Нет статистически значимых различий между группами"
        
        return TestResult(
            test_type=TestType.CONTINUOUS,
            group_a_size=n_a,
            group_b_size=n_b,
            group_a_metric=mean_a,
            group_b_metric=mean_b,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            power=power,
            is_significant=is_significant,
            conclusion=conclusion
        )
    
    def _calculate_power_proportion(self, p1: float, p2: float, 
                                  n1: int, n2: int) -> float:
        """Рассчитывает мощность теста для пропорций"""
        pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se_null = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
        se_alt = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        
        z_alpha = norm.ppf(1 - self.alpha / 2)
        z_beta = (abs(p2 - p1) - z_alpha * se_null) / se_alt
        power = norm.cdf(z_beta)
        return max(0, min(1, power))
    
    def _calculate_power_ttest(self, m1: float, m2: float, s1: float, s2: float,
                              n1: int, n2: int) -> float:
        """Рассчитывает мощность t-теста"""
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        effect_size = abs(m2 - m1) / pooled_std
        se = pooled_std * np.sqrt(1/n1 + 1/n2)
        
        df = n1 + n2 - 2
        t_alpha = stats.t.ppf(1 - self.alpha / 2, df)
        t_beta = (abs(m2 - m1) - t_alpha * se) / se
        power = 1 - stats.t.cdf(t_beta, df)
        return max(0, min(1, power))
    
    def plot_results(self, data: pd.DataFrame, result: TestResult, 
                    metric_col: str = None, save_path: str = None):
        """
        Создаёт визуализацию результатов A/B теста
        
        Args:
            data: DataFrame с данными
            result: Результаты теста
            metric_col: Название колонки с метрикой
            save_path: Путь для сохранения графика
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Результаты A/B теста', fontsize=16, fontweight='bold')
        
        if result.test_type == TestType.CONVERSION:
            self._plot_conversion_results(data, result, axes)
        else:
            self._plot_continuous_results(data, result, metric_col, axes)
        
        # Добавляем информационную панель
        self._add_info_panel(fig, result)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_conversion_results(self, data: pd.DataFrame, result: TestResult, axes):
        """Визуализация для тестов конверсии"""
        # 1. Гистограммы конверсий
        conversion_data = data.groupby('group')['converted'].agg(['count', 'sum']).reset_index()
        conversion_data['rate'] = conversion_data['sum'] / conversion_data['count']
        
        ax1 = axes[0, 0]
        bars = ax1.bar(conversion_data['group'], conversion_data['rate'], 
                      color=['skyblue', 'lightcoral'])
        ax1.set_title('Конверсия по группам')
        ax1.set_ylabel('Conversion Rate')
        ax1.set_ylim(0, max(conversion_data['rate']) * 1.2)
        
        # Добавляем значения на столбцы
        for bar, rate in zip(bars, conversion_data['rate']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.3f}', ha='center', va='bottom')
        
        # 2. Доверительный интервал
        ax2 = axes[0, 1]
        ci_lower, ci_upper = result.confidence_interval
        diff = result.group_b_metric - result.group_a_metric
        
        ax2.errorbar([0], [diff], yerr=[[diff - ci_lower], [ci_upper - diff]], 
                    fmt='o', capsize=5, capthick=2, color='red', markersize=8)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Доверительный интервал разности')
        ax2.set_ylabel('Разность конверсий')
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_xticks([])
        
        # 3. Размеры выборок
        ax3 = axes[1, 0]
        sample_sizes = [result.group_a_size, result.group_b_size]
        ax3.bar(['Group A', 'Group B'], sample_sizes, color=['skyblue', 'lightcoral'])
        ax3.set_title('Размеры выборок')
        ax3.set_ylabel('Количество пользователей')
        
        # 4. Статистические метрики
        ax4 = axes[1, 1]
        metrics = ['p-value', 'Effect Size', 'Power']
        values = [result.p_value, abs(result.effect_size), result.power]
        
        colors = ['red' if result.p_value < 0.05 else 'gray', 'blue', 'green']
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_title('Статистические метрики')
        ax4.set_ylabel('Значение')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_continuous_results(self, data: pd.DataFrame, result: TestResult, 
                                metric_col: str, axes):
        """Визуализация для непрерывных метрик"""
        group_a_data = data[data['group'] == 'A'][metric_col]
        group_b_data = data[data['group'] == 'B'][metric_col]
        
        # 1. Распределения
        ax1 = axes[0, 0]
        ax1.hist(group_a_data, alpha=0.7, bins=30, label='Group A', color='skyblue')
        ax1.hist(group_b_data, alpha=0.7, bins=30, label='Group B', color='lightcoral')
        ax1.set_title('Распределения метрики')
        ax1.set_xlabel(metric_col)
        ax1.set_ylabel('Частота')
        ax1.legend()
        
        # 2. Box plots
        ax2 = axes[0, 1]
        data.boxplot(column=metric_col, by='group', ax=ax2)
        ax2.set_title('Box plots по группам')
        ax2.set_xlabel('Группа')
        ax2.set_ylabel(metric_col)
        
        # 3. Доверительный интервал
        ax3 = axes[1, 0]
        ci_lower, ci_upper = result.confidence_interval
        diff = result.group_b_metric - result.group_a_metric
        
        ax3.errorbar([0], [diff], yerr=[[diff - ci_lower], [ci_upper - diff]], 
                    fmt='o', capsize=5, capthick=2, color='red', markersize=8)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Доверительный интервал разности')
        ax3.set_ylabel(f'Разность средних ({metric_col})')
        ax3.set_xlim(-0.5, 0.5)
        ax3.set_xticks([])
        
        # 4. Статистические метрики
        ax4 = axes[1, 1]
        metrics = ['p-value', 'Effect Size', 'Power']
        values = [result.p_value, abs(result.effect_size), result.power]
        
        colors = ['red' if result.p_value < 0.05 else 'gray', 'blue', 'green']
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_title('Статистические метрики')
        ax4.set_ylabel('Значение')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
    
    def _add_info_panel(self, fig, result: TestResult):
        """Добавляет информационную панель с результатами"""
        info_text = f"""
        📊 РЕЗУЛЬТАТЫ A/B ТЕСТА
        
        Размеры выборок: A = {result.group_a_size}, B = {result.group_b_size}
        Метрика A: {result.group_a_metric:.4f}
        Метрика B: {result.group_b_metric:.4f}
        
        p-value: {result.p_value:.4f}
        Размер эффекта: {result.effect_size:.4f}
        Мощность теста: {result.power:.4f}
        
        Доверительный интервал: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]
        
        🎯 ВЫВОД: {result.conclusion}
        """
        
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                verticalalignment='bottom')
    
    def sample_size_calculator(self, baseline_rate: float, 
                             expected_lift: float, 
                             power: float = 0.8,
                             alpha: float = 0.05) -> int:
        """
        Рассчитывает необходимый размер выборки для A/B теста
        
        Args:
            baseline_rate: Базовая конверсия
            expected_lift: Ожидаемый прирост (в долях, например 0.1 для 10%)
            power: Желаемая мощность теста
            alpha: Уровень значимости
            
        Returns:
            Необходимый размер выборки для каждой группы
        """
        treatment_rate = baseline_rate * (1 + expected_lift)
        
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        
        p_avg = (baseline_rate + treatment_rate) / 2
        
        n = (2 * p_avg * (1 - p_avg) * (z_alpha + z_beta)**2) / (baseline_rate - treatment_rate)**2
        
        return int(np.ceil(n))
    
    def generate_report(self, result: TestResult) -> str:
        """
        Генерирует текстовый отчёт по результатам A/B теста
        
        Args:
            result: Результаты теста
            
        Returns:
            Форматированный отчёт
        """
        report = f"""
        ╔═══════════════════════════════════════════════════════════════╗
        ║                        A/B TEST REPORT                        ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║ Тип теста: {result.test_type.value.upper()}
        ║ Уровень значимости: {self.alpha}
        ║ Доверительный уровень: {self.confidence_level:.1%}
        ╠═══════════════════════════════════════════════════════════════╣
        ║ РАЗМЕРЫ ВЫБОРОК:
        ║   • Группа A (контроль): {result.group_a_size:,}
        ║   • Группа B (тест):     {result.group_b_size:,}
        ║   • Общий размер:        {result.group_a_size + result.group_b_size:,}
        ╠═══════════════════════════════════════════════════════════════╣
        ║ ОСНОВНЫЕ МЕТРИКИ:
        ║   • Группа A: {result.group_a_metric:.4f}
        ║   • Группа B: {result.group_b_metric:.4f}
        ║   • Разность: {result.group_b_metric - result.group_a_metric:.4f}
        ║   • Относительное изменение: {((result.group_b_metric / result.group_a_metric - 1) * 100):+.2f}%
        ╠═══════════════════════════════════════════════════════════════╣
        ║ СТАТИСТИЧЕСКИЕ РЕЗУЛЬТАТЫ:
        ║   • p-value:           {result.p_value:.6f}
        ║   • Размер эффекта:    {result.effect_size:.4f}
        ║   • Мощность теста:    {result.power:.4f}
        ║   • Доверительный интервал: [{result.confidence_interval[0]:+.4f}, {result.confidence_interval[1]:+.4f}]
        ╠═══════════════════════════════════════════════════════════════╣
        ║ СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ: {"✓ ДА" if result.is_significant else "✗ НЕТ"}
        ║
        ║ ВЫВОД:
        ║ {result.conclusion}
        ║
        """
        
        if result.is_significant:
            report += "║ 🎯 РЕКОМЕНДАЦИЯ: Изменение имеет статистически значимый\n"
            report += "║    эффект. Рассмотрите внедрение изменений.\n"
        else:
            report += "║ ⚠️  РЕКОМЕНДАЦИЯ: Недостаточно данных для принятия решения.\n"
            report += "║    Продолжите тест или увеличьте размер выборки.\n"
        
        report += "╚═══════════════════════════════════════════════════════════════╝"
        
        return report