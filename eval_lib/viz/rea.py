
import sys
import numpy as np  
import pandas as pd  
from scipy import stats  
from scipy.stats import f  
from pathlib import Path  
import warnings  
import seaborn as sns  
import matplotlib.pyplot as plt  
import os  
import shutil  

# Geometry-based layout system
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.visualization import (
    apply_style as _apply_geometry_style,
    style_axes, save_with_vcd,
    bind_figure_region, LayoutRegion)

_STYLE_APPLIED = False

def _apply_rea_style():
    """Apply REA plotting defaults via the geometry-based style system."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    _apply_geometry_style()
    sns.set_palette("husl")
    _STYLE_APPLIED = True


# ============================================================================
# Global layout constants
# ============================================================================

# Hard minimum x-tick font size (pt).  No adaptive logic or CLI override
# should ever produce a value below this — readability is non-negotiable.
MIN_XTICK_FONTSIZE: float = 7.0

# Per-group figures (single-row, wide) use a slightly higher minimum so that
# individual metric panels remain comfortable to read at glance.
MIN_XTICK_FONTSIZE_PER_GROUP: float = 8.0


def clamp_xtick_fontsize(fs: float, *, per_group: bool = False) -> float:
    """Return *fs* clamped to the appropriate minimum x-tick size.

    Parameters
    ----------
    fs : float
        Desired font size in points.
    per_group : bool
        If True, use the stricter per-group minimum (8 pt); otherwise
        use the global minimum (7 pt).
    """
    floor = MIN_XTICK_FONTSIZE_PER_GROUP if per_group else MIN_XTICK_FONTSIZE
    return max(fs, floor)


def needs_method_split(
    method_names: list[str],
    max_methods_per_panel: int = 10,
    max_avg_label_len: int = 18) -> bool:
    """Decide whether *method_names* should be split across multiple figures.

    The heuristic fires when **any** of the following conditions hold:

    1. Method count exceeds *max_methods_per_panel*.
    2. The mean method-name length exceeds *max_avg_label_len* **and** the
       method count is above half the threshold.

    Returns ``True`` when splitting is recommended.
    """
    n = len(method_names)
    if n > max_methods_per_panel:
        return True
    avg_len = sum(len(m) for m in method_names) / max(n, 1)
    if avg_len > max_avg_label_len and n > max_methods_per_panel // 2:
        return True
    return False


# ============================================================================
# Series-specific color palettes
# ============================================================================
SERIES_PALETTES = {
    # Experiment 1: Model ablation (5 variants)
    'ablation': ['#8dd3c7', '#fb8072', '#80b1d3', '#bebada', '#e41a1c'],
    # Experiment 2: GM-VAE geometric distributions benchmark (6 variants)
    'gmvae_benchmark': ['#fdb462', '#b3de69', '#fccde5', '#bc80bd', '#80b1d3', '#e41a1c'],
    # Experiment 3: Disentanglement regularization (6 variants)
    'disentanglement': ['#d9d9d9', '#ccebc5', '#ffed6f', '#377eb8', '#984ea3', '#e41a1c'],
    'benchmark': 'husl',  # default for external benchmarks
}


def _resolve_font_family(requested):
    """Resolve a font family name to one that matplotlib can actually render.
    Falls back through: requested → Liberation Sans → DejaVu Sans."""
    import matplotlib.font_manager as fm
    _FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'fonts')
    if os.path.isdir(_FONTS_DIR):
        for f in os.listdir(_FONTS_DIR):
            if f.lower().endswith(('.ttf', '.otf')):
                fp = os.path.join(_FONTS_DIR, f)
                try:
                    fm.fontManager.addfont(fp)
                except Exception:
                    pass
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in [requested, 'Liberation Sans', 'DejaVu Sans']:
        if candidate and candidate in available:
            return candidate
    return 'sans-serif'


def _apply_font(font_family):
    """Apply resolved font to matplotlib rcParams."""
    resolved = _resolve_font_family(font_family)
    plt.rcParams['font.family'] = resolved
    return resolved


def _safe_save_figure(fig, save_path, dpi, max_pixels=20000):
    """Save figure with DPI auto-clamped to avoid backend truncation.

    Uses the geometry-based ``save_with_vcd`` for deterministic export,
    with DPI safety-clamping when the requested size would exceed
    *max_pixels* in any dimension.
    """
    width_in, height_in = fig.get_size_inches()
    width_px = width_in * dpi
    height_px = height_in * dpi

    max_dim_px = max(width_px, height_px)
    if max_dim_px > max_pixels:
        scale = max_pixels / max_dim_px
        safe_dpi = max(72, int(dpi * scale))
        print(
            f"[eval_lib.viz] Auto-clamping DPI from {dpi} to {safe_dpi} "
            f"to keep figure within {max_pixels}px (w={width_px:.0f}, "
            f"h={height_px:.0f})."
        )
    else:
        safe_dpi = dpi

    save_with_vcd(fig, Path(save_path), dpi=safe_dpi)


def prepare_merged_data_folder(merge_config, output_folder):  
    """  
    .. deprecated::
        Legacy merge function. Prefer ``eval_lib.experiment.merge.MergedExperimentConfig``
        which uses column-based filtering (more robust) and also merges series CSVs.

    从多个源文件夹中提取指定方法的数据，合并到新的CSV文件中。  
    此版本允许用户为每个源目录提供完整的方法名列表，以应对CSV中索引不规范的情况。  

    参数:  
        merge_config (dict): 一个配置字典。  
            - key: 源文件夹的路径 (str)。  
            - value: 一个字典，包含:  
                - 'all_methods' (list): 对应文件夹中CSV文件的完整、有序的方法名列表。  
                - 'selected_methods' (list): 需要从该文件夹中提取的方法名列表。  
        
        output_folder (str): 用于存放新生成的合并后CSV文件的文件夹路径。  

    返回:  
        tuple: (output_folder_path, final_method_names)  
            - output_folder_path (str): 输出文件夹的路径。  
            - final_method_names (list): 合并后所有方法的列表，可直接用于初始化分析器。  
    """  
    print("🔄 Starting robust data merge process...")  
    output_path = Path(output_folder)  
    
    if output_path.exists():  
        shutil.rmtree(output_path)  
    output_path.mkdir(parents=True)  
    
    source_paths = list(merge_config.keys())  
    if not source_paths:  
        raise ValueError("merge_config cannot be empty.")  

    base_folder = Path(source_paths[0])  
    try:  
        base_csv_files = [p.name for p in base_folder.glob("*.csv")]  
    except FileNotFoundError:  
        raise FileNotFoundError(f"The base source directory does not exist: {base_folder}")  

    if not base_csv_files:  
        print(f"⚠️ No CSV files found in the base directory: {base_folder}")  
        return str(output_path), []  

    print(f"📁 Found {len(base_csv_files)} datasets in the base folder '{base_folder.name}'. Verifying and merging...")  

    final_method_names = []  
    for folder_config in merge_config.values():  
        final_method_names.extend(folder_config['selected_methods'])  

    for csv_file_name in base_csv_files:  
        merged_rows = []  
        is_consistent = True  

        for folder in source_paths:  
            if not (Path(folder) / csv_file_name).exists():  
                print(f"⚠️ Skipping '{csv_file_name}': Not found in folder '{folder}'.")  
                is_consistent = False  
                break  
        
        if not is_consistent:  
            continue  
            
        for source_folder, folder_config in merge_config.items():  
            file_path = Path(source_folder) / csv_file_name  
            
            all_methods_in_folder = folder_config['all_methods']  
            methods_to_select = folder_config['selected_methods']  
            
            # 读取CSV时不指定索引列  
            df = pd.read_csv(file_path, header=0)  
            
            # **FIX 1: Discard the first column to remove the old index**  
            df_data_only = df.iloc[:, 1:]  
            
            if len(df_data_only) != len(all_methods_in_folder):  
                 raise ValueError(  
                     f"Row count mismatch in file '{file_path}'. "  
                     f"The CSV has {len(df_data_only)} rows, but 'all_methods' config has {len(all_methods_in_folder)} entries."  
                 )  
            
            df_data_only.index = all_methods_in_folder  
            
            selected_rows = df_data_only.loc[methods_to_select]  
            merged_rows.append(selected_rows)  
            
        if merged_rows:  
            final_df = pd.concat(merged_rows).loc[final_method_names]  
            
            # **FIX 2: Name the index before writing to the file**  
            final_df.index.name = "method"  
            
            final_df.to_csv(output_path / csv_file_name, index=True)  
    
    print(f"✅ Data merge completed. Merged data is in: '{output_path}'")  
    return str(output_path), final_method_names 


class RigorousExperimentalAnalyzer:
    """
    科学严谨的实验数据分析框架
    
    核心特性：
    - 严格的统计方法选择和实现
    - 完整的事后检验
    - 用户定义方法名称
    - 配对vs独立设计自动检测
    - 完整结果记录和可视化
    """
    
    def __init__(self, data_folder_path, method_names, selected_methods=None, 
                 method_order=None, verbose=True):
        """
        初始化分析器
        
        参数:
            data_folder_path: 包含CSV文件的文件夹路径
            method_names: 完整方法名称列表，按DataFrame索引顺序
            selected_methods: 选择要分析的方法子集（None表示全部）
            method_order: 方法显示顺序（None表示使用原顺序）
            verbose: 是否显示详细信息
            
        示例:
            # 使用全部方法
            analyzer = RigorousExperimentalAnalyzer(folder, ['VAE', 'ODE-VAE', 'β-VAE'])
            
            # 只分析其中3个方法
            analyzer = RigorousExperimentalAnalyzer(
                folder, 
                ['VAE', 'ODE-VAE', 'β-VAE', 'WAE', 'InfoVAE'], 
                selected_methods=['VAE', 'ODE-VAE', 'β-VAE']
            )
            
            # 自定义显示顺序
            analyzer = RigorousExperimentalAnalyzer(
                folder,
                ['VAE', 'ODE-VAE', 'β-VAE'],
                method_order=['β-VAE', 'VAE', 'ODE-VAE']
            )
        """
        _apply_rea_style()
        self.data_folder_path = Path(data_folder_path)
        self.all_method_names = method_names  # 完整的方法列表
        self.verbose = verbose
        
        # 处理方法选择和排序
        if selected_methods is not None:
            # 验证选择的方法是否在完整列表中
            invalid_methods = [m for m in selected_methods if m not in method_names]
            if invalid_methods:
                raise ValueError(f"Selected methods not in method_names: {invalid_methods}")
            self.method_names = selected_methods
        else:
            self.method_names = method_names.copy()
        
        # 处理方法显示顺序
        if method_order is not None:
            # 验证排序列表
            if set(method_order) != set(self.method_names):
                raise ValueError("method_order must contain exactly the same methods as selected methods")
            self.method_names = method_order
        
        # 创建方法索引映射（从原始DataFrame索引到方法名）
        method_index_map = {name:i for i, name in enumerate(self.all_method_names)}
        self.selected_indices = [method_index_map[name] for name in self.method_names]
        
        # 数据存储
        self.raw_data = None
        self.processed_data = None
        self.metrics = None
        self.n_datasets = 0
        
        # 统计结果存储
        self.statistical_results = {}
        self.design_type = None
        
        self._log(f"✅ Analyzer initialized:")
        self._log(f"   - All methods: {self.all_method_names}")
        self._log(f"   - Selected methods: {self.method_names}")
        self._log(f"   - Selected indices: {self.selected_indices}")
        
    def _log(self, message):
        """打印日志信息"""
        if self.verbose:
            print(message)
    
    # ==================== 1. 数据加载和预处理 ====================
    
    def load_experimental_data(self):
        """
        从文件夹加载所有实验数据（只保留选择的方法）
        """
        self._log("🔄 Loading experimental data...")
        
        csv_files = list(self.data_folder_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_folder_path}")
        
        self._log(f"📁 Found {len(csv_files)} dataset files")
        
        # 加载所有CSV文件
        all_data = []
        for i, file_path in enumerate(csv_files):
            try:
                df = pd.read_csv(file_path, header=0, index_col=0)
                
                # 验证数据结构
                if len(df) != len(self.all_method_names):
                    raise ValueError(f"File {file_path} has {len(df)} rows, expected {len(self.all_method_names)}")
                
                # 只保留选择的方法行
                selected_df = df.iloc[self.selected_indices].copy()
                
                # 重新索引使用选择的方法名
                selected_df.index = self.method_names
                selected_df['dataset_id'] = i + 1
                selected_df['dataset_name'] = file_path.stem
                all_data.append(selected_df)
                
            except Exception as e:
                self._log(f"⚠️ Error loading file {file_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError("Failed to load any data files")
        
        # 合并所有数据
        self.raw_data = pd.concat(all_data, ignore_index=False)
        self.n_datasets = len(csv_files)
        
        # 获取数值列作为指标
        exclude_cols = ['dataset_id', 'dataset_name', 'data_type_intrin', 'interpretation_intrin']
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        self.metrics = [col for col in numeric_cols if col not in exclude_cols]
        
        # 检测实验设计类型
        self._detect_design_type()
        
        self._log(f"✅ Data loading completed:")
        self._log(f"   - Methods: {len(self.method_names)} {self.method_names}")
        self._log(f"   - Metrics: {len(self.metrics)}")
        self._log(f"   - Datasets: {self.n_datasets}")
        self._log(f"   - Design type: {self.design_type}")
        
        return self.raw_data
    
    def _detect_design_type(self):
        """
        检测实验设计类型：配对 vs 独立
        """
        # 检查每个数据集是否包含所有方法的数据
        complete_datasets = 0
        for dataset_id in range(1, self.n_datasets + 1):
            dataset_data = self.raw_data[self.raw_data['dataset_id'] == dataset_id]
            if len(dataset_data) == len(self.method_names):
                complete_datasets += 1
        
        # 如果大部分数据集都包含所有方法，则为配对设计
        if complete_datasets >= self.n_datasets * 0.8:
            self.design_type = 'paired'
        else:
            self.design_type = 'independent'
    
    def preprocess_data(self):
        """
        预处理数据为分析格式
        
        返回:
            pandas.DataFrame: 长格式的处理后数据
        """
        if self.raw_data is None:
            self.load_experimental_data()
        
        self._log("🔄 Preprocessing data...")
        
        # 转换为长格式
        processed_data = []
        
        for _, row in self.raw_data.iterrows():
            for metric in self.metrics:
                if pd.notna(row[metric]):
                    processed_data.append({
                        'method': row.name,  # 使用索引作为方法名
                        'metric': metric,
                        'value': row[metric],
                        'dataset_id': row['dataset_id'],
                        'dataset_name': row['dataset_name']
                    })
        
        self.processed_data = pd.DataFrame(processed_data)
        
        self._log(f"✅ Data preprocessing completed, {len(self.processed_data)} data points")
        
        return self.processed_data
    
    # ==================== 2. 统计分析实现 ====================
    
    def _repeated_measures_anova(self, data_matrix):
        """
        实现重复测量ANOVA
        
        参数:
            data_matrix: 数据矩阵 (subjects x conditions)
            
        返回:
            F统计量, p值
        """
        n_subjects, n_conditions = data_matrix.shape
        
        # 计算总均值
        grand_mean = np.mean(data_matrix)
        
        # 计算平方和
        # Total Sum of Squares
        ss_total = np.sum((data_matrix - grand_mean) ** 2)
        
        # Between-subjects Sum of Squares
        subject_means = np.mean(data_matrix, axis=1)
        ss_between_subjects = n_conditions * np.sum((subject_means - grand_mean) ** 2)
        
        # Within-subjects Sum of Squares
        ss_within_subjects = ss_total - ss_between_subjects
        
        # Between-conditions Sum of Squares (treatment effect)
        condition_means = np.mean(data_matrix, axis=0)
        ss_between_conditions = n_subjects * np.sum((condition_means - grand_mean) ** 2)
        
        # Error Sum of Squares
        ss_error = ss_within_subjects - ss_between_conditions
        
        # 自由度
        df_between_conditions = n_conditions - 1
        df_error = (n_subjects - 1) * (n_conditions - 1)
        
        # 均方
        ms_between_conditions = ss_between_conditions / df_between_conditions
        ms_error = ss_error / df_error
        
        # F统计量
        if ms_error > 1e-10:
            f_statistic = ms_between_conditions / ms_error
            p_value = 1 - f.cdf(f_statistic, df_between_conditions, df_error)
        else:
            f_statistic = np.inf
            p_value = 0.0
        
        return f_statistic, p_value
    
    def _tukey_hsd_post_hoc(self, data_arrays, alpha=0.05):
        """
        实现Tukey HSD事后检验
        
        参数:
            data_arrays: 数据数组列表
            alpha: 显著性水平
            
        返回:
            事后检验结果
        """
        k = len(data_arrays)  # 组数
        n_total = sum(len(arr) for arr in data_arrays)
        
        # 计算组内误差均方 (MSE)
        ss_within = 0
        df_within = 0
        for arr in data_arrays:
            if len(arr) > 1:
                ss_within += np.sum((arr - np.mean(arr)) ** 2)
                df_within += len(arr) - 1
        
        if df_within > 0:
            mse = ss_within / df_within
        else:
            mse = 1e-10
        
        # 计算临界值 (简化：使用Bonferroni校正代替Tukey分布)
        n_comparisons = k * (k - 1) // 2
        alpha_corrected = alpha / n_comparisons
        
        # 执行所有配对比较
        results = []
        for i in range(k):
            for j in range(i + 1, k):
                data1, data2 = data_arrays[i], data_arrays[j]
                
                # 计算均值差
                mean_diff = np.mean(data2) - np.mean(data1)
                
                # 计算标准误
                if len(data1) > 0 and len(data2) > 0:
                    se = np.sqrt(mse * (1/len(data1) + 1/len(data2)))
                    
                    if se > 1e-10:
                        t_stat = mean_diff / se
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_within))
                        p_corrected = min(p_value * n_comparisons, 1.0)
                    else:
                        t_stat = 0
                        p_value = 1.0
                        p_corrected = 1.0
                else:
                    t_stat = 0
                    p_value = 1.0
                    p_corrected = 1.0
                
                results.append({
                    'group1_idx': i,
                    'group2_idx': j,
                    'mean_diff': mean_diff,
                    'se': se if 'se' in locals() else 0,
                    't_stat': t_stat,
                    'p_raw': p_value,
                    'p_corrected': p_corrected,
                    'significant': p_corrected < alpha
                })
        
        return results
    
    def perform_metric_analysis(self, metric_name):
        """
        对单个指标进行完整统计分析
        
        参数:
            metric_name: 指标名称
            
        返回:
            dict: 详细的统计分析结果
        """
        self._log(f"📊 Analyzing metric: {metric_name}")
        
        # 提取指标数据
        metric_data = {}
        for method in self.method_names:
            method_data = self.raw_data[self.raw_data.index == method][metric_name].dropna()
            metric_data[method] = method_data.values
        
        # 检查数据完整性
        data_sizes = {method: len(data) for method, data in metric_data.items()}
        valid_methods = [method for method, data in metric_data.items() if len(data) > 0]
        
        if len(valid_methods) < 2:
            return {'error': 'Insufficient valid data for analysis', 'metric': metric_name}
        
        results = {
            'metric': metric_name,
            'methods': valid_methods,
            'n_methods': len(valid_methods),
            'data_sizes': data_sizes,
            'design_type': self.design_type,
            'descriptive_stats': {}
        }
        
        # 描述性统计
        for method in valid_methods:
            data = metric_data[method]
            if len(data) > 0:
                results['descriptive_stats'][method] = {
                    'mean': np.mean(data),
                    'std': np.std(data, ddof=1) if len(data) > 1 else 0,
                    'median': np.median(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'n': len(data)
                }
        
        # 根据方法数量和设计类型选择分析
        if len(valid_methods) == 2:
            results.update(self._analyze_two_groups(metric_data, valid_methods))
        else:
            results.update(self._analyze_multiple_groups(metric_data, valid_methods))
        
        return results
    
    def _analyze_two_groups(self, metric_data, valid_methods):
        """
        两组比较的统计分析
        """
        method1, method2 = valid_methods[0], valid_methods[1]
        data1, data2 = metric_data[method1], metric_data[method2]
        
        results = {'analysis_type': 'two_groups'}
        
        # 正态性检验
        normality_results = {}
        for method, data in [(method1, data1), (method2, data2)]:
            if len(data) >= 3:
                _, p_norm = stats.shapiro(data)
                normality_results[method] = p_norm
            else:
                normality_results[method] = 0.0
        
        both_normal = all(p > 0.05 for p in normality_results.values())
        results['normality_tests'] = {**normality_results, 'both_normal': both_normal}
        
        # 根据设计类型和数据分布选择统计检验
        if self.design_type == 'paired':
            if len(data1) != len(data2):
                results['error'] = 'Paired design requires equal sample sizes'
                return results
            
            if both_normal:
                stat, p_value = stats.ttest_rel(data2, data1)
                test_name = "Paired t-test"
            else:
                stat, p_value = stats.wilcoxon(data2, data1, alternative='two-sided')
                test_name = "Wilcoxon signed-rank test"
            
            # 配对效应量
            differences = data2 - data1
            if np.std(differences, ddof=1) > 1e-10:
                effect_size = np.mean(differences) / np.std(differences, ddof=1)
            else:
                effect_size = 0.0
            effect_type = "Cohen's d (paired)"
            
        else:  # independent design
            if both_normal:
                # 方差齐性检验
                _, p_var = stats.levene(data1, data2)
                equal_var = p_var > 0.05
                
                stat, p_value = stats.ttest_ind(data2, data1, equal_var=equal_var)
                test_name = f"Independent t-test ({'equal' if equal_var else 'unequal'} variance)"
                
                results['variance_test'] = {'p_value': p_var, 'equal_var': equal_var}
            else:
                stat, p_value = stats.mannwhitneyu(data2, data1, alternative='two-sided')
                test_name = "Mann-Whitney U test"
            
            # 独立样本效应量
            pooled_std = np.sqrt(((len(data1)-1)*np.var(data1, ddof=1) + 
                                 (len(data2)-1)*np.var(data2, ddof=1)) / 
                                 (len(data1)+len(data2)-2))
            if pooled_std > 1e-10:
                effect_size = (np.mean(data2) - np.mean(data1)) / pooled_std
            else:
                effect_size = 0.0
            effect_type = "Cohen's d (independent)"
        
        results['main_test'] = {
            'test_name': test_name,
            'statistic': stat,
            'p_value': p_value
        }
        
        results['effect_size'] = {
            'value': effect_size,
            'type': effect_type
        }
        
        # 改进分析
        if np.abs(np.mean(data1)) > 1e-10:
            improvement_pct = ((np.mean(data2) - np.mean(data1)) / np.abs(np.mean(data1))) * 100
        else:
            improvement_pct = 0.0
        
        results['improvement'] = {
            'percentage': improvement_pct,
            'absolute': np.mean(data2) - np.mean(data1)
        }
        
        return results
    
    def _analyze_multiple_groups(self, metric_data, valid_methods):
        """
        多组比较的统计分析
        """
        results = {'analysis_type': 'multiple_groups'}
        
        # 准备数据
        data_arrays = [metric_data[method] for method in valid_methods]
        
        # 检查数据大小一致性
        data_sizes = [len(arr) for arr in data_arrays]
        is_balanced = len(set(data_sizes)) == 1 and min(data_sizes) > 0
        
        # 正态性检验
        normality_results = {}
        all_normal = True
        for i, method in enumerate(valid_methods):
            data = data_arrays[i]
            if len(data) >= 3:
                _, p_norm = stats.shapiro(data)
                normality_results[method] = p_norm
                if p_norm <= 0.05:
                    all_normal = False
            else:
                normality_results[method] = 0.0
                all_normal = False
        
        results['normality_tests'] = {**normality_results, 'all_normal': all_normal}
        results['balanced_design'] = is_balanced
        
        # 选择主要统计检验
        if self.design_type == 'paired' and is_balanced:
            if all_normal:
                # 重复测量ANOVA
                data_matrix = np.column_stack(data_arrays)
                stat, p_value = self._repeated_measures_anova(data_matrix)
                test_name = "Repeated measures ANOVA"
            else:
                # Friedman检验
                stat, p_value = stats.friedmanchisquare(*data_arrays)
                test_name = "Friedman test"
        else:  # independent design
            if all_normal and is_balanced:
                # 单因素ANOVA
                stat, p_value = stats.f_oneway(*data_arrays)
                test_name = "One-way ANOVA"
            else:
                # Kruskal-Wallis检验
                stat, p_value = stats.kruskal(*data_arrays)
                test_name = "Kruskal-Wallis test"
        
        results['main_test'] = {
            'test_name': test_name,
            'statistic': stat,
            'p_value': p_value
        }
        
        # 效应量计算
        if 'ANOVA' in test_name:
            # Eta squared for ANOVA
            all_data = np.concatenate(data_arrays)
            grand_mean = np.mean(all_data)
            
            ss_total = np.sum((all_data - grand_mean) ** 2)
            ss_between = sum(len(arr) * (np.mean(arr) - grand_mean) ** 2 for arr in data_arrays)
            
            eta_squared = ss_between / ss_total if ss_total > 1e-10 else 0
            results['effect_size'] = {
                'value': eta_squared,
                'type': 'eta squared'
            }
        else:
            # Epsilon squared for non-parametric tests
            n_total = sum(len(arr) for arr in data_arrays)
            k = len(data_arrays)
            epsilon_squared = (stat - k + 1) / (n_total - k) if (n_total - k) > 0 else 0
            results['effect_size'] = {
                'value': max(0, epsilon_squared),
                'type': 'epsilon squared (approximate)'
            }
        
        # 事后检验（如果主检验显著）
        if p_value < 0.05:
            results['post_hoc'] = self._perform_rigorous_post_hoc(
                data_arrays, valid_methods, test_name, all_normal
            )
        
        return results
    
    def _perform_rigorous_post_hoc(self, data_arrays, valid_methods, main_test_name, all_normal):
        """
        执行严格的事后检验
        """
        k = len(data_arrays)
        n_comparisons = k * (k - 1) // 2
        results = {'method': 'Unknown', 'comparisons': []}
        
        # 根据主检验选择合适的事后检验
        if 'ANOVA' in main_test_name and all_normal:
            # Tukey HSD for ANOVA
            tukey_results = self._tukey_hsd_post_hoc(data_arrays)
            results['method'] = 'Tukey HSD'
            
            for i, comp in enumerate(tukey_results):
                method1 = valid_methods[comp['group1_idx']]
                method2 = valid_methods[comp['group2_idx']]
                
                results['comparisons'].append({
                    'method1': method1,
                    'method2': method2,
                    'test_used': 'Tukey HSD',
                    'mean_diff': comp['mean_diff'],
                    'p_corrected': comp['p_corrected'],
                    'significant': comp['significant'],
                    'effect_size': comp['mean_diff'] / comp['se'] if comp['se'] > 1e-10 else 0
                })
        
        else:
            # Bonferroni校正的配对检验
            alpha_corrected = 0.05 / n_comparisons
            results['method'] = f'Bonferroni correction (α = {alpha_corrected:.4f})'
            
            for i in range(k):
                for j in range(i + 1, k):
                    data1, data2 = data_arrays[i], data_arrays[j]
                    method1, method2 = valid_methods[i], valid_methods[j]
                    
                    # 选择合适的检验
                    if self.design_type == 'paired' and len(data1) == len(data2):
                        if all_normal:
                            stat, p_raw = stats.ttest_rel(data2, data1)
                            test_used = "Paired t-test"
                        else:
                            stat, p_raw = stats.wilcoxon(data2, data1, alternative='two-sided')
                            test_used = "Wilcoxon signed-rank test"
                        
                        # 配对效应量
                        differences = data2 - data1
                        effect_size = (np.mean(differences) / np.std(differences, ddof=1) 
                                     if np.std(differences, ddof=1) > 1e-10 else 0)
                    else:
                        if all_normal:
                            stat, p_raw = stats.ttest_ind(data2, data1)
                            test_used = "Independent t-test"
                        else:
                            stat, p_raw = stats.mannwhitneyu(data2, data1, alternative='two-sided')
                            test_used = "Mann-Whitney U test"
                        
                        # 独立样本效应量
                        pooled_std = np.sqrt(((len(data1)-1)*np.var(data1, ddof=1) + 
                                             (len(data2)-1)*np.var(data2, ddof=1)) / 
                                             (len(data1)+len(data2)-2))
                        effect_size = ((np.mean(data2) - np.mean(data1)) / pooled_std 
                                     if pooled_std > 1e-10 else 0)
                    
                    p_corrected = min(p_raw * n_comparisons, 1.0)
                    
                    results['comparisons'].append({
                        'method1': method1,
                        'method2': method2,
                        'test_used': test_used,
                        'statistic': stat,
                        'p_raw': p_raw,
                        'p_corrected': p_corrected,
                        'significant': p_corrected < 0.05,
                        'effect_size': effect_size,
                        'mean_diff': np.mean(data2) - np.mean(data1)
                    })
        
        results['n_comparisons'] = n_comparisons
        return results
    
    
    # ==================== 完全可定制的可视化功能 ====================
    def create_metric_comparison_plot(self, metric_name, plot_type='boxplot',
                                show_significance_pairs=None,
                                max_significance_pairs=5,
                                datasets_to_show=20,
                                display_name=None,
                                stat_test_display='short',
                                # 图形创建控制
                                ax=None,  
                                figsize=(5, 5),
                                dpi=150,
    
                                # 字体大小控制
                                title_fontsize=12,
                                title_fontweight='normal',
                                axis_label_fontsize=12,
                                tick_label_fontsize=12,
                                significance_fontsize=14,
                                ns_fontsize=10,
                                legend_fontsize=8,
                                show_legend=True,  
    
                                # 颜色控制
                                palette='Set2',
                                box_color=None,
                                strip_color='black',
                                strip_alpha=0.6,
                                line_color='black',
                                mean_line_color='red',
                                # 新增：柱状图颜色控制
                                bar_color=None,
                                bar_edge_color='black',
                                bar_edge_width=1,
                                bar_alpha=0.7,
    
                                # 显著性标注控制
                                significance_marker_map={'***': '***', '**': '**', '*': '*', 'ns': 'ns'},
                                significance_y_offset=0.05,
                                significance_marker_offset=0.02,
                                ns_offset=0.01,
                                significance_line_width=3,
                                
                                # 图形元素控制
                                box_width=0.7,
                                jitter_width=0.35,
                                strip_size=4,
                                flier_size=0, 
                                flier_marker='o',  
                                flier_alpha=0.7,  
                                mean_line_width=3,
                                mean_marker_size=8,
                                # 新增：柱状图控制参数
                                bar_width=0.6,
                                show_error_bars=True,
                                error_bar_type='std',  # 'std', 'sem', 'ci95', 'ci99', 'none'
                                error_bar_capsize=5,
                                error_bar_capthick=2,
                                error_bar_color='black',
                                error_bar_alpha=0.8,
                                error_bar_width=2,
                                # 新增：柱状图上的散点控制
                                show_bar_points=True,
                                bar_strip_size=3,
                                bar_strip_alpha=0.6,
                                bar_strip_color='black',
                                bar_strip_jitter_width=0.4,
    
                                # 边框和背景控制
                                title_bbox_props=None,
                                legend_bbox_props={'boxstyle': 'round', 'facecolor': 'lightblue', 'alpha': 0.6},
                                annotation_bbox_props={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.6},
    
                                # 可视化样式控制
                                grid=True,
                                grid_alpha=0.3,
                                rotate_xlabels=None,
                                xlabel_rotation=45,
                                xlabel_ha='right',
                                font_family='Arial',  
                                spines_visible=None,  
    
                                # 布局控制
                                tight_layout=True,
                                subplot_adjust_params=None):
        """
        Create highly customizable metric comparison visualization with statistical annotations.
    
        Generates publication-ready statistical comparison plots with automatic test selection,
        post-hoc analysis, and extensive visual customization options. Supports multiple plot
        types including boxplots, violin plots, bar plots with error bars, strip plots, and
        paired line plots.
    
        Parameters
        ----------
        metric_name : str
            Name of the metric to analyze and visualize. Must exist in the loaded dataset.
        plot_type : {'boxplot', 'violin', 'strip', 'barplot', 'paired_lines'}, default='boxplot'
            Type of plot to create:
            - 'boxplot': Box-and-whisker plots with optional outliers and strip overlay
            - 'violin': Violin plots showing distribution shape with optional strip overlay
            - 'strip': Strip plots with jittered points only
            - 'barplot': Bar plots with error bars and optional jittered points
            - 'paired_lines': Connected line plots for paired data with mean overlay
        show_significance_pairs : list of tuples, optional
            Specific method pairs to show significance annotations for.
            Each tuple should contain two method names, e.g., [('Method1', 'Method2')].
            If None, shows top significant pairs automatically.
        max_significance_pairs : int, default=5
            Maximum number of significance pairs to display when show_significance_pairs
            is None. Pairs are ranked by p-value significance.
        datasets_to_show : int, default=20
            Maximum number of individual datasets to display in paired_lines plot.
            Only affects 'paired_lines' plot type.
        display_name : str, optional
            Custom display name for the metric in plot title. If None, uses metric_name.
        stat_test_display : {'full', 'short', 'none'}, default='short'
            Statistical test information display mode:
            - 'full': Full test name and p-value
            - 'short': Abbreviated test name and p-value  
            - 'none': No statistical test information in title
    
        Figure Creation Parameters
        -------------------------
        ax : matplotlib.axes.Axes, optional
            Existing axes object to plot on. If None, creates new figure.
        figsize : tuple of float, default=(5, 5)
            Figure size in inches as (width, height). Only used when ax is None.
        dpi : int, default=150
            Dots per inch resolution. Use 72 for screen display, 300+ for publication.
    
        Typography Parameters
        --------------------
        title_fontsize : int or float, default=12
            Font size for plot title in points.
        title_fontweight : {'normal', 'bold', 'light', 'heavy'}, default='normal'
            Font weight for plot title.
        axis_label_fontsize : int or float, default=12
            Font size for x and y axis labels in points.
        tick_label_fontsize : int or float, default=12
            Font size for axis tick labels in points.
        significance_fontsize : int or float, default=14
            Font size for significance markers (*, **, ***) in points.
        ns_fontsize : int or float, default=10
            Font size for non-significant (ns) markers in points.
        legend_fontsize : int or float, default=8
            Font size for legend text in points.
        font_family : str, default='Arial'
            Font family for all text elements. Common options: 'Arial', 'Times New Roman',
            'DejaVu Sans', 'Helvetica'.
        show_legend : bool, default=True
            Whether to display legend and annotation information.
    
        Color Parameters
        ---------------
        palette : str or list, default='Set2'
            Color palette for plot elements. Can be seaborn palette name or list of colors.
        box_color : str or list, optional
            Specific color(s) for boxplot boxes. Overrides palette if provided.
        strip_color : str, default='black'
            Color for strip plot points overlay.
        strip_alpha : float, default=0.6
            Transparency for strip plot points (0.0 to 1.0).
        line_color : str, default='black'
            Color for significance annotation lines and paired line plots.
        mean_line_color : str, default='red'
            Color for mean lines in paired line plots.
        bar_color : str or list, optional
            Color(s) for bar plot bars. Can be single color or list. Overrides palette.
        bar_edge_color : str, default='black'
            Color for bar plot bar edges.
        bar_edge_width : float, default=1
            Width of bar plot bar edges in points.
        bar_alpha : float, default=0.7
            Transparency for bar plot bars (0.0 to 1.0).
    
        Statistical Significance Annotation Parameters
        ---------------------------------------------
        significance_marker_map : dict, default={'***': '***', '**': '**', '*': '*', 'ns': 'ns'}
            Custom mapping for significance level markers. Keys are default markers,
            values are display strings.
        significance_y_offset : float, default=0.05
            Vertical offset for significance annotations as fraction of y-range.
        significance_marker_offset : float, default=0.02
            Additional vertical offset for significance markers as fraction of y-range.
        ns_offset : float, default=0.01
            Vertical offset for non-significant markers as fraction of y-range.
        significance_line_width : float, default=3
            Width of significance annotation lines in points.
    
        Plot Element Parameters
        ----------------------
        box_width : float, default=0.6
            Width of boxplot boxes (0.0 to 1.0).
        strip_size : int or float, default=4
            Size of strip plot points.
        flier_size : int or float, default=0
            Size of boxplot outlier markers. Set to 0 to hide outliers.
        flier_marker : str, default='o'
            Marker style for boxplot outliers. Matplotlib marker specification.
        flier_alpha : float, default=0.7
            Transparency for boxplot outliers (0.0 to 1.0).
        mean_line_width : float, default=3
            Width of mean lines in paired line plots.
        mean_marker_size : int or float, default=8
            Size of mean markers in paired line plots.
    
        Bar Plot Specific Parameters
        ---------------------------
        bar_width : float, default=0.6
            Width of bars in bar plots (0.0 to 1.0).
        show_error_bars : bool, default=True
            Whether to display error bars on bar plots.
        error_bar_type : {'std', 'sem', 'ci95', 'ci99', 'none'}, default='std'
            Type of error bars:
            - 'std': Standard deviation
            - 'sem': Standard error of the mean
            - 'ci95': 95% confidence interval
            - 'ci99': 99% confidence interval
            - 'none': No error bars
        error_bar_capsize : float, default=5
            Size of error bar caps in points.
        error_bar_capthick : float, default=2
            Thickness of error bar caps in points.
        error_bar_color : str, default='black'
            Color of error bars.
        error_bar_alpha : float, default=0.8
            Transparency of error bars (0.0 to 1.0).
        error_bar_width : float, default=2
            Width of error bar lines in points.
        show_bar_points : bool, default=True
            Whether to show individual data points on bar plots with jitter.
        bar_strip_size : int or float, default=3
            Size of jittered points on bar plots.
        bar_strip_alpha : float, default=0.6
            Transparency of jittered points on bar plots (0.0 to 1.0).
        bar_strip_color : str, default='black'
            Color of jittered points on bar plots.
        bar_strip_jitter_width : float, default=0.4
            Width of jitter for points on bar plots (0.0 to 0.5).
    
        Style and Layout Parameters
        --------------------------
        title_bbox_props : dict, optional
            Background box properties for title. Format: {'boxstyle': 'round', 
            'facecolor': 'color', 'alpha': float}.
        legend_bbox_props : dict, default={'boxstyle': 'round', 'facecolor': 'lightblue', 'alpha': 0.6}
            Background box properties for legend annotations.
        annotation_bbox_props : dict, default={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.6}
            Background box properties for general annotations.
        grid : bool, default=True
            Whether to display grid lines.
        grid_alpha : float, default=0.3
            Transparency of grid lines (0.0 to 1.0).
        rotate_xlabels : bool, optional
            Whether to rotate x-axis labels. If None, auto-rotates when >3 methods.
        xlabel_rotation : float, default=45
            Rotation angle for x-axis labels in degrees.
        xlabel_ha : {'left', 'center', 'right'}, default='right'
            Horizontal alignment for rotated x-axis labels.
        spines_visible : dict, optional
            Control visibility of plot spines. Default: {'top': False, 'right': False}.
            Keys: 'top', 'bottom', 'left', 'right'. Values: bool.
        tight_layout : bool, default=True
            Whether to apply tight layout to prevent overlapping elements.
        subplot_adjust_params : dict, optional
            Manual subplot adjustment parameters. Format: {'left': float, 'right': float, 
            'top': float, 'bottom': float, 'hspace': float, 'wspace': float}.
    
        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the plot.
        ax : matplotlib.axes.Axes
            The matplotlib axes object with the plot.
    
        Raises
        ------
        ValueError
            If metric_name is not found in the dataset or if invalid parameter
            combinations are provided.
        KeyError
            If specified method names in show_significance_pairs are not found
            in the dataset.
    
        Notes
        -----
        This function automatically selects appropriate statistical tests based on:
        - Number of groups (2 groups vs multiple groups)
        - Data distribution (normality tests)
        - Experimental design (paired vs independent)
        - Sample sizes and balance
    
        Statistical tests include:
        - Two groups: t-tests, Mann-Whitney U, Wilcoxon signed-rank
        - Multiple groups: ANOVA, Kruskal-Wallis, Friedman, repeated measures ANOVA
        - Post-hoc: Tukey HSD, Bonferroni correction
    
        The significance annotation system supports multiple comparison correction
        and customizable significance thresholds (p < 0.05, 0.01, 0.001).
    
        Examples
        --------
        Basic usage with default parameters:
    
        >>> fig, ax = analyzer.create_metric_comparison_plot('accuracy')
    
        Create a bar plot with custom error bars:
    
        >>> fig, ax = analyzer.create_metric_comparison_plot(
        ...     'f1_score', 
        ...     plot_type='barplot',
        ...     error_bar_type='sem',
        ...     bar_color=['red', 'blue', 'green']
        ... )
    
        Violin plot with custom significance pairs:
    
        >>> fig, ax = analyzer.create_metric_comparison_plot(
        ...     'precision',
        ...     plot_type='violin',
        ...     show_significance_pairs=[('Method1', 'Method2'), ('Method1', 'Method3')],
        ...     significance_fontsize=16
        ... )
    
        Publication-ready figure with custom styling:
    
        >>> fig, ax = analyzer.create_metric_comparison_plot(
        ...     'recall',
        ...     plot_type='boxplot',
        ...     figsize=(8, 6),
        ...     dpi=300,
        ...     title_fontsize=14,
        ...     title_fontweight='bold',
        ...     palette='viridis',
        ...     grid_alpha=0.2,
        ...     spines_visible={'top': False, 'right': False, 'left': True, 'bottom': True}
        ... )
    
        See Also
        --------
        perform_metric_analysis : Underlying statistical analysis function
        analyze_all_metrics : Batch analysis of multiple metrics
        create_publication_figure : Multi-panel publication figures
        """

        _apply_rea_style()

        # 设置默认的边框可见性
        if spines_visible is None:
            spines_visible = {'top': False, 'right': False, 'left': True, 'bottom': True}
    
        # 设置字体
        if font_family:
            _apply_font(font_family)
    
        # 决定是否需要创建新图形
        create_new_figure = (ax is None)
    
        if create_new_figure:
            # 创建新图形
            fig = plt.figure(figsize=figsize, dpi=dpi)
            region = bind_figure_region(fig, (0.12, 0.12, 0.95, 0.92))
            ax = region.add_axes(fig)
        else:
            # 使用传入的轴对象
            fig = ax.figure
    
        # 调用核心绘图逻辑
        self._plot_metric_on_axis(
            ax, metric_name, plot_type=plot_type,
            show_significance_pairs=show_significance_pairs,
            max_significance_pairs=max_significance_pairs,
            datasets_to_show=datasets_to_show,
            display_name=display_name,
            stat_test_display=stat_test_display,
            title_fontsize=title_fontsize,
            title_fontweight=title_fontweight,
            axis_label_fontsize=axis_label_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            significance_fontsize=significance_fontsize,
            ns_fontsize=ns_fontsize,
            legend_fontsize=legend_fontsize,
            show_legend=show_legend,  
            palette=palette,
            box_color=box_color,
            strip_color=strip_color,
            strip_alpha=strip_alpha,
            line_color=line_color,
            mean_line_color=mean_line_color,
            # 传递新的柱状图参数
            bar_color=bar_color,
            bar_edge_color=bar_edge_color,
            bar_edge_width=bar_edge_width,
            bar_alpha=bar_alpha,
            bar_width=bar_width,
            show_error_bars=show_error_bars,
            error_bar_type=error_bar_type,
            error_bar_capsize=error_bar_capsize,
            error_bar_capthick=error_bar_capthick,
            error_bar_color=error_bar_color,
            error_bar_alpha=error_bar_alpha,
            error_bar_width=error_bar_width,
            show_bar_points=show_bar_points,
            bar_strip_size=bar_strip_size,
            bar_strip_alpha=bar_strip_alpha,
            bar_strip_color=bar_strip_color,
            bar_strip_jitter_width=bar_strip_jitter_width,
            significance_marker_map=significance_marker_map,
            significance_y_offset=significance_y_offset,
            significance_marker_offset=significance_marker_offset,
            ns_offset=ns_offset,
            significance_line_width=significance_line_width,
            box_width=box_width,
            jitter_width=jitter_width,
            strip_size=strip_size,
            flier_size=flier_size,
            flier_marker=flier_marker,
            flier_alpha=flier_alpha,
            mean_line_width=mean_line_width,
            mean_marker_size=mean_marker_size,
            title_bbox_props=title_bbox_props,
            legend_bbox_props=legend_bbox_props,
            annotation_bbox_props=annotation_bbox_props,
            grid=grid,
            grid_alpha=grid_alpha,
            rotate_xlabels=rotate_xlabels,
            xlabel_rotation=xlabel_rotation,
            xlabel_ha=xlabel_ha,
            spines_visible=spines_visible
        )
    
        # 只在创建新图形时应用布局调整
        if create_new_figure:
            # Layout is fixed via bind_figure_region; apply caller overrides if given
            if subplot_adjust_params is not None:
                fig.subplots_adjust(**subplot_adjust_params)
    
        return fig, ax
    
    
    def _plot_metric_on_axis(self, ax, metric_name, **kwargs):
        """
        在指定轴上绘制指标比较图的核心逻辑
        """
        # 获取或计算统计分析结果
        if metric_name not in self.statistical_results:
            analysis_results = self.perform_metric_analysis(metric_name)
            self.statistical_results[metric_name] = analysis_results
        else:
            analysis_results = self.statistical_results[metric_name]
    
        if 'error' in analysis_results:
            ax.text(0.5, 0.5, f"Error: {analysis_results['error']}", 
                    transform=ax.transAxes, ha='center', va='center', color='red')
            return
    
        # 准备绘图数据
        plot_data = self.processed_data[self.processed_data['metric'] == metric_name].copy()
    
        if plot_data.empty:
            ax.text(0.5, 0.5, "No data available", 
                    transform=ax.transAxes, ha='center', va='center')
            return
    
        plot_type = kwargs.get('plot_type', 'boxplot')
        palette = kwargs.get('palette', 'Set2')
        box_color = kwargs.get('box_color')
        box_width = kwargs.get('box_width', 0.6)
        jitter_width = kwargs.get('jitter_width', True)
        strip_color = kwargs.get('strip_color', 'black')
        strip_alpha = kwargs.get('strip_alpha', 0.6)
        strip_size = kwargs.get('strip_size', 4)
        flier_size = kwargs.get('flier_size', 4)
        flier_marker = kwargs.get('flier_marker', 'o')
        flier_alpha = kwargs.get('flier_alpha', 0.7)
        show_legend = kwargs.get('show_legend', True)
        
        # 根据图表类型绘制
        if plot_type == 'boxplot':
            # 处理颜色
            if box_color is not None:
                box_palette = [box_color] * len(self.method_names)
            else:
                box_palette = palette
    
            # 绘制箱线图（不显示异常值）
            sns.boxplot(data=plot_data, x='method', y='value', ax=ax, 
                        palette=box_palette, width=box_width, order=self.method_names, 
                        fliersize=flier_size, flierprops={'marker': flier_marker, 'alpha': flier_alpha})
    
            # 添加散点图
            sns.stripplot(data=plot_data, x='method', y='value', ax=ax,
                            color=strip_color, alpha=strip_alpha, size=strip_size, 
                            jitter=jitter_width, order=self.method_names)
    
        elif plot_type == 'violin':
            sns.violinplot(data=plot_data, x='method', y='value', ax=ax,
                            palette=palette, inner='box', order=self.method_names)
            sns.stripplot(data=plot_data, x='method', y='value', ax=ax,
                            color=strip_color, alpha=strip_alpha*0.8, size=strip_size*0.75, 
                            jitter=jitter_width, order=self.method_names)
    
        elif plot_type == 'strip':
            sns.stripplot(data=plot_data, x='method', y='value', ax=ax,
                            palette=palette, size=strip_size*1.5, jitter=jitter_width, 
                            alpha=strip_alpha*1.2, order=self.method_names)
    
        # 新增：柱状图支持
        elif plot_type == 'barplot':
            self._create_customizable_bar_plot(
                ax, plot_data,
                bar_color=kwargs.get('bar_color'),
                bar_edge_color=kwargs.get('bar_edge_color', 'black'),
                bar_edge_width=kwargs.get('bar_edge_width', 1),
                bar_alpha=kwargs.get('bar_alpha', 0.7),
                bar_width=kwargs.get('bar_width', 0.6),
                palette=palette,
                show_error_bars=kwargs.get('show_error_bars', True),
                error_bar_type=kwargs.get('error_bar_type', 'std'),
                error_bar_capsize=kwargs.get('error_bar_capsize', 5),
                error_bar_capthick=kwargs.get('error_bar_capthick', 2),
                error_bar_color=kwargs.get('error_bar_color', 'black'),
                error_bar_alpha=kwargs.get('error_bar_alpha', 0.8),
                error_bar_width=kwargs.get('error_bar_width', 2),
                show_bar_points=kwargs.get('show_bar_points', True),
                bar_strip_size=kwargs.get('bar_strip_size', 3),
                bar_strip_alpha=kwargs.get('bar_strip_alpha', 0.6),
                bar_strip_color=kwargs.get('bar_strip_color', 'black'),
                bar_strip_jitter_width=kwargs.get('bar_strip_jitter_width', 0.4)
            )
    
        elif plot_type == 'paired_lines':
            self._create_customizable_paired_lines_plot(
                ax, plot_data, 
                datasets_to_show=kwargs.get('datasets_to_show', 20),
                line_color=kwargs.get('line_color', 'gray'), 
                mean_line_color=kwargs.get('mean_line_color', 'red'),
                mean_line_width=kwargs.get('mean_line_width', 3), 
                mean_marker_size=kwargs.get('mean_marker_size', 8),
                annotation_bbox_props=kwargs.get('annotation_bbox_props'),
                annotation_fontsize=kwargs.get('legend_fontsize', 9),
                show_legend=show_legend  
            )
    
        # 添加统计显著性标注
        self._add_customizable_significance_annotations(
            ax, analysis_results, plot_data,
            show_significance_pairs=kwargs.get('show_significance_pairs'),
            max_significance_pairs=kwargs.get('max_significance_pairs', 5),
            significance_marker_map=kwargs.get('significance_marker_map'),
            significance_y_offset=kwargs.get('significance_y_offset', 0.05),
            significance_marker_offset=kwargs.get('significance_marker_offset', 0.02),
            ns_offset=kwargs.get('ns_offset', 0.01),
            significance_line_width=kwargs.get('significance_line_width', 1),
            significance_fontsize=kwargs.get('significance_fontsize', 12),
            ns_fontsize=kwargs.get('ns_fontsize', 10),
            line_color=kwargs.get('line_color', 'black'),
            legend_bbox_props=kwargs.get('legend_bbox_props'),
            legend_fontsize=kwargs.get('legend_fontsize', 9),
            show_legend=show_legend)
    
        # 设置图形属性
        test_info = analysis_results.get('main_test', {})
        test_name = test_info.get('test_name', 'Unknown test')
        p_value = test_info.get('p_value', 1.0)
    
        if p_value < 0.001:
            sig_desc = "p < 0.001"
        elif p_value < 0.01:
            sig_desc = f"p = {p_value:.3f}"
        elif p_value < 0.05:
            sig_desc = f"p = {p_value:.3f}"
        else:
            sig_desc = f"p = {p_value:.3f}"
    
        # 设置标题 - 修正：使用正确的参数名
        display_name_param = kwargs.get('display_name') or kwargs.get('display_metric_name')
        if display_name_param:
            display_name = display_name_param
        else:
            display_name = metric_name
            
        stat_display_mode = kwargs.get('stat_test_display', 'full')
    
        if stat_display_mode == 'none':
            title_text = display_name
        elif stat_display_mode == 'short':
            short_test_names = {
                'Repeated measures ANOVA': 'ANOVA', 'One-way ANOVA': 'ANOVA',
                'Welch ANOVA': 'Welch-ANOVA', 'Kruskal-Wallis': 'K-W',
                'Paired t-test': 'Paired t', 'Independent t-test': 't-test',
                'Wilcoxon signed-rank test': 'Wilcoxon', 'Mann-Whitney U': 'Mann-Whitney',
                'Friedman test': 'Friedman'
            }
            short_name = short_test_names.get(test_name, test_name)
            title_text = f'{display_name}\n({short_name}, {sig_desc})'
        else:
            title_text = f'{display_name}\n({test_name}, {sig_desc})'
    
        title_bbox_props = kwargs.get('title_bbox_props')
        title_fontsize = kwargs.get('title_fontsize', 12)
        title_fontweight = kwargs.get('title_fontweight', 'normal')
    
        title_pad = kwargs.get('title_pad', 8)
        if title_bbox_props is not None:
            ax.set_title(title_text, fontsize=title_fontsize, fontweight=title_fontweight, 
                        pad=title_pad, bbox=title_bbox_props)
        else:
            ax.set_title(title_text, fontsize=title_fontsize, fontweight=title_fontweight, pad=title_pad)
    
        # 设置轴标签
        axis_label_fontsize = kwargs.get('axis_label_fontsize', 12)
        show_xlabel = kwargs.get('show_xlabel', False)
        if show_xlabel:
            ax.set_xlabel('Method', fontsize=axis_label_fontsize)
        else:
            ax.set_xlabel('', fontsize=axis_label_fontsize)
        ax.set_ylabel(f'{metric_name}', fontsize=axis_label_fontsize)
    
        # 设置刻度标签字体大小
        tick_label_fontsize = kwargs.get('tick_label_fontsize', 12)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    
        # Auto scientific notation for large y-axis values (e.g., CAL metric)
        y_min, y_max = ax.get_ylim()
        if abs(y_max) > 1000 or abs(y_min) > 1000:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
            if hasattr(ax, 'yaxis') and ax.yaxis.get_offset_text():
                ax.yaxis.get_offset_text().set_fontsize(tick_label_fontsize - 1)
    
        # 设置边框可见性
        spines_visible = kwargs.get('spines_visible', {'top': False, 'right': False})
        for spine, visible in spines_visible.items():
            if spine in ax.spines:
                ax.spines[spine].set_visible(visible)
    
        # 网格设置
        if kwargs.get('grid', True):
            ax.grid(True, alpha=kwargs.get('grid_alpha', 0.3))
    
        # X轴标签旋转 — adaptive rotation based on method count
        rotate_xlabels = kwargs.get('rotate_xlabels')
        n_methods = len(self.method_names)
        if rotate_xlabels is None:
            rotate_xlabels = n_methods > 3
    
        if rotate_xlabels:
            # Adaptive rotation: fewer methods → gentler angle
            user_rotation = kwargs.get('xlabel_rotation')
            if user_rotation is not None:
                xlabel_rotation = user_rotation
            elif n_methods <= 5:
                xlabel_rotation = 30
            elif n_methods <= 8:
                xlabel_rotation = 40
            else:
                xlabel_rotation = 50
            xlabel_ha = kwargs.get('xlabel_ha', 'right')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=xlabel_rotation, ha=xlabel_ha)
            # Slight rightward shift for rotated labels so they don't crowd left
            ax.tick_params(axis='x', pad=3)
    
    
    # 新增：柱状图创建函数
    def _create_customizable_bar_plot(self, ax, plot_data,
                                     bar_color=None,
                                     bar_edge_color='black',
                                     bar_edge_width=1,
                                     bar_alpha=0.7,
                                     bar_width=0.6,
                                     palette='Set2',
                                     show_error_bars=True,
                                     error_bar_type='std',
                                     error_bar_capsize=5,
                                     error_bar_capthick=2,
                                     error_bar_color='black',
                                     error_bar_alpha=0.8,
                                     error_bar_width=2,
                                     show_bar_points=True,
                                     bar_strip_size=3,
                                     bar_strip_alpha=0.6,
                                     bar_strip_color='black',
                                     bar_strip_jitter_width=0.4):
        """
        创建可定制的柱状图
        
        参数说明：
        - error_bar_type: 'std'(标准差), 'sem'(标准误), 'ci95'(95%置信区间), 'ci99'(99%置信区间), 'none'(无误差棒)
        - bar_strip_jitter_width: 散点在x轴方向的抖动宽度，0-0.5之间
        """
        import numpy as np
        from scipy import stats
        
        # 计算每个方法的统计量
        x_positions = []
        heights = []
        error_values = []
        bar_colors = []
        
        # 处理颜色
        if bar_color is not None:
            if isinstance(bar_color, (list, tuple)):
                colors_to_use = bar_color
            else:
                colors_to_use = [bar_color] * len(self.method_names)
        else:
            # 使用调色板
            if isinstance(palette, str):
                colors_to_use = sns.color_palette(palette, len(self.method_names))
            else:
                colors_to_use = palette
        
        for i, method in enumerate(self.method_names):
            method_data = plot_data[plot_data['method'] == method]['value']
            
            if len(method_data) == 0:
                continue
                
            x_positions.append(i)
            mean_val = method_data.mean()
            heights.append(mean_val)
            
            # 计算误差棒
            if show_error_bars and error_bar_type != 'none':
                if error_bar_type == 'std':
                    error_val = method_data.std()
                elif error_bar_type == 'sem':
                    error_val = method_data.std() / np.sqrt(len(method_data))
                elif error_bar_type == 'ci95':
                    confidence_level = 0.95
                    degrees_freedom = len(method_data) - 1
                    if degrees_freedom > 0:
                        t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
                        margin_of_error = t_value * (method_data.std() / np.sqrt(len(method_data)))
                        error_val = margin_of_error
                    else:
                        error_val = 0
                elif error_bar_type == 'ci99':
                    confidence_level = 0.99
                    degrees_freedom = len(method_data) - 1
                    if degrees_freedom > 0:
                        t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
                        margin_of_error = t_value * (method_data.std() / np.sqrt(len(method_data)))
                        error_val = margin_of_error
                    else:
                        error_val = 0
                else:
                    error_val = 0
            else:
                error_val = 0
                
            error_values.append(error_val)
            bar_colors.append(colors_to_use[i % len(colors_to_use)])
        
        # 绘制柱状图
        bars = ax.bar(x_positions, heights, width=bar_width, 
                      color=bar_colors, alpha=bar_alpha,
                      edgecolor=bar_edge_color, linewidth=bar_edge_width)
        
        # 添加误差棒
        if show_error_bars and error_bar_type != 'none':
            ax.errorbar(x_positions, heights, yerr=error_values,
                       fmt='none', ecolor=error_bar_color, alpha=error_bar_alpha,
                       capsize=error_bar_capsize, capthick=error_bar_capthick,
                       elinewidth=error_bar_width, zorder=10)
        
        # 添加散点（抖动）
        if show_bar_points:
            for i, method in enumerate(self.method_names):
                method_data = plot_data[plot_data['method'] == method]['value']
                
                if len(method_data) == 0:
                    continue
                
                # 生成抖动的x坐标
                x_center = i
                jitter_range = bar_strip_jitter_width * bar_width
                x_jittered = np.random.normal(x_center, jitter_range/3, len(method_data))
                # 限制抖动范围
                x_jittered = np.clip(x_jittered, 
                                   x_center - jitter_range/2, 
                                   x_center + jitter_range/2)
                
                ax.scatter(x_jittered, method_data.values, 
                          color=bar_strip_color, alpha=bar_strip_alpha, 
                          s=bar_strip_size**2, zorder=15)
        
        # 设置x轴刻度
        ax.set_xticks(range(len(self.method_names)))
        ax.set_xticklabels(self.method_names)
    
    
    def _create_customizable_paired_lines_plot(self, ax, plot_data,
                                                datasets_to_show=20,
                                                line_color='gray', 
                                                mean_line_color='red',
                                                mean_line_width=3,
                                                mean_marker_size=8,
                                                annotation_bbox_props=None,
                                                annotation_fontsize=9,
                                                show_legend=True):
        """
        创建可定制的配对线图
        """
        if self.design_type != 'paired':
            self._log("Warning: paired_lines plot is most suitable for paired design")
    
        # 为每个数据集绘制连接线
        datasets_plotted = 0
        max_datasets_to_show = min(datasets_to_show, self.n_datasets)
    
        for dataset_id in plot_data['dataset_id'].unique():
            if datasets_plotted >= max_datasets_to_show:
                break
    
            dataset_data = plot_data[plot_data['dataset_id'] == dataset_id]
    
            if len(dataset_data) == len(self.method_names):
                x_positions = []
                y_values = []
    
                for i, method in enumerate(self.method_names):
                    method_data = dataset_data[dataset_data['method'] == method]
                    if not method_data.empty:
                        x_positions.append(i)
                        y_values.append(method_data['value'].iloc[0])
    
                if len(x_positions) == len(self.method_names):
                    alpha = max(0.2, min(0.8, 1.0 - datasets_plotted * 0.02))
                    ax.plot(x_positions, y_values, 'o-', alpha=alpha, 
                            linewidth=1, markersize=3, color=line_color)
                    datasets_plotted += 1
    
        # 添加均值线
        means = []
        stds = []
        for method in self.method_names:
            method_data = plot_data[plot_data['method'] == method]['value']
            if len(method_data) > 0:
                means.append(method_data.mean())
                stds.append(method_data.std())
            else:
                means.append(np.nan)
                stds.append(np.nan)
    
        valid_indices = [i for i, mean in enumerate(means) if not np.isnan(mean)]
        if valid_indices:
            valid_means = [means[i] for i in valid_indices]
            valid_stds = [stds[i] for i in valid_indices]
    
            ax.plot(valid_indices, valid_means, 'o-', color=mean_line_color,
                    linewidth=mean_line_width, markersize=mean_marker_size, 
                    label='Mean', alpha=0.9, zorder=10)
    
            ax.errorbar(valid_indices, valid_means, yerr=valid_stds, 
                        fmt='none', ecolor=mean_line_color, alpha=0.6, 
                        capsize=5, zorder=9)
    
        ax.set_xticks(range(len(self.method_names)))
        ax.set_xticklabels(self.method_names)
    
        if show_legend:
            ax.legend()
    
        if datasets_plotted < self.n_datasets:
            bbox_props = annotation_bbox_props if annotation_bbox_props else \
                        {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.7}
    
            ax.text(0.02, 0.98, f'Showing {datasets_plotted}/{self.n_datasets} datasets', 
                    transform=ax.transAxes, fontsize=annotation_fontsize, va='top',
                    bbox=bbox_props)
    
    
    def _add_customizable_significance_annotations(self, ax, analysis_results, plot_data,
                                                    show_significance_pairs=None,
                                                    max_significance_pairs=5,
                                                    significance_marker_map=None,
                                                    significance_y_offset=0.05,
                                                    significance_marker_offset=0.02,
                                                    ns_offset=0.01,
                                                    significance_line_width=1,
                                                    significance_fontsize=12,
                                                    ns_fontsize=10,
                                                    line_color='black',
                                                    legend_bbox_props=None,
                                                    legend_fontsize=9,
                                                    show_legend=True): 
        """
        添加可定制的统计显著性标注
        """
        if significance_marker_map is None:
            significance_marker_map = {'***': '***', '**': '**', '*': '*', 'ns': 'ns'}
    
        if analysis_results['analysis_type'] == 'two_groups':
            p_value = analysis_results['main_test']['p_value']
            y_max = plot_data['value'].max()
            y_range = plot_data['value'].max() - plot_data['value'].min()
            y_line = y_max + significance_y_offset * y_range
    
            ax.plot([0, 1], [y_line, y_line], color=line_color, linewidth=significance_line_width)
            ax.plot([0, 0], [y_line-0.01*y_range, y_line+0.01*y_range], color=line_color, linewidth=significance_line_width)
            ax.plot([1, 1], [y_line-0.01*y_range, y_line+0.01*y_range], color=line_color, linewidth=significance_line_width)
            
            if p_value < 0.001: sig_text = significance_marker_map.get('***', '***')
            elif p_value < 0.01: sig_text = significance_marker_map.get('**', '**')
            elif p_value < 0.05: sig_text = significance_marker_map.get('*', '*')
            else: sig_text = significance_marker_map.get('ns', 'ns')
            
            if sig_text == 'ns': 
                ax.text(0.5, y_line+ns_offset*y_range, sig_text, ha='center', va='bottom', fontsize=ns_fontsize)
            else: 
                ax.text(0.5, y_line+significance_marker_offset*y_range, sig_text, ha='center', va='bottom', fontsize=significance_fontsize, fontweight='bold')
    
        elif (analysis_results['analysis_type'] == 'multiple_groups' and 'post_hoc' in analysis_results):
            post_hoc = analysis_results['post_hoc']
            significant_comparisons = [comp for comp in post_hoc['comparisons']]
    
            if not significant_comparisons: 
                return
    
            if show_significance_pairs is not None:
                pairs_to_show = [comp for comp in significant_comparisons 
                               for m1, m2 in show_significance_pairs 
                               if (comp['method1'] == m1 and comp['method2'] == m2) or 
                                  (comp['method1'] == m2 and comp['method2'] == m1)]
                # Sort by significance and respect the max cap even for user-specified pairs
                pairs_to_show.sort(key=lambda x: x.get('p_corrected', x.get('p_raw', 1)))
                pairs_to_show = pairs_to_show[:max_significance_pairs]
            else:
                significant_comparisons.sort(key=lambda x: x.get('p_corrected', x.get('p_raw', 1)))
                pairs_to_show = significant_comparisons[:max_significance_pairs]
    
            if not pairs_to_show: 
                return
    
            y_max = plot_data['value'].max()
            y_min = plot_data['value'].min()
            y_range = y_max - y_min

            # Adaptive per-bracket increment: more brackets → tighter stacking
            n_brackets = len(pairs_to_show)
            if n_brackets <= 3:
                bracket_step = 0.06
            elif n_brackets <= 5:
                bracket_step = 0.05
            else:
                bracket_step = 0.04
    
            for i, comp in enumerate(pairs_to_show):
                try:
                    method1, method2 = comp['method1'], comp['method2']
                    idx1, idx2 = self.method_names.index(method1), self.method_names.index(method2)
                    if idx1 > idx2: 
                        idx1, idx2 = idx2, idx1
    
                    y_offset = significance_y_offset + bracket_step * i
                    y_line = y_max + y_offset * y_range
    
                    ax.plot([idx1, idx2], [y_line, y_line], color=line_color, linewidth=significance_line_width)
                    ax.plot([idx1, idx1], [y_line-0.01*y_range, y_line+0.01*y_range], color=line_color, linewidth=significance_line_width)
                    ax.plot([idx2, idx2], [y_line-0.01*y_range, y_line+0.01*y_range], color=line_color, linewidth=significance_line_width)
    
                    p_val = comp.get('p_corrected', comp.get('p_raw', 1))
                    if p_val < 0.001: 
                        sig_marker = significance_marker_map.get('***', '***')
                    elif p_val < 0.01: 
                        sig_marker = significance_marker_map.get('**', '**')
                    elif p_val < 0.05: 
                        sig_marker = significance_marker_map.get('*', '*')
                    else: 
                        sig_marker = 'ns'
                        
                    if sig_marker == 'ns': 
                        ax.text((idx1+idx2)/2, y_line+ns_offset*y_range, sig_marker, ha='center', va='bottom', fontsize=ns_fontsize)
                    else: 
                        ax.text((idx1+idx2)/2, y_line+significance_marker_offset*y_range, sig_marker, ha='center', va='bottom', fontsize=significance_fontsize, fontweight='bold')
                    
                except (ValueError, IndexError) as e:
                    self._log(f"Warning: Could not annotate pair {comp['method1']} vs {comp['method2']}: {e}")
                    continue
    
            # 根据参数决定是否显示图例说明
            if show_legend and len(pairs_to_show) > 0:
                legend_text = f"Significant pairs shown: {len(pairs_to_show)}/{len(significant_comparisons)}"
                if len(significant_comparisons) > max_significance_pairs and show_significance_pairs is None:
                    legend_text += f" (top {max_significance_pairs} by p-value)"
    
                bbox_props = legend_bbox_props if legend_bbox_props else {'boxstyle': 'round', 'facecolor': 'lightblue', 'alpha': 0.7}
    
                ax.text(0.98, 0.02, legend_text, transform=ax.transAxes, fontsize=legend_fontsize, ha='right', va='bottom', bbox=bbox_props)


    # ==================== 4. 综合结果分析和展示 ====================
    
    def analyze_all_metrics(self):
        """
        分析所有指标并返回结果汇总
        
        返回:
            pandas.DataFrame: 所有指标的分析结果汇总
        """
        self._log("🔄 Analyzing all metrics...")
        
        all_results = []
        
        for metric in self.metrics:
            try:
                analysis = self.perform_metric_analysis(metric)
                
                if 'error' in analysis:
                    self._log(f"❌ Error analyzing {metric}: {analysis['error']}")
                    continue
                
                self.statistical_results[metric] = analysis
                
                # 构建结果行
                result_row = {'Metric': metric}
                
                # 描述性统计
                for method in self.method_names:
                    if method in analysis['descriptive_stats']:
                        stats_data = analysis['descriptive_stats'][method]
                        result_row[f'{method}_Mean'] = stats_data['mean']
                        result_row[f'{method}_Std'] = stats_data['std']
                        result_row[f'{method}_N'] = stats_data['n']
                    else:
                        result_row[f'{method}_Mean'] = np.nan
                        result_row[f'{method}_Std'] = np.nan
                        result_row[f'{method}_N'] = 0
                
                # 主要统计测试
                main_test = analysis.get('main_test', {})
                result_row['Test_Used'] = main_test.get('test_name', 'Unknown')
                result_row['P_Value'] = main_test.get('p_value', np.nan)
                result_row['Test_Statistic'] = main_test.get('statistic', np.nan)
                
                # 效应量
                effect_size = analysis.get('effect_size', {})
                result_row['Effect_Size'] = effect_size.get('value', np.nan)
                result_row['Effect_Type'] = effect_size.get('type', 'Unknown')
                
                # 两组比较的改进信息
                if analysis['analysis_type'] == 'two_groups':
                    improvement = analysis.get('improvement', {})
                    result_row['Improvement_Pct'] = improvement.get('percentage', np.nan)
                    result_row['Improvement_Abs'] = improvement.get('absolute', np.nan)
                
                # 显著性标记
                p_val = result_row['P_Value']
                if pd.isna(p_val):
                    result_row['Significance'] = 'Unknown'
                elif p_val < 0.001:
                    result_row['Significance'] = '***'
                elif p_val < 0.01:
                    result_row['Significance'] = '**'
                elif p_val < 0.05:
                    result_row['Significance'] = '*'
                else:
                    result_row['Significance'] = 'ns'
                
                # 事后检验信息
                if 'post_hoc' in analysis:
                    significant_pairs = sum(1 for comp in analysis['post_hoc']['comparisons'] 
                                          if comp['significant'])
                    result_row['Significant_Pairs'] = significant_pairs
                    result_row['Total_Pairs'] = len(analysis['post_hoc']['comparisons'])
                else:
                    result_row['Significant_Pairs'] = np.nan
                    result_row['Total_Pairs'] = np.nan
                
                all_results.append(result_row)
                
            except Exception as e:
                self._log(f"❌ Error analyzing {metric}: {e}")
                continue
        
        results_df = pd.DataFrame(all_results)
        self._log(f"✅ Completed analysis of {len(all_results)} metrics")
        
        return results_df
    
    def print_comprehensive_summary(self):
        """
        打印全面的分析摘要，包含所有统计结果
        """
        print("="*120)
        print("                           📊 RIGOROUS EXPERIMENTAL ANALYSIS REPORT")
        print("="*120)
        
        # 实验设计摘要
        print(f"\n📋 Experimental Design Summary:")
        print(f"   • Methods: {len(self.method_names)} - {', '.join(self.method_names)}")
        print(f"   • Datasets: {self.n_datasets}")
        print(f"   • Metrics: {len(self.metrics)}")
        print(f"   • Design type: {self.design_type.title()}")
        
        # 分析所有指标
        summary_df = self.analyze_all_metrics()
        
        if summary_df.empty:
            print("\n❌ No valid analysis results available")
            return summary_df
        
        # 总体性能摘要
        print(f"\n📈 Overall Performance Summary:")
        
        if len(self.method_names) == 2:
            # 两组比较摘要
            method1, method2 = self.method_names[0], self.method_names[1]
            
            if 'Improvement_Pct' in summary_df.columns:
                significant_improvements = len(summary_df[
                    (summary_df['P_Value'] < 0.05) & (summary_df['Improvement_Pct'] > 0)
                ])
                significant_degradations = len(summary_df[
                    (summary_df['P_Value'] < 0.05) & (summary_df['Improvement_Pct'] < 0)
                ])
                
                print(f"   • Significant improvements ({method2} > {method1}): {significant_improvements}/{len(summary_df)}")
                print(f"   • Significant degradations ({method2} < {method1}): {significant_degradations}/{len(summary_df)}")
                
                # 最大改进
                top_improvements = summary_df.nlargest(5, 'Improvement_Pct')
                print(f"\n🏆 Top 5 Performance Improvements ({method2} vs {method1}):")
                for _, row in top_improvements.iterrows():
                    print(f"   • {row['Metric']}: {row['Improvement_Pct']:+.1f}% ({row['Significance']})")
        else:
            # 多组比较摘要
            significant_overall = len(summary_df[summary_df['P_Value'] < 0.05])
            print(f"   • Significant overall differences: {significant_overall}/{len(summary_df)} metrics")
            
            if 'Significant_Pairs' in summary_df.columns:
                total_significant_pairs = summary_df['Significant_Pairs'].sum()
                total_possible_pairs = summary_df['Total_Pairs'].sum()
                print(f"   • Significant pairwise differences: {total_significant_pairs:.0f}/{total_possible_pairs:.0f} pairs")
        
        # 统计测试摘要
        print(f"\n🔬 Statistical Test Distribution:")
        test_counts = summary_df['Test_Used'].value_counts()
        for test_name, count in test_counts.items():
            print(f"   • {test_name}: {count} metrics")
        
        # 效应量分布
        if 'Effect_Size' in summary_df.columns:
            effect_sizes = summary_df['Effect_Size'].dropna()
            if len(effect_sizes) > 0:
                print(f"\n📏 Effect Size Distribution:")
                print(f"   • Mean effect size: {effect_sizes.mean():.3f} (±{effect_sizes.std():.3f})")
                
                # 效应量分类
                large_effects = len(effect_sizes[abs(effect_sizes) > 0.8])
                medium_effects = len(effect_sizes[(abs(effect_sizes) > 0.5) & (abs(effect_sizes) <= 0.8)])
                small_effects = len(effect_sizes[(abs(effect_sizes) > 0.2) & (abs(effect_sizes) <= 0.5)])
                negligible_effects = len(effect_sizes[abs(effect_sizes) <= 0.2])
                
                print(f"   • Large effects (|ES| > 0.8): {large_effects}")
                print(f"   • Medium effects (0.5 < |ES| ≤ 0.8): {medium_effects}")
                print(f"   • Small effects (0.2 < |ES| ≤ 0.5): {small_effects}")
                print(f"   • Negligible effects (|ES| ≤ 0.2): {negligible_effects}")
        
        # 详细结果表格
        print(f"\n📊 Detailed Statistical Results:")
        print("="*120)
        
        # 选择要显示的列
        display_cols = ['Metric']
        for method in self.method_names:
            if f'{method}_Mean' in summary_df.columns:
                display_cols.append(f'{method}_Mean')
        display_cols.extend(['Test_Used', 'P_Value', 'Effect_Size', 'Significance'])
        
        if 'Improvement_Pct' in summary_df.columns:
            display_cols.append('Improvement_Pct')
        
        display_df = summary_df[display_cols].copy()
        
        # 格式化数值
        for col in display_df.columns:
            if col.endswith('_Mean') or col in ['P_Value', 'Effect_Size', 'Improvement_Pct']:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
        
        print(display_df.to_string(index=False, max_colwidth=20))
        
        # 事后检验详细结果（对于多组比较）
        if len(self.method_names) > 2:
            print(f"\n🔍 Post-hoc Analysis Details:")
            print("-"*100)
            
            for metric, analysis in self.statistical_results.items():
                if 'post_hoc' in analysis and analysis['main_test']['p_value'] < 0.05:
                    print(f"\n{metric} ({analysis['post_hoc']['method']}):")
                    
                    for comp in analysis['post_hoc']['comparisons']:
                        if comp['significant']:
                            print(f"   • {comp['method1']} vs {comp['method2']}: "
                                  f"p = {comp['p_corrected']:.4f} *** "
                                  f"(effect size = {comp['effect_size']:.3f})")
        
        print("="*120)
        
        return summary_df


# ==================== 使用示例 ====================

def create_publication_figure(analyzer, metrics, figsize=(16, 12), dpi=300, 
                            ncols=2, metric_display_names=None,
                            shared_y_axis=False, suptitle=None,
                            # 精确的子图布局控制
                            subplot_adjust_params=None,
                            # Slightly increased left margin to avoid y-label /
                            # y-tick overlap with the data region. Right margin
                            # is tightened a little to preserve total width.
                            left=0.10, right=0.96, top=0.92, bottom=0.12,
                            hspace=0.25, wspace=0.20,
                            # 面板标签样式
                            panel_labels=True, panel_label_start='A',
                            custom_panel_labels=None, panel_label_offset=(0, 0),
                            panel_label_fontsize=14, panel_label_fontweight='bold',
                            panel_label_position=(-0.15, 1.05),
                            **visual_params):
    """
    Create publication-ready multi-panel statistical comparison figure.

    Generates a professionally formatted multi-panel figure with statistical 
    analysis plots for multiple metrics. Each panel displays a separate metric
    with consistent styling, automatic panel labeling, and precise layout control
    suitable for scientific publications.

    Parameters
    ----------
    analyzer : RigorousExperimentalAnalyzer
        Initialized analyzer object containing loaded experimental data and 
        computed statistical results.
    metrics : list of str
        List of metric names to include in the multi-panel figure. Each metric
        will be displayed in a separate subplot panel.
    figsize : tuple of float, default=(16, 12)
        Overall figure size in inches as (width, height). Should be scaled 
        appropriately for the number of panels and desired resolution.
    dpi : int, default=300
        Dots per inch resolution for the figure. Use 300+ for publication quality,
        150 for high-quality screen display, 72 for draft/preview.
    ncols : int, default=2
        Number of columns in the subplot grid. Total rows are calculated 
        automatically based on len(metrics) and ncols.
    metric_display_names : dict, optional
        Custom display names for metrics. Keys should be metric names from 
        the metrics list, values are the corresponding display names.
        Format: {'metric1': 'Custom Display Name 1', 'metric2': 'Name 2'}.
        If None, uses original metric names.
    shared_y_axis : bool, default=False
        Whether to share y-axis scaling across all panels. When True, all panels
        will have the same y-axis range, facilitating cross-panel comparisons.
        Only recommended when metrics have similar scales.
    suptitle : str, optional
        Overall figure title displayed above all panels. If None, no main title
        is added.

    Layout Control Parameters
    ------------------------
    subplot_adjust_params : dict, optional
        Complete override of subplot spacing parameters. If provided, other 
        individual spacing parameters (left, right, etc.) are ignored.
        Format: {'left': float, 'right': float, 'top': float, 'bottom': float,
                 'hspace': float, 'wspace': float}.
    left : float, default=0.08
        Left margin of the subplot area as fraction of figure width (0.0 to 1.0).
        Increase to accommodate longer y-axis labels.
    right : float, default=0.95
        Right margin of the subplot area as fraction of figure width (0.0 to 1.0).
        Decrease to add right-side annotations or wider margins.
    top : float, default=0.92
        Top margin of the subplot area as fraction of figure height (0.0 to 1.0).
        Decrease to accommodate suptitle or top annotations.
    bottom : float, default=0.08
        Bottom margin of the subplot area as fraction of figure height (0.0 to 1.0).
        Increase for longer x-axis labels or bottom annotations.
    hspace : float, default=0.25
        Vertical spacing between subplot rows as fraction of average subplot height.
        Increase to prevent overlap of titles and significance annotations.
    wspace : float, default=0.20
        Horizontal spacing between subplot columns as fraction of average subplot width.
        Increase to prevent overlap of y-axis labels and annotations.

    Panel Label Parameters
    ---------------------
    panel_labels : bool, default=True
        Whether to add panel labels (A, B, C, etc.) to each subplot.
        Essential for publication figures with multiple panels.
    panel_label_start : str, default='A'
        Starting letter for panel labels. Subsequent panels use consecutive letters.
        Common options: 'A' for uppercase, 'a' for lowercase.
    custom_panel_labels : list of str, optional
        Custom labels for each panel. Must have at least len(metrics) elements.
        Overrides automatic lettering when provided.
        Example: ['(a)', '(b)', '(c)'] or ['Fig. 1A', 'Fig. 1B'].
    panel_label_offset : tuple of float, default=(0, 0)
        Additional (x, y) offset for panel labels in axes coordinates.
        Use to fine-tune label positioning relative to panel_label_position.
    panel_label_fontsize : int or float, default=14
        Font size for panel labels in points. Should be larger than other text
        elements for prominence.
    panel_label_fontweight : {'normal', 'bold', 'light', 'heavy'}, default='bold'
        Font weight for panel labels. 'bold' is standard for publications.
    panel_label_position : tuple of float, default=(-0.15, 1.05)
        Base position for panel labels in axes coordinates (x, y). 
        (-0.15, 1.05) places labels to the upper-left of each panel.
        Coordinates are relative to axes: (0,0)=bottom-left, (1,1)=top-right.

    Visual Styling Parameters
    ------------------------
    **visual_params : dict
        Additional keyword arguments passed to create_metric_comparison_plot()
        for each individual panel. All visualization parameters from that function
        are supported, including:
        
        plot_type : {'boxplot', 'violin', 'strip', 'barplot', 'paired_lines'}
            Plot type for all panels.
        palette : str or list
            Color scheme for all panels.
        title_fontsize, axis_label_fontsize, etc. : int or float
            Typography parameters.
        significance_fontsize, max_significance_pairs : int
            Statistical annotation parameters.
        grid, spines_visible : bool or dict
            Styling parameters.
        
        See create_metric_comparison_plot() documentation for complete list.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The complete multi-panel figure object with all subplots.
    axes : list of matplotlib.axes.Axes
        List of individual axes objects for each panel, in the same order
        as the metrics list. Can be used for further customization.

    Raises
    ------
    ValueError
        If metrics list is empty, contains invalid metric names, or if 
        custom_panel_labels is provided but has insufficient elements.
    AttributeError
        If analyzer object is not properly initialized or lacks required data.

    Notes
    -----
    Panel Layout Calculation:
    - Total panels = len(metrics)
    - Grid dimensions: nrows = ceil(len(metrics) / ncols), ncols = ncols
    - Empty panels in the last row are left blank

    Font and Style Consistency:
    - All panels use consistent styling parameters from visual_params
    - Panel labels use separate formatting parameters for emphasis
    - Shared y-axis scaling maintains visual coherence when appropriate

    Publication Guidelines:
    - Use dpi=300 or higher for journal submission
    - Ensure panel labels are clearly visible (bold, appropriate size)
    - Allow sufficient spacing between panels for annotations
    - Consider colorblind-friendly palettes for broad accessibility

    Memory Considerations:
    - Large multi-panel figures with high DPI can consume significant memory
    - Consider reducing DPI for preview versions
    - Close figure objects when no longer needed to free memory

    Examples
    --------
    Basic multi-panel figure with default settings:

    >>> fig, axes = create_publication_figure(
    ...     analyzer, 
    ...     ['accuracy', 'precision', 'recall', 'f1_score']
    ... )

    Custom 3-column layout with boxplots and custom display names:

    >>> display_names = {
    ...     'accuracy': 'Classification Accuracy',
    ...     'precision': 'Precision Score', 
    ...     'recall': 'Recall Score',
    ...     'f1_score': 'F1-Score'
    ... }
    >>> fig, axes = create_publication_figure(
    ...     analyzer,
    ...     ['accuracy', 'precision', 'recall', 'f1_score'],
    ...     figsize=(18, 12),
    ...     ncols=3,
    ...     metric_display_names=display_names,
    ...     plot_type='boxplot',
    ...     palette='viridis'
    ... )

    Publication-ready figure with custom panel labels and shared y-axis:

    >>> fig, axes = create_publication_figure(
    ...     analyzer,
    ...     ['metric1', 'metric2', 'metric3'],
    ...     figsize=(15, 10),
    ...     dpi=300,
    ...     shared_y_axis=True,
    ...     suptitle='Comprehensive Performance Comparison',
    ...     custom_panel_labels=['(a)', '(b)', '(c)'],
    ...     panel_label_fontsize=16,
    ...     left=0.10,
    ...     right=0.98,
    ...     hspace=0.30,
    ...     plot_type='barplot',
    ...     error_bar_type='sem',
    ...     significance_fontsize=12,
    ...     title_fontsize=11
    ... )

    High-density figure with precise layout control:

    >>> fig, axes = create_publication_figure(
    ...     analyzer,
    ...     metrics_list,
    ...     figsize=(20, 16),
    ...     ncols=4,
    ...     subplot_adjust_params={
    ...         'left': 0.06, 'right': 0.98, 
    ...         'top': 0.94, 'bottom': 0.06,
    ...         'hspace': 0.35, 'wspace': 0.25
    ...     },
    ...     panel_label_position=(-0.12, 1.02),
    ...     plot_type='violin',
    ...     strip_alpha=0.4,
    ...     box_width=0.5
    ... )

    See Also
    --------
    create_metric_comparison_plot : Single panel plotting function
    RigorousExperimentalAnalyzer.analyze_all_metrics : Batch statistical analysis
    matplotlib.pyplot.subplots_adjust : Manual subplot spacing control
    matplotlib.pyplot.tight_layout : Automatic layout optimization
    """
    _apply_rea_style()
    import string

    n_metrics = len(metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    font_family = visual_params.get('font_family')
    if font_family:
        _apply_font(font_family)
    # 创建图形
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # 默认视觉参数
    default_visual_params = {
        'plot_type': 'boxplot',
        'box_width': 0.7,
        'jitter_width': 0.35,
        'strip_size': 3,
        'flier_size': 0,
        'title_fontsize': 12,
        'axis_label_fontsize': 10,
        'tick_label_fontsize': 10,
        'significance_fontsize': 12,
        'legend_fontsize': 10,
        'max_significance_pairs': 3,
        'stat_test_display': 'short',
        'palette': 'husl',
        'grid': True,
        'grid_alpha': 0.25,
        'rotate_xlabels': True,
        'spines_visible': {'top': False, 'right': False}
    }
    
    # 更新参数
    default_visual_params.update(visual_params)
    
    # 准备面板标签
    if panel_labels:
        if custom_panel_labels is not None:
            if len(custom_panel_labels) < n_metrics:
                raise ValueError(f"custom_panel_labels长度({len(custom_panel_labels)})必须≥指标数量({n_metrics})")
            labels = custom_panel_labels[:n_metrics]
        else:
            # 从指定字母开始生成标签
            start_index = ord(panel_label_start.upper()) - ord('A')
            labels = [chr(ord('A') + start_index + i) for i in range(n_metrics)]
    else:
        labels = [None] * n_metrics
    
    # 创建子图
    axes = []
    for i, metric in enumerate(metrics):
        ax = plt.subplot(nrows, ncols, i + 1)
        axes.append(ax)
        # 获取该指标的显示名称
        if metric_display_names and metric in metric_display_names:
            display_name = metric_display_names[metric]
        else:
            display_name = None
        # 绘制时传递显示名称
        default_visual_params['display_metric_name'] = display_name
        # 直接在轴上绘制（重用核心绘图逻辑）
        analyzer._plot_metric_on_axis(ax, metric, **default_visual_params)
        
        # 添加面板标签
        if labels[i] is not None:
            label_x = panel_label_position[0] + panel_label_offset[0]
            label_y = panel_label_position[1] + panel_label_offset[1]
                
            ax.text(label_x, label_y, labels[i], 
                    transform=ax.transAxes,
                    fontsize=panel_label_fontsize, 
                    fontweight=panel_label_fontweight, 
                    va='top', ha='left')
        
        # 共享Y轴设置
        if shared_y_axis and i > 0:
            axes[0].get_shared_y_axes().join(axes[0], ax)
            if i % ncols != 0:  # 不是第一列
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)
    
    # 添加总标题
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    
    # 精确的布局调整
    if subplot_adjust_params is not None:
        fig.subplots_adjust(**subplot_adjust_params)
    else:
        fig.subplots_adjust(
            left=left, right=right, top=top, bottom=bottom,
            hspace=hspace, wspace=wspace
        )
    
    # Save if path provided (with DPI clamping to avoid backend truncation)
    save_path = visual_params.get('save_path')
    if save_path:
        _safe_save_figure(fig, save_path, dpi=dpi)
    
    return fig, axes


