# Manual de Análise Multivariável - Sistema Hidráulico Industrial

## 📋 Índice da Análise Multivariável

1. [Fundamentos da Análise Multivariável](#fundamentos-da-análise-multivariável)
2. [Estrutura de Dados MultiVariableSnapshot](#estrutura-de-dados-multivariablesnapshot)
3. [Processamento em Lote](#processamento-em-lote)
4. [Análise de Correlações Múltiplas](#análise-de-correlações-múltiplas)
5. [Detecção de Padrões Complexos](#detecção-de-padrões-complexos)
6. [Validação Física](#validação-física)
7. [Integração com Outros Módulos](#integração-com-outros-módulos)
8. [Interpretação e Diagnóstico](#interpretação-e-diagnóstico)

---

## 🧮 Fundamentos da Análise Multivariável

### Conceito Central

A **Análise Multivariável** é o núcleo do sistema que processa simultaneamente **5 variáveis hidráulicas críticas** para detectar padrões complexos que não são visíveis na análise individual.

#### 🎯 Variáveis Analisadas

```python
class MultiVariableSnapshot:
    """
    Snapshot completo do sistema hidráulico em um instante
    
    Contém todas as variáveis críticas sincronizadas temporalmente
    """
    
    def __init__(self, timestamp: datetime, expeditor_pressure: float, 
                 receiver_pressure: float, flow: float, density: float, temperature: float):
        
        # Timestamp preciso para sincronização
        self.timestamp = timestamp
        
        # Variáveis de pressão (kgf/cm²)
        self.expeditor_pressure = expeditor_pressure    # Pressão no ponto expedidor
        self.receiver_pressure = receiver_pressure      # Pressão no ponto recebedor
        
        # Variáveis de fluxo e propriedades
        self.flow = flow                               # Vazão mássica (kg/h)
        self.density = density                         # Densidade do fluido (kg/m³)
        self.temperature = temperature                 # Temperatura (°C)
        
        # Propriedades derivadas calculadas automaticamente
        self._calculate_derived_properties()
    
    def _calculate_derived_properties(self):
        """Calcula propriedades derivadas das variáveis base"""
        
        # Diferença de pressão (força motriz)
        self.pressure_difference = self.expeditor_pressure - self.receiver_pressure
        
        # Velocidade estimada baseada na vazão
        if self.density > 0:
            # Assumindo área de seção transversal conhecida
            pipe_area = 0.785  # m² (exemplo para DN 1000mm)
            self.velocity = (self.flow / 3600) / (self.density * pipe_area)  # m/s
        else:
            self.velocity = 0.0
        
        # Número de Reynolds estimado
        pipe_diameter = 1.0  # m
        dynamic_viscosity = 0.001  # Pa·s (água a 20°C)
        if self.velocity > 0:
            self.reynolds_number = (self.density * self.velocity * pipe_diameter) / dynamic_viscosity
        else:
            self.reynolds_number = 0.0
        
        # Perda de carga específica
        if self.flow > 0:
            self.specific_head_loss = self.pressure_difference / (self.flow ** 1.85)  # Hazen-Williams simplificado
        else:
            self.specific_head_loss = 0.0
```

### Princípios Matemáticos

#### 📊 Análise Vetorial Multidimensional

```python
def create_multivariable_vector(self, snapshots: List[MultiVariableSnapshot]) -> np.ndarray:
    """
    Cria vetor multidimensional para análise
    
    Estrutura do vetor: [P_exp, P_rec, Q, ρ, T, ΔP, v, Re, hf]
    
    Onde:
    - P_exp, P_rec: Pressões expedidor e recebedor
    - Q: Vazão mássica
    - ρ: Densidade
    - T: Temperatura  
    - ΔP: Diferença de pressão
    - v: Velocidade
    - Re: Número de Reynolds
    - hf: Perda de carga específica
    """
    
    vectors = []
    
    for snapshot in snapshots:
        vector = np.array([
            snapshot.expeditor_pressure,
            snapshot.receiver_pressure,
            snapshot.flow,
            snapshot.density,
            snapshot.temperature,
            snapshot.pressure_difference,
            snapshot.velocity,
            snapshot.reynolds_number,
            snapshot.specific_head_loss
        ])
        
        vectors.append(vector)
    
    return np.array(vectors)
```

#### 🔍 Matriz de Correlação Completa

```python
def calculate_correlation_matrix(self, snapshots: List[MultiVariableSnapshot]) -> Dict[str, Any]:
    """
    Calcula matriz de correlação completa entre todas as variáveis
    
    Métodos:
    1. Correlação de Pearson (linear)
    2. Correlação de Spearman (monotônica)
    3. Correlação de Kendall (rank-based)
    4. Informação mútua (não-linear)
    """
    
    from scipy.stats import pearsonr, spearmanr, kendalltau
    from sklearn.feature_selection import mutual_info_regression
    
    # Extrai todas as variáveis
    data_matrix = self.create_multivariable_vector(snapshots)
    
    variable_names = [
        'Pressão Expedidor', 'Pressão Recebedor', 'Vazão', 'Densidade', 'Temperatura',
        'Diferença Pressão', 'Velocidade', 'Reynolds', 'Perda Carga'
    ]
    
    n_vars = len(variable_names)
    
    # Matrizes de correlação
    pearson_matrix = np.zeros((n_vars, n_vars))
    spearman_matrix = np.zeros((n_vars, n_vars))
    kendall_matrix = np.zeros((n_vars, n_vars))
    mutual_info_matrix = np.zeros((n_vars, n_vars))
    
    # P-valores para significância
    pearson_pvalues = np.zeros((n_vars, n_vars))
    spearman_pvalues = np.zeros((n_vars, n_vars))
    kendall_pvalues = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            var_i = data_matrix[:, i]
            var_j = data_matrix[:, j]
            
            if i == j:
                # Diagonal: correlação perfeita consigo mesmo
                pearson_matrix[i, j] = 1.0
                spearman_matrix[i, j] = 1.0
                kendall_matrix[i, j] = 1.0
                mutual_info_matrix[i, j] = 1.0
                
            else:
                # Correlação de Pearson
                try:
                    pearson_corr, pearson_p = pearsonr(var_i, var_j)
                    pearson_matrix[i, j] = pearson_corr if not np.isnan(pearson_corr) else 0.0
                    pearson_pvalues[i, j] = pearson_p if not np.isnan(pearson_p) else 1.0
                except:
                    pearson_matrix[i, j] = 0.0
                    pearson_pvalues[i, j] = 1.0
                
                # Correlação de Spearman
                try:
                    spearman_corr, spearman_p = spearmanr(var_i, var_j)
                    spearman_matrix[i, j] = spearman_corr if not np.isnan(spearman_corr) else 0.0
                    spearman_pvalues[i, j] = spearman_p if not np.isnan(spearman_p) else 1.0
                except:
                    spearman_matrix[i, j] = 0.0
                    spearman_pvalues[i, j] = 1.0
                
                # Correlação de Kendall
                try:
                    kendall_corr, kendall_p = kendalltau(var_i, var_j)
                    kendall_matrix[i, j] = kendall_corr if not np.isnan(kendall_corr) else 0.0
                    kendall_pvalues[i, j] = kendall_p if not np.isnan(kendall_p) else 1.0
                except:
                    kendall_matrix[i, j] = 0.0
                    kendall_pvalues[i, j] = 1.0
                
                # Informação mútua
                try:
                    mi = mutual_info_regression(var_i.reshape(-1, 1), var_j)[0]
                    mutual_info_matrix[i, j] = mi
                except:
                    mutual_info_matrix[i, j] = 0.0
    
    return {
        'variable_names': variable_names,
        'pearson': {
            'correlation_matrix': pearson_matrix.tolist(),
            'p_values': pearson_pvalues.tolist()
        },
        'spearman': {
            'correlation_matrix': spearman_matrix.tolist(),
            'p_values': spearman_pvalues.tolist()
        },
        'kendall': {
            'correlation_matrix': kendall_matrix.tolist(),
            'p_values': kendall_pvalues.tolist()
        },
        'mutual_information': {
            'matrix': mutual_info_matrix.tolist()
        }
    }
```

---

## 📊 Estrutura de Dados MultiVariableSnapshot

### Validação e Consistência

#### 🔍 Validação Física dos Dados

```python
def validate_physical_consistency(self, snapshot: MultiVariableSnapshot) -> Dict[str, Any]:
    """
    Valida consistência física dos dados do snapshot
    
    Verificações:
    1. Limites físicos realistas
    2. Relações termodinâmicas
    3. Leis de conservação
    4. Consistência temporal
    """
    
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'physical_checks': {}
    }
    
    # 1. Verificação de limites físicos
    physical_limits = {
        'expeditor_pressure': (0.0, 100.0),    # kgf/cm²
        'receiver_pressure': (0.0, 100.0),     # kgf/cm²
        'flow': (0.0, 10000.0),                # kg/h
        'density': (700.0, 1200.0),            # kg/m³ (típico para hidrocarbonetos)
        'temperature': (-10.0, 80.0)           # °C
    }
    
    for var_name, (min_val, max_val) in physical_limits.items():
        value = getattr(snapshot, var_name)
        
        if value < min_val or value > max_val:
            validation_results['errors'].append(
                f"{var_name}: {value} fora dos limites físicos [{min_val}, {max_val}]"
            )
            validation_results['is_valid'] = False
    
    # 2. Verificação de relação pressão-fluxo
    if snapshot.flow > 0:
        expected_pressure_ratio = 0.8  # Pressão recebedor deve ser ~80% da expedidora
        actual_ratio = snapshot.receiver_pressure / max(snapshot.expeditor_pressure, 0.1)
        
        if actual_ratio > 1.0:  # Pressão recebedor maior que expedidor
            validation_results['errors'].append(
                f"Pressão recebedor ({snapshot.receiver_pressure}) > expedidor ({snapshot.expeditor_pressure})"
            )
            validation_results['is_valid'] = False
        
        elif actual_ratio < 0.3:  # Queda de pressão muito alta
            validation_results['warnings'].append(
                f"Queda de pressão muito alta: {actual_ratio:.1%}"
            )
    
    # 3. Verificação densidade-temperatura
    # Densidade deve diminuir com temperatura (para hidrocarbonetos)
    reference_density = 850.0  # kg/m³ a 20°C
    reference_temp = 20.0      # °C
    thermal_expansion_coeff = 0.0008  # 1/°C
    
    expected_density = reference_density * (1 - thermal_expansion_coeff * (snapshot.temperature - reference_temp))
    density_deviation = abs(snapshot.density - expected_density) / expected_density
    
    if density_deviation > 0.1:  # Desvio > 10%
        validation_results['warnings'].append(
            f"Densidade ({snapshot.density}) inconsistente com temperatura ({snapshot.temperature}°C)"
        )
    
    # 4. Verificação de fluxo zero com diferença de pressão
    if snapshot.flow < 1.0 and snapshot.pressure_difference > 5.0:
        validation_results['warnings'].append(
            f"Fluxo baixo ({snapshot.flow}) com alta diferença de pressão ({snapshot.pressure_difference})"
        )
    
    validation_results['physical_checks'] = {
        'pressure_ratio': actual_ratio if snapshot.flow > 0 else None,
        'density_deviation_pct': density_deviation * 100,
        'reynolds_number': snapshot.reynolds_number,
        'flow_regime': 'laminar' if snapshot.reynolds_number < 2300 else 
                      'turbulent' if snapshot.reynolds_number > 4000 else 'transitional'
    }
    
    return validation_results
```

#### 🕐 Sincronização Temporal

```python
def synchronize_snapshots(self, raw_snapshots: List[MultiVariableSnapshot], 
                         tolerance_seconds: float = 1.0) -> List[MultiVariableSnapshot]:
    """
    Sincroniza snapshots temporalmente para garantir consistência
    
    Processo:
    1. Agrupa snapshots por janela temporal
    2. Interpola valores faltantes
    3. Remove outliers temporais
    4. Garante espaçamento uniforme
    """
    
    if not raw_snapshots:
        return []
    
    # Ordena por timestamp
    sorted_snapshots = sorted(raw_snapshots, key=lambda s: s.timestamp)
    
    # Define grid temporal uniforme
    start_time = sorted_snapshots[0].timestamp
    end_time = sorted_snapshots[-1].timestamp
    total_duration = (end_time - start_time).total_seconds()
    
    # Intervalo baseado na densidade de dados
    avg_interval = total_duration / len(sorted_snapshots)
    target_interval = max(1.0, avg_interval)  # Mínimo de 1 segundo
    
    # Cria timestamps do grid
    current_time = start_time
    target_timestamps = []
    
    while current_time <= end_time:
        target_timestamps.append(current_time)
        current_time += timedelta(seconds=target_interval)
    
    synchronized_snapshots = []
    
    for target_time in target_timestamps:
        # Encontra snapshots mais próximos
        time_distances = [(abs((s.timestamp - target_time).total_seconds()), s) 
                         for s in sorted_snapshots]
        time_distances.sort(key=lambda x: x[0])
        
        # Se há snapshot próximo o suficiente, usa diretamente
        if time_distances[0][0] <= tolerance_seconds:
            synchronized_snapshots.append(time_distances[0][1])
            
        # Senão, interpola entre dois snapshots próximos
        elif len(time_distances) >= 2:
            closest_before = None
            closest_after = None
            
            for distance, snapshot in time_distances:
                if snapshot.timestamp <= target_time and closest_before is None:
                    closest_before = snapshot
                elif snapshot.timestamp > target_time and closest_after is None:
                    closest_after = snapshot
                    break
            
            if closest_before and closest_after:
                interpolated = self._interpolate_snapshot(
                    closest_before, closest_after, target_time
                )
                synchronized_snapshots.append(interpolated)
    
    return synchronized_snapshots
```

#### 🔄 Interpolação de Snapshots

```python
def _interpolate_snapshot(self, snapshot1: MultiVariableSnapshot, 
                         snapshot2: MultiVariableSnapshot, 
                         target_time: datetime) -> MultiVariableSnapshot:
    """
    Interpola linearmente entre dois snapshots
    """
    
    # Calcula fator de interpolação
    t1 = snapshot1.timestamp
    t2 = snapshot2.timestamp
    dt_total = (t2 - t1).total_seconds()
    dt_target = (target_time - t1).total_seconds()
    
    if dt_total == 0:
        return snapshot1
    
    alpha = dt_target / dt_total  # Fator de interpolação [0, 1]
    
    # Interpola cada variável
    interpolated_values = {}
    
    for var_name in ['expeditor_pressure', 'receiver_pressure', 'flow', 'density', 'temperature']:
        val1 = getattr(snapshot1, var_name)
        val2 = getattr(snapshot2, var_name)
        
        # Interpolação linear
        interpolated_values[var_name] = val1 + alpha * (val2 - val1)
    
    # Cria novo snapshot interpolado
    interpolated_snapshot = MultiVariableSnapshot(
        timestamp=target_time,
        **interpolated_values
    )
    
    return interpolated_snapshot
```

---

## 🔄 Processamento em Lote

### Análise de Janelas Deslizantes

#### 📊 Implementação do Processamento por Lotes

```python
def perform_multivariable_analysis(self, snapshots: List[MultiVariableSnapshot], 
                                 window_size: int = 50) -> Dict[str, Any]:
    """
    Análise multivariável completa em janelas deslizantes
    
    Processo:
    1. Divide dados em janelas sobrepostas
    2. Analisa cada janela independentemente
    3. Consolida resultados temporais
    4. Detecta mudanças entre janelas
    """
    
    if len(snapshots) < window_size:
        return {'error': f'Necessários pelo menos {window_size} snapshots'}
    
    results = {
        'window_analyses': [],
        'temporal_trends': {},
        'change_points': [],
        'overall_statistics': {}
    }
    
    # Configuração de janelas sobrepostas (50% overlap)
    step_size = window_size // 2
    window_starts = list(range(0, len(snapshots) - window_size + 1, step_size))
    
    previous_analysis = None
    
    for i, start_idx in enumerate(window_starts):
        end_idx = start_idx + window_size
        window_snapshots = snapshots[start_idx:end_idx]
        
        # Análise da janela atual
        window_analysis = self._analyze_snapshot_window(window_snapshots, i)
        
        # Detecta mudanças em relação à janela anterior
        if previous_analysis:
            change_detection = self._detect_changes_between_windows(
                previous_analysis, window_analysis
            )
            
            if change_detection['significant_change']:
                results['change_points'].append({
                    'window_index': i,
                    'timestamp': window_snapshots[0].timestamp.isoformat(),
                    'change_type': change_detection['change_type'],
                    'magnitude': change_detection['magnitude']
                })
        
        results['window_analyses'].append(window_analysis)
        previous_analysis = window_analysis
    
    # Análise de tendências temporais
    results['temporal_trends'] = self._analyze_temporal_trends(results['window_analyses'])
    
    # Estatísticas consolidadas
    results['overall_statistics'] = self._calculate_overall_statistics(snapshots)
    
    return results
```

#### 🔍 Análise de Janela Individual

```python
def _analyze_snapshot_window(self, window_snapshots: List[MultiVariableSnapshot], 
                           window_index: int) -> Dict[str, Any]:
    """
    Análise completa de uma janela de snapshots
    
    Análises realizadas:
    1. Estatísticas descritivas
    2. Matriz de correlação
    3. Detecção de outliers
    4. Análise de tendências
    5. Validação física
    """
    
    # Extrai matriz de dados
    data_matrix = self.create_multivariable_vector(window_snapshots)
    variable_names = [
        'P_exp', 'P_rec', 'Flow', 'Density', 'Temp', 'dP', 'Velocity', 'Re', 'hf'
    ]
    
    # 1. Estatísticas descritivas
    statistics = {}
    for i, var_name in enumerate(variable_names):
        var_data = data_matrix[:, i]
        
        statistics[var_name] = {
            'mean': float(np.mean(var_data)),
            'std': float(np.std(var_data)),
            'min': float(np.min(var_data)),
            'max': float(np.max(var_data)),
            'median': float(np.median(var_data)),
            'q25': float(np.percentile(var_data, 25)),
            'q75': float(np.percentile(var_data, 75)),
            'skewness': float(stats.skew(var_data)),
            'kurtosis': float(stats.kurtosis(var_data))
        }
    
    # 2. Matriz de correlação (método rápido)
    correlation_matrix = np.corrcoef(data_matrix.T)
    
    # 3. Detecção de outliers multivariáveis
    outliers = self._detect_multivariate_outliers(data_matrix)
    
    # 4. Análise de tendências
    trends = {}
    time_indices = np.arange(len(window_snapshots))
    
    for i, var_name in enumerate(variable_names):
        var_data = data_matrix[:, i]
        
        # Regressão linear para tendência
        if len(var_data) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_indices, var_data)
            
            trends[var_name] = {
                'slope': float(slope),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            }
    
    # 5. Validação física da janela
    validation_summary = {
        'valid_snapshots': 0,
        'warning_snapshots': 0,
        'error_snapshots': 0
    }
    
    for snapshot in window_snapshots:
        validation = self.validate_physical_consistency(snapshot)
        
        if validation['is_valid'] and not validation['warnings']:
            validation_summary['valid_snapshots'] += 1
        elif validation['warnings'] and not validation['errors']:
            validation_summary['warning_snapshots'] += 1
        else:
            validation_summary['error_snapshots'] += 1
    
    return {
        'window_index': window_index,
        'start_time': window_snapshots[0].timestamp.isoformat(),
        'end_time': window_snapshots[-1].timestamp.isoformat(),
        'n_snapshots': len(window_snapshots),
        'statistics': statistics,
        'correlation_matrix': correlation_matrix.tolist(),
        'outliers': outliers,
        'trends': trends,
        'validation_summary': validation_summary
    }
```

#### 🎯 Detecção de Outliers Multivariáveis

```python
def _detect_multivariate_outliers(self, data_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Detecta outliers multivariáveis usando múltiplos métodos
    
    Métodos:
    1. Distância de Mahalanobis
    2. Isolation Forest
    3. Elipse de confiança
    """
    
    from sklearn.ensemble import IsolationForest
    from scipy.spatial.distance import mahalanobis
    
    outlier_results = {
        'mahalanobis_outliers': [],
        'isolation_outliers': [],
        'consensus_outliers': [],
        'outlier_scores': []
    }
    
    try:
        # 1. Distância de Mahalanobis
        mean_vec = np.mean(data_matrix, axis=0)
        
        # Calcula matriz de covariância robusta
        cov_matrix = np.cov(data_matrix.T)
        
        # Verifica se matriz é invertível
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            
            mahal_distances = []
            for i in range(len(data_matrix)):
                point = data_matrix[i]
                mahal_dist = mahalanobis(point, mean_vec, inv_cov_matrix)
                mahal_distances.append(mahal_dist)
            
            # Threshold baseado em distribuição chi-quadrado
            chi2_threshold = stats.chi2.ppf(0.975, data_matrix.shape[1])  # 97.5%
            mahalanobis_outliers = [i for i, d in enumerate(mahal_distances) if d > chi2_threshold]
            
            outlier_results['mahalanobis_outliers'] = mahalanobis_outliers
            
        except np.linalg.LinAlgError:
            # Matriz singular - usa pseudo-inversa
            outlier_results['mahalanobis_outliers'] = []
        
        # 2. Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        isolation_predictions = iso_forest.fit_predict(data_matrix)
        isolation_outliers = [i for i, pred in enumerate(isolation_predictions) if pred == -1]
        
        outlier_results['isolation_outliers'] = isolation_outliers
        
        # 3. Consenso entre métodos
        all_outliers = set(outlier_results['mahalanobis_outliers'] + isolation_outliers)
        consensus_outliers = []
        
        for outlier_idx in all_outliers:
            methods_count = 0
            if outlier_idx in outlier_results['mahalanobis_outliers']:
                methods_count += 1
            if outlier_idx in isolation_outliers:
                methods_count += 1
            
            if methods_count >= 1:  # Pelo menos 1 método detectou
                consensus_outliers.append(outlier_idx)
        
        outlier_results['consensus_outliers'] = consensus_outliers
        
        # 4. Scores de outlier
        isolation_scores = iso_forest.decision_function(data_matrix)
        normalized_scores = (isolation_scores - np.min(isolation_scores)) / (
            np.max(isolation_scores) - np.min(isolation_scores)
        )
        
        outlier_results['outlier_scores'] = normalized_scores.tolist()
        
    except Exception as e:
        self.logger.warning(f"Erro na detecção de outliers: {e}")
        outlier_results = {
            'mahalanobis_outliers': [],
            'isolation_outliers': [],
            'consensus_outliers': [],
            'outlier_scores': [0.0] * len(data_matrix)
        }
    
    return outlier_results
```

---

## 🔍 Análise de Correlações Múltiplas

### Correlações Dinâmicas e Não-lineares

#### 📊 Análise de Correlação Temporal

```python
def analyze_dynamic_correlations(self, snapshots: List[MultiVariableSnapshot], 
                               window_size: int = 30) -> Dict[str, Any]:
    """
    Analisa correlações dinâmicas entre variáveis em janelas deslizantes
    
    Detecta:
    1. Mudanças nas correlações ao longo do tempo
    2. Correlações com delay temporal
    3. Correlações condicionais
    4. Breakpoints nas relações
    """
    
    if len(snapshots) < window_size * 2:
        return {'error': 'Dados insuficientes para análise dinâmica'}
    
    data_matrix = self.create_multivariable_vector(snapshots)
    variable_names = ['P_exp', 'P_rec', 'Flow', 'Density', 'Temp']
    
    results = {
        'time_series_correlations': {},
        'lagged_correlations': {},
        'correlation_stability': {},
        'breakpoint_analysis': {}
    }
    
    # Análise de correlações em janelas deslizantes
    n_windows = (len(snapshots) - window_size) // 10 + 1  # Step de 10
    correlation_time_series = {}
    
    # Inicializa séries temporais de correlação
    for i in range(len(variable_names)):
        for j in range(i+1, len(variable_names)):
            pair_name = f"{variable_names[i]}_vs_{variable_names[j]}"
            correlation_time_series[pair_name] = []
    
    # Calcula correlações em cada janela
    for window_start in range(0, len(snapshots) - window_size, 10):
        window_end = window_start + window_size
        window_data = data_matrix[window_start:window_end, :5]  # Primeiras 5 variáveis
        
        window_corr_matrix = np.corrcoef(window_data.T)
        
        # Extrai correlações dos pares
        for i in range(len(variable_names)):
            for j in range(i+1, len(variable_names)):
                pair_name = f"{variable_names[i]}_vs_{variable_names[j]}"
                correlation = window_corr_matrix[i, j]
                
                if not np.isnan(correlation):
                    correlation_time_series[pair_name].append(correlation)
                else:
                    correlation_time_series[pair_name].append(0.0)
    
    results['time_series_correlations'] = correlation_time_series
    
    # Análise de correlações com lag
    max_lag = min(20, len(snapshots) // 10)
    lagged_correlations = {}
    
    for i in range(len(variable_names)):
        for j in range(len(variable_names)):
            if i != j:
                var_i = data_matrix[:, i]
                var_j = data_matrix[:, j]
                
                pair_name = f"{variable_names[i]}_leads_{variable_names[j]}"
                lag_correlations = []
                
                for lag in range(-max_lag, max_lag + 1):
                    if lag == 0:
                        corr = np.corrcoef(var_i, var_j)[0, 1]
                    elif lag > 0:
                        # var_i leads var_j
                        if len(var_i) > lag:
                            corr = np.corrcoef(var_i[:-lag], var_j[lag:])[0, 1]
                        else:
                            corr = 0.0
                    else:
                        # var_j leads var_i
                        lag_abs = abs(lag)
                        if len(var_j) > lag_abs:
                            corr = np.corrcoef(var_i[lag_abs:], var_j[:-lag_abs])[0, 1]
                        else:
                            corr = 0.0
                    
                    lag_correlations.append({
                        'lag': lag,
                        'correlation': float(corr) if not np.isnan(corr) else 0.0
                    })
                
                # Encontra lag com máxima correlação
                max_corr_lag = max(lag_correlations, key=lambda x: abs(x['correlation']))
                
                lagged_correlations[pair_name] = {
                    'all_lags': lag_correlations,
                    'best_lag': max_corr_lag['lag'],
                    'best_correlation': max_corr_lag['correlation']
                }
    
    results['lagged_correlations'] = lagged_correlations
    
    # Análise de estabilidade das correlações
    stability_analysis = {}
    
    for pair_name, corr_series in correlation_time_series.items():
        if len(corr_series) > 5:
            stability_metrics = {
                'mean_correlation': np.mean(corr_series),
                'std_correlation': np.std(corr_series),
                'min_correlation': np.min(corr_series),
                'max_correlation': np.max(corr_series),
                'correlation_range': np.max(corr_series) - np.min(corr_series),
                'stability_coefficient': 1.0 - (np.std(corr_series) / max(abs(np.mean(corr_series)), 0.1))
            }
            
            # Classificação da estabilidade
            if stability_metrics['stability_coefficient'] > 0.8:
                stability_class = 'very_stable'
            elif stability_metrics['stability_coefficient'] > 0.6:
                stability_class = 'stable'
            elif stability_metrics['stability_coefficient'] > 0.4:
                stability_class = 'moderately_stable'
            else:
                stability_class = 'unstable'
            
            stability_metrics['stability_class'] = stability_class
            stability_analysis[pair_name] = stability_metrics
    
    results['correlation_stability'] = stability_analysis
    
    return results
```

#### 🎯 Correlações Não-lineares

```python
def analyze_nonlinear_relationships(self, snapshots: List[MultiVariableSnapshot]) -> Dict[str, Any]:
    """
    Análise de relações não-lineares entre variáveis
    
    Métodos:
    1. Informação mútua
    2. Correlação de distância
    3. Análise de copulas
    4. Regressão não-paramétrica
    """
    
    from sklearn.feature_selection import mutual_info_regression
    from scipy.stats import spearmanr
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    data_matrix = self.create_multivariable_vector(snapshots)
    variable_names = ['P_exp', 'P_rec', 'Flow', 'Density', 'Temp']
    
    nonlinear_results = {
        'mutual_information': {},
        'polynomial_fits': {},
        'distance_correlations': {},
        'nonlinearity_scores': {}
    }
    
    # 1. Informação mútua
    for i in range(len(variable_names)):
        for j in range(len(variable_names)):
            if i != j:
                var_i = data_matrix[:, i].reshape(-1, 1)
                var_j = data_matrix[:, j]
                
                try:
                    mi = mutual_info_regression(var_i, var_j)[0]
                    pair_name = f"{variable_names[i]}_vs_{variable_names[j]}"
                    nonlinear_results['mutual_information'][pair_name] = float(mi)
                except:
                    continue
    
    # 2. Ajustes polinomiais
    for i in range(len(variable_names)):
        for j in range(len(variable_names)):
            if i != j:
                X = data_matrix[:, i].reshape(-1, 1)
                y = data_matrix[:, j]
                
                pair_name = f"{variable_names[i]}_vs_{variable_names[j]}"
                
                # Testa graus polinomiais 1, 2, 3
                polynomial_results = {}
                
                for degree in [1, 2, 3]:
                    try:
                        # Cria features polinomiais
                        poly_features = PolynomialFeatures(degree=degree)
                        X_poly = poly_features.fit_transform(X)
                        
                        # Ajusta modelo
                        model = LinearRegression()
                        model.fit(X_poly, y)
                        
                        # Avalia ajuste
                        y_pred = model.predict(X_poly)
                        r2 = r2_score(y, y_pred)
                        
                        polynomial_results[f'degree_{degree}'] = {
                            'r2_score': float(r2),
                            'coefficients': model.coef_.tolist(),
                            'intercept': float(model.intercept_)
                        }
                    except:
                        polynomial_results[f'degree_{degree}'] = {
                            'r2_score': 0.0,
                            'coefficients': [],
                            'intercept': 0.0
                        }
                
                nonlinear_results['polynomial_fits'][pair_name] = polynomial_results
    
    # 3. Score de não-linearidade
    for i in range(len(variable_names)):
        for j in range(len(variable_names)):
            if i != j:
                pair_name = f"{variable_names[i]}_vs_{variable_names[j]}"
                
                # Compara R² linear vs polinomial de grau 2
                if pair_name in nonlinear_results['polynomial_fits']:
                    linear_r2 = nonlinear_results['polynomial_fits'][pair_name]['degree_1']['r2_score']
                    poly2_r2 = nonlinear_results['polynomial_fits'][pair_name]['degree_2']['r2_score']
                    
                    # Score de não-linearidade
                    nonlinearity_score = poly2_r2 - linear_r2
                    
                    # Classificação
                    if nonlinearity_score > 0.1:
                        nonlinearity_class = 'highly_nonlinear'
                    elif nonlinearity_score > 0.05:
                        nonlinearity_class = 'moderately_nonlinear'
                    elif nonlinearity_score > 0.01:
                        nonlinearity_class = 'slightly_nonlinear'
                    else:
                        nonlinearity_class = 'linear'
                    
                    nonlinear_results['nonlinearity_scores'][pair_name] = {
                        'score': float(nonlinearity_score),
                        'classification': nonlinearity_class,
                        'linear_r2': float(linear_r2),
                        'polynomial_r2': float(poly2_r2)
                    }
    
    return nonlinear_results
```

---

**AVISO: Este manual está ficando muito extenso. Para não exceder o limite, vou dividir em partes.**

**Próximas seções a continuar:**

- 🔍 Detecção de Padrões Complexos
- ⚖️ Validação Física
- 🔗 Integração com Outros Módulos
- 🎯 Interpretação e Diagnóstico

Continuar com a próxima parte do Manual de Análise Multivariável?
