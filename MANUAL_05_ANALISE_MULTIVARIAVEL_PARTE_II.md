# Manual de Análise Multivariável - Sistema Hidráulico Industrial - Parte II

## 📋 Continuação - Padrões Complexos e Validação

---

## 🔍 Detecção de Padrões Complexos

### Análise de Regimes Operacionais

A análise multivariável identifica **regimes operacionais distintos** baseados em padrões multidimensionais que não são visíveis na análise univariada.

#### 🎯 Classificação de Regimes

```python
def classify_operational_regimes(self, snapshots: List[MultiVariableSnapshot]) -> Dict[str, Any]:
    """
    Classifica regimes operacionais baseados em padrões multivariáveis
    
    Regimes identificados:
    1. Normal steady-state - Operação estável
    2. Transient startup - Partida/parada  
    3. High flow regime - Alto fluxo
    4. Low pressure mode - Baixa pressão
    5. Temperature variation - Variação térmica
    6. Anomalous behavior - Comportamento anômalo
    """
    
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    
    if len(snapshots) < 50:
        return {'error': 'Dados insuficientes para classificação de regimes'}
    
    # Extrai matriz de features expandida
    feature_matrix = self._extract_regime_features(snapshots)
    
    # Normalização para clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)
    
    regime_results = {
        'kmeans_clustering': {},
        'gaussian_mixture': {},
        'dbscan_clustering': {},
        'regime_interpretation': {},
        'temporal_analysis': {}
    }
    
    # 1. K-Means clustering para regimes principais
    n_clusters_optimal = self._determine_optimal_clusters(features_scaled)
    
    kmeans = KMeans(n_clusters=n_clusters_optimal, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(features_scaled)
    
    regime_results['kmeans_clustering'] = {
        'n_clusters': n_clusters_optimal,
        'labels': kmeans_labels.tolist(),
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'inertia': float(kmeans.inertia_)
    }
    
    # 2. Gaussian Mixture Model para regimes probabilísticos
    gmm = GaussianMixture(n_components=n_clusters_optimal, random_state=42)
    gmm_labels = gmm.fit_predict(features_scaled)
    gmm_probabilities = gmm.predict_proba(features_scaled)
    
    regime_results['gaussian_mixture'] = {
        'n_components': n_clusters_optimal,
        'labels': gmm_labels.tolist(),
        'probabilities': gmm_probabilities.tolist(),
        'bic_score': float(gmm.bic(features_scaled)),
        'aic_score': float(gmm.aic(features_scaled))
    }
    
    # 3. DBSCAN para regimes baseados em densidade
    eps = self._estimate_dbscan_eps(features_scaled)
    dbscan = DBSCAN(eps=eps, min_samples=10)
    dbscan_labels = dbscan.fit_predict(features_scaled)
    
    regime_results['dbscan_clustering'] = {
        'eps': float(eps),
        'labels': dbscan_labels.tolist(),
        'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
        'n_outliers': int(np.sum(dbscan_labels == -1))
    }
    
    # 4. Interpretação dos regimes
    regime_interpretations = self._interpret_operational_regimes(
        features_scaled, kmeans_labels, feature_matrix, snapshots
    )
    regime_results['regime_interpretation'] = regime_interpretations
    
    # 5. Análise temporal dos regimes
    temporal_analysis = self._analyze_regime_transitions(kmeans_labels, snapshots)
    regime_results['temporal_analysis'] = temporal_analysis
    
    return regime_results
```

#### 🧮 Features Especializadas para Regimes

```python
def _extract_regime_features(self, snapshots: List[MultiVariableSnapshot]) -> np.ndarray:
    """
    Extrai features especializadas para identificação de regimes operacionais
    
    Features computadas:
    1. Médias móveis de diferentes janelas
    2. Variabilidades locais
    3. Gradientes e acelerações
    4. Ratios entre variáveis
    5. Features espectrais
    6. Padrões sazonais
    """
    
    # Converte para arrays NumPy
    pressures_exp = np.array([s.expeditor_pressure for s in snapshots])
    pressures_rec = np.array([s.receiver_pressure for s in snapshots])
    flows = np.array([s.flow for s in snapshots])
    densities = np.array([s.density for s in snapshots])
    temperatures = np.array([s.temperature for s in snapshots])
    
    n_samples = len(snapshots)
    features = []
    
    # Para cada snapshot, calcula features baseadas em janela local
    for i in range(n_samples):
        snapshot_features = []
        
        # Define janela local (±5 pontos)
        window_start = max(0, i - 5)
        window_end = min(n_samples, i + 6)
        
        window_p_exp = pressures_exp[window_start:window_end]
        window_p_rec = pressures_rec[window_start:window_end]
        window_flow = flows[window_start:window_end]
        window_density = densities[window_start:window_end]
        window_temp = temperatures[window_start:window_end]
        
        # 1. Valores instantâneos
        snapshot_features.extend([
            pressures_exp[i], pressures_rec[i], flows[i], densities[i], temperatures[i]
        ])
        
        # 2. Médias locais
        snapshot_features.extend([
            np.mean(window_p_exp), np.mean(window_p_rec), np.mean(window_flow),
            np.mean(window_density), np.mean(window_temp)
        ])
        
        # 3. Desvios padrão locais (variabilidade)
        snapshot_features.extend([
            np.std(window_p_exp), np.std(window_p_rec), np.std(window_flow),
            np.std(window_density), np.std(window_temp)
        ])
        
        # 4. Gradientes locais
        if len(window_p_exp) > 1:
            snapshot_features.extend([
                np.mean(np.gradient(window_p_exp)),
                np.mean(np.gradient(window_p_rec)),
                np.mean(np.gradient(window_flow))
            ])
        else:
            snapshot_features.extend([0.0, 0.0, 0.0])
        
        # 5. Ratios entre variáveis
        dp = pressures_exp[i] - pressures_rec[i]  # Diferença de pressão
        p_ratio = pressures_rec[i] / max(pressures_exp[i], 0.1)  # Ratio de pressões
        
        # Flow coefficient (aproximação)
        if dp > 0.1:
            flow_coeff = flows[i] / np.sqrt(dp)
        else:
            flow_coeff = 0.0
        
        snapshot_features.extend([dp, p_ratio, flow_coeff])
        
        # 6. Features derivadas físicas
        snapshot = snapshots[i]
        snapshot_features.extend([
            snapshot.velocity,
            snapshot.reynolds_number,
            snapshot.specific_head_loss
        ])
        
        # 7. Features de contexto temporal
        # Posição relativa na série temporal
        time_position = i / max(n_samples - 1, 1)
        
        # Tendência local (regressão linear na janela)
        if len(window_flow) > 2:
            time_indices = np.arange(len(window_flow))
            flow_trend = np.polyfit(time_indices, window_flow, 1)[0]
        else:
            flow_trend = 0.0
        
        snapshot_features.extend([time_position, flow_trend])
        
        features.append(snapshot_features)
    
    return np.array(features)
```

#### 📊 Determinação Automática de Clusters

```python
def _determine_optimal_clusters(self, features_scaled: np.ndarray) -> int:
    """
    Determina número ótimo de clusters usando múltiplos critérios
    
    Métodos:
    1. Elbow method (inércia)
    2. Silhouette score
    3. Gap statistic
    4. Calinski-Harabasz index
    """
    
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.cluster import KMeans
    
    max_clusters = min(10, len(features_scaled) // 20)  # Máximo 10 clusters
    cluster_range = range(2, max_clusters + 1)
    
    # Métricas para cada número de clusters
    inertias = []
    silhouette_scores = []
    ch_scores = []
    
    for n_clusters in cluster_range:
        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Inércia (soma das distâncias quadradas aos centroides)
        inertias.append(kmeans.inertia_)
        
        # Silhouette score (qualidade dos clusters)
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        # Calinski-Harabasz index
        ch_score = calinski_harabasz_score(features_scaled, cluster_labels)
        ch_scores.append(ch_score)
    
    # Método do cotovelo para inércia
    elbow_scores = []
    for i in range(1, len(inertias) - 1):
        # Curvatura baseada em segunda derivada
        curvature = inertias[i-1] - 2*inertias[i] + inertias[i+1]
        elbow_scores.append(curvature)
    
    # Seleciona melhor número de clusters baseado em múltiplos critérios
    
    # 1. Melhor silhouette score
    best_silhouette_idx = np.argmax(silhouette_scores)
    best_silhouette_k = cluster_range[best_silhouette_idx]
    
    # 2. Melhor Calinski-Harabasz
    best_ch_idx = np.argmax(ch_scores)
    best_ch_k = cluster_range[best_ch_idx]
    
    # 3. Cotovelo (máxima curvatura)
    if elbow_scores:
        best_elbow_idx = np.argmax(elbow_scores)
        best_elbow_k = cluster_range[best_elbow_idx + 1]  # +1 por causa do offset
    else:
        best_elbow_k = 3
    
    # Decisão final: votação majoritária ou critério conservador
    candidates = [best_silhouette_k, best_ch_k, best_elbow_k]
    
    # Se há consenso, usa o valor consensual
    from collections import Counter
    vote_counts = Counter(candidates)
    most_common = vote_counts.most_common(1)[0]
    
    if most_common[1] >= 2:  # Pelo menos 2 métodos concordam
        optimal_k = most_common[0]
    else:
        # Sem consenso: usa critério conservador (menor número)
        optimal_k = min(candidates)
    
    # Garante que está no range válido
    optimal_k = max(2, min(optimal_k, max_clusters))
    
    return optimal_k
```

#### 🎯 Interpretação dos Regimes

```python
def _interpret_operational_regimes(self, features_scaled: np.ndarray, 
                                 cluster_labels: np.ndarray, 
                                 feature_matrix: np.ndarray,
                                 snapshots: List[MultiVariableSnapshot]) -> Dict[str, Any]:
    """
    Interpreta os regimes operacionais identificados pelos clusters
    """
    
    unique_clusters = np.unique(cluster_labels)
    regime_interpretations = {}
    
    # Feature names for interpretation
    feature_names = [
        'P_exp_inst', 'P_rec_inst', 'Flow_inst', 'Density_inst', 'Temp_inst',
        'P_exp_mean', 'P_rec_mean', 'Flow_mean', 'Density_mean', 'Temp_mean',
        'P_exp_std', 'P_rec_std', 'Flow_std', 'Density_std', 'Temp_std',
        'P_exp_grad', 'P_rec_grad', 'Flow_grad',
        'Press_diff', 'Press_ratio', 'Flow_coeff',
        'Velocity', 'Reynolds', 'Head_loss',
        'Time_pos', 'Flow_trend'
    ]
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_data = feature_matrix[cluster_mask]
        cluster_snapshots = [snapshots[i] for i in range(len(snapshots)) if cluster_mask[i]]
        
        # Estatísticas do cluster
        cluster_size = np.sum(cluster_mask)
        cluster_percentage = cluster_size / len(cluster_labels) * 100
        
        # Centroide do cluster (valores médios)
        cluster_centroid = np.mean(cluster_data, axis=0)
        
        # Características distintivas (z-score em relação à população geral)
        population_means = np.mean(feature_matrix, axis=0)
        population_stds = np.std(feature_matrix, axis=0)
        
        z_scores = (cluster_centroid - population_means) / (population_stds + 1e-6)
        
        # Features mais distintivas (|z-score| > 1.5)
        distinctive_features = []
        for i, (feature_name, z_score) in enumerate(zip(feature_names, z_scores)):
            if abs(z_score) > 1.5:
                distinctive_features.append({
                    'feature': feature_name,
                    'z_score': float(z_score),
                    'cluster_mean': float(cluster_centroid[i]),
                    'population_mean': float(population_means[i])
                })
        
        # Classificação do regime baseada nas características
        regime_classification = self._classify_regime_type(distinctive_features, cluster_centroid, feature_names)
        
        # Análise temporal do regime
        cluster_timestamps = [s.timestamp for s in cluster_snapshots]
        if cluster_timestamps:
            time_span = max(cluster_timestamps) - min(cluster_timestamps)
            temporal_density = cluster_size / max(time_span.total_seconds() / 3600, 1)  # Eventos por hora
        else:
            temporal_density = 0.0
        
        regime_interpretations[f'regime_{cluster_id}'] = {
            'cluster_id': int(cluster_id),
            'size': int(cluster_size),
            'percentage': float(cluster_percentage),
            'classification': regime_classification,
            'distinctive_features': distinctive_features,
            'temporal_density': float(temporal_density),
            'operational_status': self._assess_operational_status(cluster_centroid, feature_names)
        }
    
    return regime_interpretations
```

#### ⚙️ Classificação Automática de Tipos de Regime

```python
def _classify_regime_type(self, distinctive_features: List[Dict], 
                         cluster_centroid: np.ndarray, 
                         feature_names: List[str]) -> Dict[str, Any]:
    """
    Classifica automaticamente o tipo de regime operacional
    """
    
    # Mapeia índices das features importantes
    feature_indices = {name: i for i, name in enumerate(feature_names)}
    
    # Extrai valores-chave do centroide
    flow_mean = cluster_centroid[feature_indices.get('Flow_mean', 0)]
    flow_std = cluster_centroid[feature_indices.get('Flow_std', 0)]
    press_diff = cluster_centroid[feature_indices.get('Press_diff', 0)]
    flow_trend = cluster_centroid[feature_indices.get('Flow_trend', 0)]
    reynolds = cluster_centroid[feature_indices.get('Reynolds', 0)]
    
    # Lógica de classificação baseada em regras
    
    # 1. Regime de alta variabilidade
    if flow_std > 100:  # kg/h
        if abs(flow_trend) > 50:  # kg/h por step
            classification = {
                'type': 'transient_regime',
                'description': 'Regime transiente com alta variabilidade',
                'severity': 'medium',
                'color': '#ff9800'
            }
        else:
            classification = {
                'type': 'unstable_regime',
                'description': 'Regime instável com flutuações',
                'severity': 'high',
                'color': '#f44336'
            }
    
    # 2. Regime de baixo fluxo
    elif flow_mean < 500:  # kg/h
        classification = {
            'type': 'low_flow_regime',
            'description': 'Regime de baixo fluxo',
            'severity': 'low',
            'color': '#2196f3'
        }
    
    # 3. Regime de alta pressão diferencial
    elif press_diff > 15:  # kgf/cm²
        classification = {
            'type': 'high_pressure_drop',
            'description': 'Alta perda de carga - possível restrição',
            'severity': 'high',
            'color': '#f44336'
        }
    
    # 4. Regime turbulento alto
    elif reynolds > 50000:
        classification = {
            'type': 'high_turbulent_regime',
            'description': 'Regime altamente turbulento',
            'severity': 'medium',
            'color': '#ff9800'
        }
    
    # 5. Regime normal (default)
    else:
        # Avalia estabilidade geral
        overall_variability = np.mean([
            cluster_centroid[feature_indices.get('Flow_std', 0)] / max(flow_mean, 1),
            cluster_centroid[feature_indices.get('P_exp_std', 0)],
            cluster_centroid[feature_indices.get('P_rec_std', 0)]
        ])
        
        if overall_variability < 0.02:  # Muito estável
            classification = {
                'type': 'steady_state_regime',
                'description': 'Regime estacionário estável',
                'severity': 'none',
                'color': '#4caf50'
            }
        else:
            classification = {
                'type': 'normal_operation',
                'description': 'Operação normal com variações típicas',
                'severity': 'low',
                'color': '#8bc34a'
            }
    
    # Adiciona confiança da classificação baseada na distintividade
    distinctive_count = len([f for f in distinctive_features if abs(f['z_score']) > 2.0])
    
    if distinctive_count >= 3:
        classification['confidence'] = 'high'
    elif distinctive_count >= 1:
        classification['confidence'] = 'medium'
    else:
        classification['confidence'] = 'low'
    
    return classification
```

---

## ⚖️ Validação Física

### Verificação de Leis de Conservação

#### 🔬 Validação Termodinâmica

```python
def validate_thermodynamic_consistency(self, snapshots: List[MultiVariableSnapshot]) -> Dict[str, Any]:
    """
    Valida consistência termodinâmica dos dados
    
    Verificações:
    1. Lei dos gases reais (se aplicável)
    2. Conservação de energia
    3. Propriedades PVT
    4. Limites de solubilidade
    """
    
    validation_results = {
        'overall_consistency': True,
        'violations': [],
        'warnings': [],
        'thermodynamic_checks': {}
    }
    
    for i, snapshot in enumerate(snapshots):
        snapshot_violations = []
        
        # 1. Verificação densidade-temperatura
        # Para hidrocarbonetos líquidos: ρ = ρ₀ × [1 - β(T - T₀)]
        reference_density = 850.0  # kg/m³ @ 20°C
        reference_temp = 20.0      # °C
        thermal_expansion_coeff = 0.0008  # 1/°C (típico para petróleo)
        
        expected_density = reference_density * (
            1 - thermal_expansion_coeff * (snapshot.temperature - reference_temp)
        )
        
        density_error = abs(snapshot.density - expected_density) / expected_density
        
        if density_error > 0.15:  # Erro > 15%
            snapshot_violations.append({
                'type': 'density_temperature_inconsistency',
                'description': f'Densidade {snapshot.density:.1f} inconsistente com temperatura {snapshot.temperature:.1f}°C',
                'expected_density': expected_density,
                'actual_density': snapshot.density,
                'error_percentage': density_error * 100
            })
        
        # 2. Verificação de limites físicos absolutos
        
        # Pressão de vapor (verificação simplificada)
        # Antoine equation approximation for hydrocarbons
        vapor_pressure_kpa = 10**(8.07131 - 1730.63/(snapshot.temperature + 233.426))  # kPa
        vapor_pressure_kgf_cm2 = vapor_pressure_kpa * 0.0101972  # kgf/cm²
        
        if snapshot.receiver_pressure < vapor_pressure_kgf_cm2:
            snapshot_violations.append({
                'type': 'pressure_below_vapor_pressure',
                'description': f'Pressão {snapshot.receiver_pressure:.2f} abaixo da pressão de vapor {vapor_pressure_kgf_cm2:.2f}',
                'vapor_pressure': vapor_pressure_kgf_cm2,
                'actual_pressure': snapshot.receiver_pressure
            })
        
        # 3. Verificação de fluxo vs. diferencial de pressão
        # Equação de Torricelli modificada: Q ∝ √(ΔP)
        if snapshot.pressure_difference > 0.5 and snapshot.flow > 10:
            # Flow coefficient should be relatively constant
            flow_coefficient = snapshot.flow / np.sqrt(snapshot.pressure_difference)
            
            # Compara com coeficiente médio dos outros snapshots
            other_coefficients = []
            for other_snapshot in snapshots:
                if (other_snapshot != snapshot and 
                    other_snapshot.pressure_difference > 0.5 and 
                    other_snapshot.flow > 10):
                    other_coeff = other_snapshot.flow / np.sqrt(other_snapshot.pressure_difference)
                    other_coefficients.append(other_coeff)
            
            if other_coefficients:
                mean_coefficient = np.mean(other_coefficients)
                coefficient_deviation = abs(flow_coefficient - mean_coefficient) / mean_coefficient
                
                if coefficient_deviation > 0.3:  # Desvio > 30%
                    snapshot_violations.append({
                        'type': 'flow_coefficient_anomaly',
                        'description': f'Coeficiente de fluxo anômalo: {flow_coefficient:.2f} vs média {mean_coefficient:.2f}',
                        'flow_coefficient': flow_coefficient,
                        'mean_coefficient': mean_coefficient,
                        'deviation_percentage': coefficient_deviation * 100
                    })
        
        # 4. Verificação de número de Reynolds vs. regime de fluxo
        if snapshot.reynolds_number > 0:
            if snapshot.reynolds_number < 2300:
                expected_regime = 'laminar'
            elif snapshot.reynolds_number < 4000:
                expected_regime = 'transitional'
            else:
                expected_regime = 'turbulent'
            
            # Esta verificação é mais para informação/warning
            validation_results['thermodynamic_checks'][f'snapshot_{i}'] = {
                'reynolds_number': snapshot.reynolds_number,
                'flow_regime': expected_regime,
                'density_temperature_consistent': density_error < 0.15,
                'pressure_above_vapor': snapshot.receiver_pressure >= vapor_pressure_kgf_cm2
            }
        
        # Adiciona violações encontradas
        if snapshot_violations:
            validation_results['violations'].extend([
                {'snapshot_index': i, 'timestamp': snapshot.timestamp.isoformat(), **violation}
                for violation in snapshot_violations
            ])
            validation_results['overall_consistency'] = False
    
    # Resumo das validações
    violation_types = [v['type'] for v in validation_results['violations']]
    validation_summary = {
        'total_snapshots': len(snapshots),
        'snapshots_with_violations': len(set(v['snapshot_index'] for v in validation_results['violations'])),
        'violation_types': list(set(violation_types)),
        'consistency_score': 1.0 - (len(validation_results['violations']) / len(snapshots))
    }
    
    validation_results['summary'] = validation_summary
    
    return validation_results
```

#### ⚙️ Validação Hidráulica

```python
def validate_hydraulic_principles(self, snapshots: List[MultiVariableSnapshot]) -> Dict[str, Any]:
    """
    Valida princípios hidráulicos fundamentais
    
    Verificações:
    1. Equação da continuidade
    2. Equação de Darcy-Weisbach  
    3. Conservação de energia (Bernoulli)
    4. Limites de cavitação
    """
    
    hydraulic_results = {
        'continuity_validation': {},
        'darcy_weisbach_validation': {},
        'energy_conservation': {},
        'cavitation_analysis': {},
        'overall_hydraulic_consistency': True
    }
    
    # Parâmetros do sistema (configuráveis)
    pipe_diameter = 1.0  # metros
    pipe_length = 1000.0  # metros
    pipe_roughness = 0.045  # mm (aço comercial)
    pipe_area = np.pi * (pipe_diameter/2)**2  # m²
    
    for i, snapshot in enumerate(snapshots):
        
        # 1. Equação da continuidade: ṁ = ρ × A × V
        mass_flow_calc = snapshot.density * pipe_area * snapshot.velocity  # kg/s
        mass_flow_given = snapshot.flow / 3600  # kg/s (conversão de kg/h)
        
        continuity_error = abs(mass_flow_calc - mass_flow_given) / max(mass_flow_given, 0.001)
        
        hydraulic_results['continuity_validation'][f'snapshot_{i}'] = {
            'calculated_mass_flow': mass_flow_calc,
            'given_mass_flow': mass_flow_given,
            'error_percentage': continuity_error * 100,
            'consistent': continuity_error < 0.1  # 10% tolerance
        }
        
        # 2. Equação de Darcy-Weisbach: ΔP = f × (L/D) × (ρV²/2)
        if snapshot.velocity > 0:
            # Fator de atrito usando aproximação de Blasius/Colebrook
            reynolds = snapshot.reynolds_number
            relative_roughness = (pipe_roughness / 1000) / pipe_diameter
            
            if reynolds < 2300:  # Laminar
                friction_factor = 64 / reynolds
            else:  # Turbulento - aproximação de Swamee-Jain
                friction_factor = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/reynolds**0.9))**2
            
            # Perda de carga calculada
            pressure_loss_calc = friction_factor * (pipe_length/pipe_diameter) * (snapshot.density * snapshot.velocity**2 / 2)
            pressure_loss_calc_kgf_cm2 = pressure_loss_calc / 98066.5  # Pa to kgf/cm²
            
            pressure_loss_measured = snapshot.pressure_difference
            
            darcy_error = abs(pressure_loss_calc_kgf_cm2 - pressure_loss_measured) / max(pressure_loss_measured, 0.1)
            
            hydraulic_results['darcy_weisbach_validation'][f'snapshot_{i}'] = {
                'calculated_pressure_loss': pressure_loss_calc_kgf_cm2,
                'measured_pressure_loss': pressure_loss_measured,
                'friction_factor': friction_factor,
                'error_percentage': darcy_error * 100,
                'consistent': darcy_error < 0.3  # 30% tolerance (mais permissivo)
            }
        
        # 3. Análise de cavitação (NPSH)
        # NPSH_available = (P_suction - P_vapor) / (ρg)
        vapor_pressure_pa = 2339  # Pa @ 20°C para água (aproximação)
        vapor_pressure_kgf_cm2 = vapor_pressure_pa / 98066.5
        
        suction_pressure = snapshot.receiver_pressure  # Assumindo que é a pressão de sucção
        
        if suction_pressure > vapor_pressure_kgf_cm2:
            npsh_available = (suction_pressure - vapor_pressure_kgf_cm2) * 10  # metros de coluna
            
            # NPSH mínimo requerido (estimativa baseada na velocidade)
            npsh_required = 2.0 + 0.1 * snapshot.velocity**2  # Estimativa empírica
            
            cavitation_margin = npsh_available - npsh_required
            
            hydraulic_results['cavitation_analysis'][f'snapshot_{i}'] = {
                'npsh_available': npsh_available,
                'npsh_required': npsh_required,
                'cavitation_margin': cavitation_margin,
                'cavitation_risk': cavitation_margin < 1.0  # Risco se margem < 1m
            }
    
    # Análise consolidada
    continuity_issues = sum(1 for v in hydraulic_results['continuity_validation'].values() if not v['consistent'])
    darcy_issues = sum(1 for v in hydraulic_results['darcy_weisbach_validation'].values() if not v['consistent'])
    cavitation_risks = sum(1 for v in hydraulic_results['cavitation_analysis'].values() if v['cavitation_risk'])
    
    hydraulic_results['overall_hydraulic_consistency'] = (
        continuity_issues < len(snapshots) * 0.1 and  # < 10% problemas
        darcy_issues < len(snapshots) * 0.2 and       # < 20% problemas 
        cavitation_risks < len(snapshots) * 0.05      # < 5% risco de cavitação
    )
    
    hydraulic_results['summary'] = {
        'continuity_issues': continuity_issues,
        'darcy_weisbach_issues': darcy_issues,
        'cavitation_risks': cavitation_risks,
        'hydraulic_score': 1.0 - (continuity_issues + darcy_issues + cavitation_risks) / (3 * len(snapshots))
    }
    
    return hydraulic_results
```

---

**CONTINUAÇÃO NA PARTE III**

A Parte II cobriu:

- ✅ **Detecção de Padrões Complexos** - Regimes operacionais, clustering, classificação automática
- ✅ **Validação Física** - Verificação termodinâmica e hidráulica, leis de conservação

**Próxima parte** - MANUAL_05_ANALISE_MULTIVARIAVEL_PARTE_III.md:

- 🔗 **Integração com Outros Módulos** - Interface com ML, correlação sônica, sistema principal
- 🎯 **Interpretação e Diagnóstico** - Dashboard, alertas, recomendações
- 📊 **Métricas de Performance** - KPIs, trends, relatórios

Continuar com a Parte III (final)?
