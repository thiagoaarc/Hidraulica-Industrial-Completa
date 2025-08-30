# Manual de Machine Learning - Sistema Hidráulico Industrial - Parte II

## 📋 Continuação - Análise Espectral e Temporal

---

## 🎵 Features Espectrais

### Análise de Domínio da Frequência

As features espectrais capturam características no **domínio da frequência**, essenciais para detectar padrões oscilatórios e componentes harmônicas indicativas de problemas operacionais.

#### 🔬 Implementação da Análise Espectral

```python
def _extract_spectral_features(self, features, feature_names, exp_pressure, rec_pressure):
    """
    Extrai features espectrais avançadas usando FFT
    
    Para pressão expedidor e recebedor:
    1. Energia em baixa frequência (0 - 10% Nyquist)
    2. Energia em média frequência (10% - 30% Nyquist)  
    3. Energia em alta frequência (30% - 100% Nyquist)
    4. Frequência dominante
    
    Total: 2 sinais × 4 features = 8 features espectrais
    """
    
    for signal, name in [(exp_pressure, 'exp_p'), (rec_pressure, 'rec_p')]:
        # FFT do sinal
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        magnitude = np.abs(fft_signal)
        
        # Define bandas de frequência
        low_freq_mask = np.abs(freqs) < 0.1      # 0-10% da frequência de Nyquist
        mid_freq_mask = (np.abs(freqs) >= 0.1) & (np.abs(freqs) < 0.3)  # 10-30%
        high_freq_mask = np.abs(freqs) >= 0.3    # 30-100%
        
        # Energia total para normalização
        total_energy = np.sum(magnitude)
        
        # Energias por banda (normalizadas)
        low_energy = np.sum(magnitude[low_freq_mask]) / max(total_energy, 1e-10)
        mid_energy = np.sum(magnitude[mid_freq_mask]) / max(total_energy, 1e-10)
        high_energy = np.sum(magnitude[high_freq_mask]) / max(total_energy, 1e-10)
        
        # Frequência dominante (apenas frequências positivas)
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        dominant_freq_idx = np.argmax(positive_magnitude)
        dominant_freq = positive_freqs[dominant_freq_idx]
        
        features.extend([low_energy, mid_energy, high_energy, dominant_freq])
        feature_names.extend([
            f'{name}_low_energy', f'{name}_mid_energy', f'{name}_high_energy', f'{name}_dom_freq'
        ])
```

#### 📊 Interpretação das Bandas Espectrais

##### **Energia em Baixa Frequência (0-10% Nyquist)**

```python
def interpret_low_frequency_energy(self, low_energy_ratio):
    """
    Interpreta energia em baixa frequência
    
    Características:
    - Tendências de longo prazo
    - Componente DC
    - Variações lentas de pressão
    """
    
    if low_energy_ratio > 0.8:
        return {
            'interpretation': 'Dominância de componentes lentas',
            'indication': 'Sistema estável, possível drift lento',
            'action': 'Monitorar tendências de longo prazo'
        }
    elif low_energy_ratio < 0.3:
        return {
            'interpretation': 'Baixa energia em componentes lentas',
            'indication': 'Possível instabilidade ou ruído dominante',
            'action': 'Investigar fontes de ruído de alta frequência'
        }
    else:
        return {
            'interpretation': 'Distribuição espectral equilibrada',
            'indication': 'Sistema com comportamento normal',
            'action': 'Monitoramento rotineiro'
        }
```

##### **Energia em Alta Frequência (30-100% Nyquist)**

```python
def interpret_high_frequency_energy(self, high_energy_ratio):
    """
    Interpreta energia em alta frequência
    
    Características:
    - Ruído de instrumentação
    - Turbulência local
    - Vibrações mecânicas
    - Possíveis vazamentos (jato turbulento)
    """
    
    if high_energy_ratio > 0.3:
        return {
            'interpretation': 'Alta energia em componentes rápidas',
            'indication': 'Possível turbulência, vazamento ou ruído',
            'severity': 'Alto',
            'action': 'Investigação imediata recomendada'
        }
    elif high_energy_ratio > 0.15:
        return {
            'interpretation': 'Energia moderada em alta frequência',
            'indication': 'Possível início de instabilidade',
            'severity': 'Moderado',
            'action': 'Monitoramento intensificado'
        }
    else:
        return {
            'interpretation': 'Baixa energia em alta frequência',
            'indication': 'Sistema estável, ruído controlado',
            'severity': 'Baixo',
            'action': 'Monitoramento normal'
        }
```

#### 🎼 Análise de Harmônicos

```python
def harmonic_analysis(self, signal, fundamental_freq=None):
    """
    Análise de conteúdo harmônico para detecção de padrões periódicos
    
    Detecta:
    - Frequência fundamental
    - Harmônicos (2f, 3f, 4f, ...)
    - Distorção harmônica total (THD)
    - Componentes inter-harmônicas
    """
    
    # FFT do sinal
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))
    magnitude = np.abs(fft_signal)
    
    # Encontra frequência fundamental se não fornecida
    if fundamental_freq is None:
        # Pico de maior magnitude (excluindo DC)
        positive_freqs = freqs[1:len(freqs)//2]  # Exclui DC e negativos
        positive_magnitude = magnitude[1:len(magnitude)//2]
        fundamental_idx = np.argmax(positive_magnitude)
        fundamental_freq = positive_freqs[fundamental_idx]
    
    # Localiza harmônicos
    harmonics = {}
    total_power = np.sum(magnitude**2)
    fundamental_power = 0
    
    for n in range(1, 6):  # Até 5º harmônico
        harmonic_freq = n * fundamental_freq
        
        # Encontra bin mais próximo da frequência harmônica
        harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
        harmonic_magnitude = magnitude[harmonic_idx]
        harmonic_power = harmonic_magnitude**2
        
        if n == 1:
            fundamental_power = harmonic_power
        
        harmonics[f'harmonic_{n}'] = {
            'frequency': freqs[harmonic_idx],
            'magnitude': harmonic_magnitude,
            'power_ratio': harmonic_power / max(total_power, 1e-10)
        }
    
    # Distorção Harmônica Total (THD)
    harmonic_powers = sum([h['magnitude']**2 for n, h in harmonics.items() if n != 'harmonic_1'])
    thd = np.sqrt(harmonic_powers / max(fundamental_power, 1e-10)) * 100  # Percentual
    
    return {
        'fundamental_frequency': fundamental_freq,
        'harmonics': harmonics,
        'thd_percent': thd,
        'spectral_quality': 'clean' if thd < 5 else 'distorted' if thd < 15 else 'heavily_distorted'
    }
```

---

## ⏱️ Features de Estabilidade Temporal

### Análise de Persistência e Flutuações

As features temporais quantificam a **estabilidade** dos sinais ao longo do tempo, usando janelas móveis para detectar mudanças graduais.

#### 📈 Implementação da Análise Temporal

```python
def _extract_temporal_features(self, features, feature_names, exp_pressure, rec_pressure, flow):
    """
    Extrai features de estabilidade temporal usando janelas móveis
    
    Para pressão expedidor, pressão recebedor e fluxo:
    1. Estabilidade média em janela móvel
    2. Máximo coeficiente de variação local
    
    Total: 3 sinais × 2 features = 6 features temporais
    """
    
    for signal, name in [(exp_pressure, 'exp_p'), (rec_pressure, 'rec_p'), (flow, 'flow')]:
        # Janela móvel adaptativa
        window_size = min(10, len(signal)//4)
        
        if window_size > 1:
            stabilities = []
            cv_locals = []  # Coeficientes de variação locais
            
            # Análise em janelas deslizantes
            for i in range(len(signal) - window_size + 1):
                window_data = signal[i:i + window_size]
                
                # Estabilidade = 1 / (1 + coeficiente de variação)
                mean_window = np.mean(window_data)
                std_window = np.std(window_data)
                cv = std_window / max(abs(mean_window), 1e-6)  # Coef. variação
                
                stability = 1.0 / (1.0 + cv)
                stabilities.append(stability)
                cv_locals.append(cv)
            
            stability_mean = np.mean(stabilities)
            cv_max = np.max(cv_locals)
        else:
            # Dados insuficientes para janela móvel
            stability_mean = 1.0  # Assumir estável
            cv_max = 0.0
        
        features.extend([stability_mean, cv_max])
        feature_names.extend([f'{name}_stability', f'{name}_cv_max'])
```

#### 🎯 Métricas de Estabilidade

##### **Coeficiente de Variação Local**

```
CV = σ_window / |μ_window|
```

- **Interpretação**: Variabilidade relativa em cada janela
- **Aplicação**: Detecta regiões de instabilidade localizada
- **Vazamentos**: CV elevado indica flutuações características

##### **Índice de Estabilidade**

```
Stability = 1 / (1 + CV)
```

- **Valores**: 0 (instável) a 1 (perfeitamente estável)
- **Interpretação**: Resistência a flutuações
- **Aplicação**: Quantifica qualidade operacional

#### 🔍 Análise de Mudanças de Regime

```python
def detect_regime_changes(self, signal, min_regime_length=20):
    """
    Detecta mudanças de regime operacional usando CUSUM e análise de variância
    
    Identifica:
    - Pontos de mudança na média
    - Pontos de mudança na variância  
    - Duração de cada regime
    - Caracterização dos regimes
    """
    
    # CUSUM para detecção de mudança na média
    mean_baseline = np.mean(signal)
    cusum_pos = np.zeros(len(signal))
    cusum_neg = np.zeros(len(signal))
    
    drift = 0.01 * np.std(signal)  # Drift mínimo para detecção
    threshold = 3 * np.std(signal)  # Threshold de detecção
    
    change_points_mean = []
    
    for i in range(1, len(signal)):
        # CUSUM acumulativo
        cusum_pos[i] = max(0, cusum_pos[i-1] + signal[i] - mean_baseline - drift)
        cusum_neg[i] = max(0, cusum_neg[i-1] - signal[i] + mean_baseline - drift)
        
        # Detecção de mudança
        if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
            change_points_mean.append(i)
            # Reset CUSUM
            cusum_pos[i] = 0
            cusum_neg[i] = 0
    
    # Análise de mudança na variância usando F-test em janelas
    change_points_variance = []
    window_size = min(50, len(signal)//5)
    
    for i in range(window_size, len(signal) - window_size):
        # Janelas antes e depois do ponto candidato
        before_window = signal[i-window_size:i]
        after_window = signal[i:i+window_size]
        
        # F-test para igualdade de variâncias
        var_before = np.var(before_window)
        var_after = np.var(after_window)
        
        f_ratio = max(var_before, var_after) / max(min(var_before, var_after), 1e-10)
        
        # Threshold empírico para F-test
        if f_ratio > 2.5:  # Mudança significativa na variância
            change_points_variance.append(i)
    
    # Consolida pontos de mudança
    all_changes = sorted(set(change_points_mean + change_points_variance))
    
    # Remove pontos muito próximos
    filtered_changes = []
    for change in all_changes:
        if not filtered_changes or change - filtered_changes[-1] > min_regime_length:
            filtered_changes.append(change)
    
    # Caracteriza cada regime
    regimes = []
    start_points = [0] + filtered_changes
    end_points = filtered_changes + [len(signal)]
    
    for i, (start, end) in enumerate(zip(start_points, end_points)):
        regime_data = signal[start:end]
        
        regimes.append({
            'regime_id': i + 1,
            'start_index': start,
            'end_index': end,
            'duration': end - start,
            'mean_level': np.mean(regime_data),
            'std_level': np.std(regime_data),
            'trend': np.polyfit(range(len(regime_data)), regime_data, 1)[0] if len(regime_data) > 1 else 0.0
        })
    
    return {
        'n_regimes': len(regimes),
        'change_points': filtered_changes,
        'regimes': regimes,
        'stability_score': 1.0 / max(1, len(filtered_changes))  # Mais mudanças = menos estável
    }
```

---

## 🔗 Relações Entre Variáveis

### Correlações Cruzadas Multivariáveis

As features relacionais capturam **interações entre diferentes variáveis**, essenciais para compreender o comportamento sistêmico do processo hidráulico.

#### 🎯 Implementação das Correlações Cruzadas

```python
def _extract_relational_features(self, features, feature_names, exp_pressure, rec_pressure, flow, density, temperature):
    """
    Extrai features baseadas em correlações entre pares de variáveis
    
    Pares analisados:
    1. (Diferença de pressão) vs (Fluxo)
    2. (Densidade) vs (Temperatura)  
    3. (Fluxo) vs (Temperatura)
    4. (Diferença de pressão) vs (Densidade)
    
    Total: 4 pares × 2 features = 8 features relacionais
    """
    
    # Calcula diferença de pressão (indicador de força motriz)
    pressure_diff = exp_pressure - rec_pressure
    
    # Define pares de variáveis para análise
    variable_pairs = [
        (pressure_diff, flow, 'press_diff_flow'),      # Lei de Darcy-Weisbach
        (density, temperature, 'density_temp'),        # Relação termodinâmica
        (flow, temperature, 'flow_temp'),              # Efeito térmico no fluxo
        (pressure_diff, density, 'press_diff_density') # Efeito barométrico
    ]
    
    correlations = []
    correlation_names = []
    
    for var1, var2, name in variable_pairs:
        if len(var1) > 1 and len(var2) > 1:
            # Correlação de Pearson
            try:
                corr_matrix = np.corrcoef(var1, var2)
                correlation = corr_matrix[0, 1]
                
                # Trata NaN (variáveis constantes)
                if np.isnan(correlation):
                    correlation = 0.0
                
            except:
                correlation = 0.0
            
            # Correlação cruzada com delay (correlação máxima com lag)
            try:
                cross_corr = signal.correlate(var1 - np.mean(var1), var2 - np.mean(var2), mode='full')
                cross_corr_normalized = cross_corr / (len(var1) * np.std(var1) * np.std(var2))
                max_cross_corr = np.max(np.abs(cross_corr_normalized))
                
                if np.isnan(max_cross_corr):
                    max_cross_corr = 0.0
                    
            except:
                max_cross_corr = 0.0
        else:
            correlation = 0.0
            max_cross_corr = 0.0
        
        correlations.extend([correlation, max_cross_corr])
        correlation_names.extend([f'corr_{name}', f'xcorr_max_{name}'])
    
    features.extend(correlations)
    feature_names.extend(correlation_names)
```

#### 📊 Interpretação Física das Correlações

##### **Diferença de Pressão vs Fluxo**

```python
def interpret_pressure_flow_correlation(self, correlation):
    """
    Interpreta correlação entre diferença de pressão e fluxo
    
    Baseado na Equação de Darcy-Weisbach:
    ΔP = f × (L/D) × (ρV²/2)
    
    Onde ΔP ∝ Q² (fluxo ao quadrado)
    """
    
    if correlation > 0.8:
        return {
            'interpretation': 'Correlação forte e positiva',
            'physical_meaning': 'Comportamento hidráulico normal',
            'flow_regime': 'Turbulento bem estabelecido',
            'system_health': 'Bom'
        }
    elif correlation > 0.5:
        return {
            'interpretation': 'Correlação moderada',
            'physical_meaning': 'Possível transição de regime ou perdas adicionais',
            'flow_regime': 'Transição laminar-turbulento',
            'system_health': 'Atenção'
        }
    elif correlation < 0.3:
        return {
            'interpretation': 'Correlação fraca ou ausente',
            'physical_meaning': 'Possível vazamento, bloqueio ou instrumentação defeituosa',
            'flow_regime': 'Indeterminado',
            'system_health': 'Crítico'
        }
```

##### **Densidade vs Temperatura**

```python
def interpret_density_temperature_correlation(self, correlation):
    """
    Interpreta correlação entre densidade e temperatura
    
    Baseado na equação de estado:
    ρ = ρ₀ × [1 - β(T - T₀)]
    
    Onde β é o coeficiente de expansão térmica
    """
    
    expected_correlation = -0.7  # Correlação negativa esperada
    
    if correlation < -0.6:
        return {
            'interpretation': 'Correlação negativa forte (esperada)',
            'physical_meaning': 'Expansão térmica normal do fluido',
            'fluid_behavior': 'Conforme esperado',
            'measurement_quality': 'Boa'
        }
    elif correlation > -0.3:
        return {
            'interpretation': 'Correlação negativa fraca ou positiva',
            'physical_meaning': 'Possível mistura de fluidos, mudança de composição ou erro de instrumentação',
            'fluid_behavior': 'Anômalo',
            'measurement_quality': 'Questionável'
        }
```

#### 🔍 Análise de Causalidade Temporal

```python
def granger_causality_analysis(self, var1, var2, max_lag=10):
    """
    Teste de Causalidade de Granger entre duas variáveis
    
    Determina se var1 "causa" var2 no sentido estatístico:
    - var1 ajuda a predizer var2 melhor que apenas o histórico de var2
    """
    
    from statsmodels.tsa.stattools import grangercausalitytests
    
    try:
        # Prepara dados para teste
        data = np.column_stack([var2, var1])  # [y, x] - ordem importante
        
        # Executa teste de Granger
        results = grangercausalitytests(data, max_lag, verbose=False)
        
        # Extrai p-valores para diferentes lags
        p_values = {}
        f_statistics = {}
        
        for lag in range(1, max_lag + 1):
            if lag in results:
                test_result = results[lag][0]['ssr_ftest']
                p_values[lag] = test_result[1]  # p-valor
                f_statistics[lag] = test_result[0]  # estatística F
        
        # Determina o melhor lag (menor p-valor)
        best_lag = min(p_values.keys(), key=lambda k: p_values[k])
        best_p_value = p_values[best_lag]
        
        # Interpretação
        significance_level = 0.05
        is_causal = best_p_value < significance_level
        
        return {
            'is_causal': is_causal,
            'best_lag': best_lag,
            'p_value': best_p_value,
            'f_statistic': f_statistics[best_lag],
            'all_lags': {
                'p_values': p_values,
                'f_statistics': f_statistics
            },
            'interpretation': 'var1 Granger-causa var2' if is_causal else 'Sem causalidade detectada'
        }
        
    except Exception as e:
        return {
            'error': f'Erro no teste de Granger: {str(e)}',
            'is_causal': False,
            'p_value': 1.0
        }
```

---

## 🎪 Integração de Features

### Vetor de Features Completo

Ao final do processo de extração, o sistema produz um **vetor de 81 features** altamente especializadas:

#### 📋 Resumo das Features por Categoria

| Categoria | Quantidade | Descrição | Aplicação Principal |
|-----------|------------|-----------|-------------------|
| **Estatísticas** | 45 | Momentos, percentis, tendências | Caracterização básica |
| **Gradientes** | 9 | Derivadas temporais | Detecção de transientes |
| **Correlação Cruzada** | 5 | Análise sônica integrada | Propagação de ondas |
| **Espectrais** | 8 | Conteúdo de frequências | Padrões oscilatórios |
| **Relacionais** | 8 | Correlações entre variáveis | Comportamento sistêmico |
| **Temporais** | 6 | Estabilidade e persistência | Mudanças de regime |
| **TOTAL** | **81** | **Features altamente especializadas** | **Detecção completa** |

#### 🧮 Vetor de Features Normalizado

```python
def prepare_feature_vector_for_ml(self, features_raw):
    """
    Prepara vetor de features para algoritmos de ML
    
    Processos:
    1. Normalização Z-score
    2. Tratamento de outliers
    3. Preenchimento de valores faltantes
    4. Validação de consistência
    """
    
    # Remove NaN e infinitos
    features_clean = np.array(features_raw)
    nan_mask = np.isnan(features_clean) | np.isinf(features_clean)
    features_clean[nan_mask] = 0.0  # Ou mediana/interpolação
    
    # Detecção de outliers extremos (> 5 sigma)
    outlier_mask = np.abs(features_clean) > 5 * np.std(features_clean)
    features_clean[outlier_mask] = np.clip(
        features_clean[outlier_mask],
        -5 * np.std(features_clean),
        5 * np.std(features_clean)
    )
    
    # Normalização usando scaler treinado
    if hasattr(self.scaler, 'scale_'):  # Scaler já foi treinado
        features_normalized = self.scaler.transform(features_clean.reshape(1, -1))[0]
    else:
        # Primeira vez - fit e transform
        features_normalized = features_clean  # Retorna sem normalizar
    
    # Validação final
    assert len(features_normalized) == 81, f"Esperadas 81 features, obtidas {len(features_normalized)}"
    assert not np.any(np.isnan(features_normalized)), "Features contêm NaN após normalização"
    
    return features_normalized
```

---

**CONTINUAÇÃO NA PARTE III**

A Parte II cobriu detalhadamente:

- ✅ **Features Espectrais** - Análise FFT, bandas de frequência, harmônicos
- ✅ **Features Temporais** - Estabilidade, janelas móveis, mudanças de regime  
- ✅ **Features Relacionais** - Correlações cruzadas, causalidade de Granger
- ✅ **Integração de Features** - Vetor final de 81 features

**Próxima parte** - MANUAL_04_MACHINE_LEARNING_PARTE_III.md:

- 🔄 **Algoritmos ML** - Isolation Forest, Random Forest, SVM, DBSCAN
- 🧠 **Treinamento Adaptativo** - Retreino automático, threshold dinâmico
- 📊 **Análise PCA** - Redução de dimensionalidade, componentes principais
- 🎯 **Predição e Fusão** - Combinação de modelos, confidence scoring

Continuar com a Parte III?
