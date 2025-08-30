# Manual da Interface Gráfica do Sistema Hidráulico Industrial

## 📋 Índice da Interface

1. [Visão Geral da Interface](#visão-geral-da-interface)
2. [Janela Principal](#janela-principal)
3. [Barra de Ferramentas](#barra-de-ferramentas)
4. [Painel de Controles](#painel-de-controles)
5. [Abas de Visualização](#abas-de-visualização)
6. [Diálogos de Configuração](#diálogos-de-configuração)
7. [Sistema de Status](#sistema-de-status)
8. [Componentes Assíncronos](#componentes-assíncronos)

---

## 🖥️ Visão Geral da Interface

### Arquitetura da Interface

A interface é construída com **PyQt6**, oferecendo uma experiência industrial robusta e responsiva:

```
┌─────────────────────────────────────────────────────────┐
│                    BARRA DE FERRAMENTAS                 │
├─────────────────┬───────────────────────────────────────┤
│                 │                                       │
│   CONTROLES     │           VISUALIZAÇÕES               │
│                 │                                       │
│   - Sistema     │     ┌─────────────────────────────┐   │
│   - Análise     │     │        ABAS GRÁFICAS       │   │
│   - ML          │     │                             │   │
│   - Status      │     │  • Sinais Principais       │   │
│   - Progresso   │     │  • Machine Learning        │   │
│                 │     │  • Status Operacional      │   │
│                 │     │  • Análise de Ruídos       │   │
│                 │     │  • Perfil Hidráulico       │   │
│                 │     │  • Análise de Ondas        │   │
│                 │     │  • Filtros Temporais       │   │
│                 │     │  • Configurações           │   │
│                 │     │  • Monitor do Sistema      │   │
│                 │     │  • Relatórios              │   │
│                 │     │  • 📁 Arquivos             │   │
│                 │     └─────────────────────────────┘   │
├─────────────────┴───────────────────────────────────────┤
│                    BARRA DE STATUS                      │
└─────────────────────────────────────────────────────────┘
```

### Características da Interface

#### 🎨 Design Industrial

- **Tema Fusion**: Aparência consistente multiplataforma
- **Cores Profissionais**: Azul/cinza com alertas em vermelho/laranja
- **Ícones Unicode**: Símbolos universais (▶ ⏹ 📊 ⚙️ 📁)
- **Layout Responsivo**: Redimensionamento automático

#### ⚡ Performance

- **Interface Assíncrona**: Sem travamentos durante processamento
- **Atualizações em Tempo Real**: Gráficos streaming
- **Lazy Loading**: Carregamento otimizado de componentes
- **Cache Visual**: Otimização de renderização

---

## 🏠 Janela Principal

### Classe `HydraulicAnalysisMainWindow`

A janela principal herda de `QMainWindow` e centraliza toda a funcionalidade:

```python
class HydraulicAnalysisMainWindow(QMainWindow):
    """Janela principal do sistema de análise hidráulica"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema Industrial de Análise Hidráulica v2.0")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Configuração inicial
        self.current_system = None
        self.processor = None
        self.plot_data = {}
        self.file_loader = None
        
        # Interface
        self.setup_ui()
        self.setup_plots()
        self.setup_connections()
        
        # Status inicial
        self.log_status("Sistema iniciado - Configure um sistema para começar")
```

#### 📐 Dimensões e Layout

- **Tamanho Mínimo**: 1400x900 pixels
- **Tamanho Padrão**: 1600x1000 pixels
- **Layout Principal**: `QSplitter` horizontal (350px controles + 1250px gráficos)
- **Redimensionamento**: Automático e proporcional

---

## 🛠️ Barra de Ferramentas

### Componentes da Barra Superior

A barra de ferramentas oferece acesso rápido às funções principais:

```python
def setup_toolbar(self):
    """Configura barra de ferramentas principal"""
    toolbar_layout = QHBoxLayout()
    
    # Status do sistema
    self.system_label = QLabel("Nenhum sistema carregado")
    self.system_label.setStyleSheet("font-weight: bold; color: #ff6b35;")
    
    # Botões principais
    self.new_system_btn = QPushButton("Novo Sistema")
    self.load_system_btn = QPushButton("Carregar Sistema")  
    self.load_data_btn = QPushButton("Carregar Dados")
    
    # Separador visual
    toolbar_layout.addWidget(QLabel("|"))
    
    # Controles de simulação
    self.start_simulation_btn = QPushButton("▶ Simular")
    self.stop_simulation_btn = QPushButton("⏹ Parar")
    
    # Cenários de teste
    self.scenario_combo = QComboBox()
    self.scenario_combo.addItems([
        'normal', 'leak_gradual', 'leak_sudden',
        'blockage', 'pump_failure', 'sensor_drift'
    ])
```

#### 🎯 Botões e Funções

##### **Novo Sistema**

- **Função**: Abre diálogo de configuração para novo sistema
- **Ícone**: Texto simples
- **Status**: Sempre habilitado
- **Ação**: `self.new_system()`

```python
def new_system(self):
    """Cria novo sistema via diálogo de configuração"""
    dialog = SystemConfigDialog(self)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        config = dialog.get_configuration()
        self.on_system_configured(config)
```

##### **Carregar Sistema**

- **Função**: Carrega sistema existente da base de dados
- **Ícone**: Texto simples  
- **Status**: Sempre habilitado
- **Ação**: `self.load_system()`

```python
def load_system(self):
    """Carrega sistema existente da base de dados"""
    systems = database.get_all_systems()
    if not systems:
        QMessageBox.information(self, "Informação", 
                               "Nenhum sistema encontrado. Crie um novo sistema primeiro.")
        return
    
    # Dialog de seleção
    system_names = [f"{s['name']} ({s['system_id']}) - {s['location']}" for s in systems]
    system_name, ok = QInputDialog.getItem(
        self, "Carregar Sistema", "Selecione o sistema:", system_names, 0, False
    )
```

##### **Carregar Dados**

- **Função**: Carrega arquivos de dados (Excel/CSV)
- **Ícone**: Texto simples
- **Status**: Habilitado apenas com sistema configurado
- **Ação**: `self.load_data_file()`

```python
def load_data_file(self):
    """Carrega arquivo de dados para análise"""
    if not self.current_system:
        QMessageBox.warning(self, "Aviso", "Configure um sistema primeiro!")
        return
    
    file_path, _ = QFileDialog.getOpenFileName(
        self, "Carregar Dados", "", 
        "Arquivos de dados (*.csv *.xlsx *.xls);;Todos os arquivos (*)"
    )
```

##### **▶ Simular**  

- **Função**: Inicia simulação com cenários predefinidos
- **Ícone**: ▶ (play)
- **Estilo**: Fundo verde `#4CAF50`
- **Status**: Habilitado com sistema e dados carregados
- **Ação**: `self.start_simulation()`

##### **⏹ Parar**

- **Função**: Para simulação em execução
- **Ícone**: ⏹ (stop)
- **Estilo**: Fundo vermelho `#f44336`
- **Status**: Habilitado apenas durante simulação
- **Ação**: `self.stop_simulation()`

#### 🎛️ ComboBox de Cenários

Permite seleção de cenários de teste:

```python
scenarios = {
    'normal': "Operação Normal",
    'leak_gradual': "Vazamento Gradual", 
    'leak_sudden': "Vazamento Súbito",
    'blockage': "Bloqueio Parcial",
    'pump_failure': "Falha de Bomba",
    'sensor_drift': "Deriva de Sensor"
}
```

---

## 🎛️ Painel de Controles

### Layout do Painel Esquerdo

O painel de controles ocupa 350px e contém todos os controles de análise:

```python
def setup_control_panel(self):
    """Configura painel de controles à esquerda"""
    control_widget = QWidget()
    control_widget.setMaximumWidth(350)
    control_widget.setMinimumWidth(300)
    
    control_layout = QVBoxLayout(control_widget)
    
    # Grupos de controles
    control_layout.addWidget(self.create_system_info_group())
    control_layout.addWidget(self.create_analysis_group())  
    control_layout.addWidget(self.create_ml_group())
    control_layout.addWidget(self.create_metrics_group())
    control_layout.addWidget(self.create_learning_group())
    control_layout.addWidget(self.create_progress_group())
    
    control_layout.addStretch()  # Expansão no final
```

### 📊 Grupo Informações do Sistema

```python
def create_system_info_group(self):
    """Grupo com informações do sistema atual"""
    info_group = QGroupBox("Informações do Sistema")
    info_layout = QVBoxLayout(info_group)
    
    # Labels informativos
    self.system_info_label = QLabel("Sistema: Não configurado")
    self.system_info_label.setWordWrap(True)
    self.system_info_label.setStyleSheet("font-weight: bold; color: #2196F3;")
    
    self.sensor_info_label = QLabel("Sensores: -")
    self.distance_info_label = QLabel("Distância: -") 
    self.fluid_info_label = QLabel("Fluido: -")
    
    info_layout.addWidget(self.system_info_label)
    info_layout.addWidget(self.sensor_info_label)
    info_layout.addWidget(self.distance_info_label)
    info_layout.addWidget(self.fluid_info_label)
    
    return info_group
```

### ⚙️ Grupo Controles de Análise

```python
def create_analysis_group(self):
    """Controles principais de análise"""
    analysis_group = QGroupBox("Controles de Análise")
    analysis_layout = QGridLayout(analysis_group)
    
    # Modo de análise
    analysis_layout.addWidget(QLabel("Modo:"), 0, 0)
    self.analysis_mode_combo = QComboBox()
    self.analysis_mode_combo.addItems(['Tempo Real', 'Histórico', 'Batch'])
    self.analysis_mode_combo.currentTextChanged.connect(self.analysis_mode_changed)
    analysis_layout.addWidget(self.analysis_mode_combo, 0, 1)
    
    # Controle de velocidade
    analysis_layout.addWidget(QLabel("Velocidade:"), 1, 0)
    self.speed_slider = QSlider(Qt.Orientation.Horizontal)
    self.speed_slider.setRange(1, 100)
    self.speed_slider.setValue(10)
    self.speed_slider.valueChanged.connect(self.speed_changed)
    analysis_layout.addWidget(self.speed_slider, 1, 1)
    
    self.speed_label = QLabel("10x")
    analysis_layout.addWidget(self.speed_label, 1, 2)
    
    # Sensibilidade ML
    analysis_layout.addWidget(QLabel("Sensibilidade ML:"), 2, 0)
    self.sensitivity_combo = QComboBox()
    self.sensitivity_combo.addItems(['Baixa', 'Média', 'Alta'])
    self.sensitivity_combo.setCurrentText('Média')
    self.sensitivity_combo.currentTextChanged.connect(self.sensitivity_changed)
    analysis_layout.addWidget(self.sensitivity_combo, 2, 1)
    
    # Botões de ação
    self.start_analysis_btn = QPushButton("Iniciar Análise")
    self.start_analysis_btn.clicked.connect(self.start_analysis)
    self.start_analysis_btn.setEnabled(False)
    self.start_analysis_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
    analysis_layout.addWidget(self.start_analysis_btn, 3, 0)
    
    self.stop_analysis_btn = QPushButton("Parar Análise") 
    self.stop_analysis_btn.clicked.connect(self.stop_analysis)
    self.stop_analysis_btn.setEnabled(False)
    self.stop_analysis_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
    analysis_layout.addWidget(self.stop_analysis_btn, 3, 1)
    
    return analysis_group
```

#### 🎮 Controles Detalhados

##### **Modo de Análise**

- **Tempo Real**: Processamento contínuo de dados streaming
- **Histórico**: Análise de dados carregados de arquivos
- **Batch**: Processamento em lotes otimizado

##### **Slider de Velocidade**

- **Faixa**: 1x a 100x velocidade normal
- **Padrão**: 10x
- **Função**: Controla velocidade de simulação/reprodução
- **Callback**: `self.speed_changed(value)`

```python
def speed_changed(self, value):
    """Atualiza velocidade de análise"""
    self.speed_label.setText(f"{value}x")
    if hasattr(self, 'analysis_timer'):
        # Ajusta timer de análise
        new_interval = max(10, 1000 // value)
        self.analysis_timer.setInterval(new_interval)
```

##### **Sensibilidade ML**

- **Baixa**: 0.3 - Menos false positives
- **Média**: 0.5 - Balanceada (padrão)  
- **Alta**: 0.8 - Maior detecção de anomalias

### 🤖 Grupo Machine Learning

```python
def create_ml_group(self):
    """Controles específicos de Machine Learning"""
    ml_group = QGroupBox("Machine Learning")
    ml_layout = QVBoxLayout(ml_group)
    
    # Algoritmo selecionado
    ml_layout.addWidget(QLabel("Algoritmo:"))
    self.ml_algorithm_combo = QComboBox()
    self.ml_algorithm_combo.addItems([
        'Isolation Forest',
        'Random Forest', 
        'SVM One-Class',
        'DBSCAN',
        'Ensemble'
    ])
    self.ml_algorithm_combo.currentTextChanged.connect(self.ml_algorithm_changed)
    ml_layout.addWidget(self.ml_algorithm_combo)
    
    # Features selecionadas
    features_group = QGroupBox("Features Ativas")
    features_layout = QVBoxLayout(features_group)
    
    self.feature_pressure_cb = QCheckBox("Pressão")
    self.feature_pressure_cb.setChecked(True)
    self.feature_flow_cb = QCheckBox("Vazão")
    self.feature_flow_cb.setChecked(True)
    self.feature_temp_cb = QCheckBox("Temperatura")
    self.feature_temp_cb.setChecked(True)
    self.feature_density_cb = QCheckBox("Densidade")
    self.feature_density_cb.setChecked(True)
    
    features_layout.addWidget(self.feature_pressure_cb)
    features_layout.addWidget(self.feature_flow_cb)
    features_layout.addWidget(self.feature_temp_cb)
    features_layout.addWidget(self.feature_density_cb)
    
    ml_layout.addWidget(features_group)
    
    return ml_group
```

### 📈 Grupo Métricas do Sistema

```python
def create_metrics_group(self):
    """Métricas em tempo real"""
    metrics_group = QGroupBox("Métricas do Sistema")
    metrics_layout = QGridLayout(metrics_group)
    
    # Labels de métricas  
    self.leak_probability_label = QLabel("Prob. Vazamento: 0.00%")
    self.leak_probability_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
    
    self.correlation_label = QLabel("Correlação: -") 
    self.system_status_label = QLabel("Status: Normal")
    self.confidence_label = QLabel("Confiança: -")
    
    metrics_layout.addWidget(QLabel("Detecção:"), 0, 0)
    metrics_layout.addWidget(self.leak_probability_label, 0, 1)
    
    metrics_layout.addWidget(QLabel("Correlação:"), 1, 0)
    metrics_layout.addWidget(self.correlation_label, 1, 1)
    
    metrics_layout.addWidget(QLabel("Sistema:"), 2, 0) 
    metrics_layout.addWidget(self.system_status_label, 2, 1)
    
    metrics_layout.addWidget(self.confidence_label, 3, 0, 1, 2)
    
    return metrics_group
```

#### 🎯 Métricas Monitoradas

##### **Probabilidade de Vazamento**

- **Faixa**: 0.00% a 100.00%
- **Cores**: Verde (<10%), Amarelo (10-50%), Vermelho (>50%)
- **Atualização**: Tempo real durante análise

##### **Correlação Sônica**

- **Faixa**: -1.0 a +1.0
- **Interpretação**: >0.8 = Normal, <0.3 = Possível anomalia

##### **Status do Sistema**

- **Estados**: Normal, Alerta, Crítico, Manutenção, Indeterminado
- **Cores**: Verde, Amarelo, Vermelho, Azul, Cinza

### 🎓 Grupo Aprendizado do Sistema

```python
def create_learning_group(self):
    """Controles de aprendizado e feedback"""
    learning_group = QGroupBox("Aprendizado do Sistema") 
    learning_layout = QVBoxLayout(learning_group)
    
    # Botão para reportar vazamento confirmado
    self.report_leak_btn = QPushButton("Reportar Vazamento Confirmado")
    self.report_leak_btn.clicked.connect(self.report_confirmed_leak)
    self.report_leak_btn.setEnabled(False)
    self.report_leak_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px;")
    learning_layout.addWidget(self.report_leak_btn)
    
    # Botão para retreinar modelo
    self.retrain_btn = QPushButton("Retreinar Modelo ML")
    self.retrain_btn.clicked.connect(self.retrain_ml_model)
    self.retrain_btn.setEnabled(False)
    self.retrain_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
    learning_layout.addWidget(self.retrain_btn)
    
    return learning_group
```

### ⏳ Grupo Progresso

```python
def create_progress_group(self):
    """Barra de progresso e status"""
    # Barra de progresso
    self.progress_bar = QProgressBar()
    self.progress_bar.setVisible(False)
    
    return self.progress_bar
```

---

## 📊 Abas de Visualização

### Sistema de Abas Principal

O sistema utiliza `QTabWidget` para organizar diferentes visualizações:

```python
def setup_plots(self):
    """Configura todas as abas de visualização"""
    self.plots_tab_widget = QTabWidget()
    
    # Criação das abas
    self.setup_signals_tab()           # Aba 1: Sinais Principais
    self.setup_ml_tab()               # Aba 2: Machine Learning  
    self.setup_operational_tab()      # Aba 3: Status Operacional
    self.setup_noise_tab()            # Aba 4: Análise de Ruídos
    self.setup_hydraulic_tab()        # Aba 5: Perfil Hidráulico
    self.setup_wave_tab()             # Aba 6: Análise de Ondas
    self.setup_filter_tab()           # Aba 7: Filtros Temporais
    self.setup_config_tab()           # Aba 8: Configurações
    self.setup_monitor_tab()          # Aba 9: Monitor do Sistema
    self.setup_report_tab()           # Aba 10: Relatórios
    self.setup_files_tab()            # Aba 11: 📁 Arquivos
```

### 📈 Aba 1: Sinais Principais

A aba mais importante, mostra os sinais fundamentais do sistema:

```python
def setup_signals_tab(self):
    """Aba dos sinais principais"""
    signals_widget = QWidget()
    signals_layout = QGridLayout(signals_widget)
    
    # Plot 1: Pressões (Expedidor e Recebedor)
    self.pressure_plot = PlotWidget(title="Pressões do Sistema")
    self.pressure_plot.setLabel('left', 'Pressão (kgf/cm²)')
    self.pressure_plot.setLabel('bottom', 'Tempo')
    self.pressure_plot.addLegend()
    self.pressure_plot.showGrid(x=True, y=True, alpha=0.3)
    
    # Curvas de pressão
    self.exp_pressure_curve = self.pressure_plot.plot(
        pen=mkPen('blue', width=2), name='Expedidor'
    )
    self.rec_pressure_curve = self.pressure_plot.plot(
        pen=mkPen('red', width=2), name='Recebedor'  
    )
    
    signals_layout.addWidget(self.pressure_plot, 0, 0)
    
    # Plot 2: Vazão
    self.flow_plot = PlotWidget(title="Vazão do Sistema")
    self.flow_plot.setLabel('left', 'Vazão (m³/h)')
    self.flow_plot.setLabel('bottom', 'Tempo')
    self.flow_plot.showGrid(x=True, y=True, alpha=0.3)
    
    self.flow_curve = self.flow_plot.plot(
        pen=mkPen('green', width=2), name='Vazão'
    )
    
    signals_layout.addWidget(self.flow_plot, 0, 1)
    
    # Plot 3: Densidade e Temperatura
    self.density_temp_plot = PlotWidget(title="Densidade e Temperatura")
    self.density_temp_plot.setLabel('left', 'Densidade (g/cm³) / Temperatura (°C)')
    self.density_temp_plot.setLabel('bottom', 'Tempo')
    self.density_temp_plot.addLegend()
    self.density_temp_plot.showGrid(x=True, y=True, alpha=0.3)
    
    self.density_curve = self.density_temp_plot.plot(
        pen=mkPen('magenta', width=2), name='Densidade'
    )
    self.temperature_curve = self.density_temp_plot.plot(
        pen=mkPen('red', width=2), name='Temperatura'
    )
    
    signals_layout.addWidget(self.density_temp_plot, 1, 0)
    
    # Plot 4: Correlação Cruzada
    self.correlation_plot = PlotWidget(title="Correlação Cruzada")
    self.correlation_plot.setLabel('left', 'Correlação')
    self.correlation_plot.setLabel('bottom', 'Delay (amostras)')
    self.correlation_plot.showGrid(x=True, y=True, alpha=0.3)
    
    self.correlation_curve = self.correlation_plot.plot(
        pen=mkPen('yellow', width=2)
    )
    
    signals_layout.addWidget(self.correlation_plot, 1, 1)
    
    self.plots_tab_widget.addTab(signals_widget, "Sinais Principais")
```

#### 📊 Gráficos da Aba Sinais Principais

##### **Gráfico de Pressões** (Posição: 0,0)

- **Eixo Y**: Pressão (kgf/cm²)
- **Eixo X**: Tempo (com formatação automática de data/hora)
- **Curvas**:
  - Expedidor (azul, linha sólida, largura 2px)
  - Recebedor (vermelha, linha sólida, largura 2px)
- **Grade**: Habilitada com transparência 30%
- **Legenda**: Posicionada automaticamente

##### **Gráfico de Vazão** (Posição: 0,1)

- **Eixo Y**: Vazão (m³/h)
- **Eixo X**: Tempo
- **Curva**: Verde, linha sólida, largura 2px
- **Funcionalidade**: Mostra vazão média entre expedidor e recebedor

##### **Gráfico Densidade/Temperatura** (Posição: 1,0)

- **Eixo Y**: Valores normalizados
- **Eixo X**: Tempo
- **Curvas**:
  - Densidade (magenta) - g/cm³
  - Temperatura (vermelha) - °C
- **Correlação**: Mostra relação inversa densidade-temperatura

##### **Gráfico de Correlação** (Posição: 1,1)

- **Eixo Y**: Coeficiente de correlação (-1 a +1)
- **Eixo X**: Delay em amostras
- **Curva**: Amarela, mostra correlação cruzada
- **Interpretação**: Pico indica delay de propagação sônica

### 🤖 Aba 2: Machine Learning

Visualizações específicas dos algoritmos de ML:

```python
def setup_ml_tab(self):
    """Aba de Machine Learning"""
    ml_widget = QWidget()
    ml_layout = QGridLayout(ml_widget)
    
    # Plot 1: Probabilidade de Vazamento
    self.ml_prob_plot = PlotWidget(title="Probabilidade de Vazamento (ML)")
    self.ml_prob_plot.setLabel('left', 'Probabilidade')
    self.ml_prob_plot.setLabel('bottom', 'Tempo')
    self.ml_prob_plot.setYRange(0, 1)
    self.ml_prob_plot.showGrid(x=True, y=True, alpha=0.3)
    
    self.ml_prob_curve = self.ml_prob_plot.plot(
        pen=mkPen('red', width=2), name='Prob. ML'
    )
    
    # Linha de threshold
    self.threshold_line = pg.InfiniteLine(
        pos=0.5, angle=0, pen=mkPen('orange', width=2, style=Qt.PenStyle.DashLine)
    )
    self.ml_prob_plot.addItem(self.threshold_line)
    
    ml_layout.addWidget(self.ml_prob_plot, 0, 0)
    
    # Plot 2: Detecção de Anomalias
    self.anomaly_plot = PlotWidget(title="Detecção de Anomalias")
    self.anomaly_plot.setLabel('left', 'Score de Anomalia')
    self.anomaly_plot.setLabel('bottom', 'Tempo')
    self.anomaly_plot.showGrid(x=True, y=True, alpha=0.3)
    
    self.anomaly_curve = self.anomaly_plot.plot(
        pen=mkPen('orange', width=2), name='Score Anomalia'
    )
    
    ml_layout.addWidget(self.anomaly_plot, 0, 1)
    
    # Plot 3: Confiança do Modelo
    self.confidence_plot = PlotWidget(title="Confiança do Modelo")
    self.confidence_plot.setLabel('left', 'Confiança')
    self.confidence_plot.setLabel('bottom', 'Tempo')
    self.confidence_plot.setYRange(0, 1)
    self.confidence_plot.showGrid(x=True, y=True, alpha=0.3)
    
    self.confidence_curve = self.confidence_plot.plot(
        pen=mkPen('blue', width=2), name='Confiança'
    )
    
    ml_layout.addWidget(self.confidence_plot, 1, 0)
    
    # Plot 4: Features Principais
    self.features_plot = PlotWidget(title="Features Principais") 
    self.features_plot.setLabel('left', 'Valor da Feature')
    self.features_plot.setLabel('bottom', 'Tempo')
    self.features_plot.showGrid(x=True, y=True, alpha=0.3)
    
    # Múltiplas curvas de features serão adicionadas dinamicamente
    
    ml_layout.addWidget(self.features_plot, 1, 1)
    
    self.plots_tab_widget.addTab(ml_widget, "Machine Learning")
```

### 🔧 Aba 3: Status Operacional

Monitora estados operacionais do sistema:

```python
def setup_operational_tab(self):
    """Aba de status operacional"""
    status_widget = QWidget() 
    status_layout = QGridLayout(status_widget)
    
    # Plot 1: Status de Válvulas/Bombas
    self.operational_status_plot = PlotWidget(title="Status Operacional")
    self.operational_status_plot.setLabel('left', 'Status')
    self.operational_status_plot.setLabel('bottom', 'Tempo')
    self.operational_status_plot.showGrid(x=True, y=True, alpha=0.3)
    
    # Estados: 0=Fechado, 1=Aberto, 2=Parcial, 3=Falha
    self.status_curve = self.operational_status_plot.plot(
        pen=None, symbol='o', symbolBrush='blue', symbolSize=8
    )
    
    status_layout.addWidget(self.operational_status_plot, 0, 0)
    
    # Plot 2: Eficiência do Sistema
    self.efficiency_plot = PlotWidget(title="Eficiência do Sistema")
    self.efficiency_plot.setLabel('left', 'Eficiência (%)')
    self.efficiency_plot.setLabel('bottom', 'Tempo')
    self.efficiency_plot.setYRange(0, 100)
    self.efficiency_plot.showGrid(x=True, y=True, alpha=0.3)
    
    self.efficiency_curve = self.efficiency_plot.plot(
        pen=mkPen('green', width=2), name='Eficiência'
    )
    
    status_layout.addWidget(self.efficiency_plot, 0, 1)
    
    self.plots_tab_widget.addTab(status_widget, "Status Operacional")
```

### 🔊 Aba 4: Análise de Ruídos

Análise espectral e caracterização de ruídos:

```python
def setup_noise_tab(self):
    """Aba de análise de ruídos"""
    noise_widget = QWidget()
    noise_layout = QGridLayout(noise_widget)
    
    # Plot 1: FFT - Análise de Frequências
    self.fft_plot = PlotWidget(title="Análise de Frequências (FFT)")
    self.fft_plot.setLabel('left', 'Magnitude (dB)')
    self.fft_plot.setLabel('bottom', 'Frequência (Hz)')
    self.fft_plot.showGrid(x=True, y=True, alpha=0.3)
    
    noise_layout.addWidget(self.fft_plot, 0, 0)
    
    # Plot 2: Espectrograma de Ruído
    self.noise_spectrogram_plot = PlotWidget(title="Espectrograma de Ruído")
    self.noise_spectrogram_plot.setLabel('left', 'Frequência (Hz)')
    self.noise_spectrogram_plot.setLabel('bottom', 'Tempo')
    noise_layout.addWidget(self.noise_spectrogram_plot, 0, 1)
    
    # Plot 3: Detecção de Ruído Anômalo  
    self.anomaly_noise_plot = PlotWidget(title="Detecção de Ruído Anômalo")
    self.anomaly_noise_plot.setLabel('left', 'Intensidade de Ruído')
    self.anomaly_noise_plot.setLabel('bottom', 'Tempo')
    noise_layout.addWidget(self.anomaly_noise_plot, 1, 0, 1, 2)
    
    self.plots_tab_widget.addTab(noise_widget, "Análise de Ruídos")
```

### 🌊 Aba 5: Perfil Hidráulico

Visualização das características hidráulicas do duto:

```python
def setup_hydraulic_tab(self):
    """Aba de perfil hidráulico"""
    hydraulic_widget = QWidget()
    hydraulic_layout = QGridLayout(hydraulic_widget)
    
    # Plot 1: Perfil de Pressão no Duto
    self.pressure_profile_plot = PlotWidget(title="Perfil de Pressão no Duto")
    self.pressure_profile_plot.setLabel('left', 'Pressão (kgf/cm²)')
    self.pressure_profile_plot.setLabel('bottom', 'Distância (km)')
    hydraulic_layout.addWidget(self.pressure_profile_plot, 0, 0)
    
    # Plot 2: Perfil de Elevação
    self.elevation_profile_plot = PlotWidget(title="Perfil de Elevação")
    self.elevation_profile_plot.setLabel('left', 'Elevação (m)')
    self.elevation_profile_plot.setLabel('bottom', 'Distância (km)')
    hydraulic_layout.addWidget(self.elevation_profile_plot, 0, 1)
    
    # Plot 3: Perfil de Velocidades
    self.velocity_plot = PlotWidget(title="Perfil de Velocidades")
    self.velocity_plot.setLabel('left', 'Velocidade (m/s)')
    self.velocity_plot.setLabel('bottom', 'Distância (km)')
    hydraulic_layout.addWidget(self.velocity_plot, 1, 0, 1, 2)
    
    self.plots_tab_widget.addTab(hydraulic_widget, "Perfil Hidráulico")
```

### 📊 Aba 11: 📁 Arquivos

Nova aba para informações dos arquivos carregados:

```python
def setup_files_tab(self):
    """Aba com informações dos arquivos carregados"""
    files_info_widget = QWidget()
    files_info_layout = QVBoxLayout(files_info_widget)
    
    # Grupo de arquivos carregados
    files_group = QGroupBox("Arquivos Carregados")
    files_layout = QVBoxLayout(files_group)
    
    # Tabela de arquivos
    self.files_table = QTableWidget(0, 6)
    self.files_table.setHorizontalHeaderLabels([
        "Arquivo", "Sensor", "Variável", "Unidade", "Registros", "Status"
    ])
    self.files_table.horizontalHeader().setStretchLastSection(True)
    files_layout.addWidget(self.files_table)
    
    files_info_layout.addWidget(files_group)
    
    # Grupo de estatísticas resumo
    stats_group = QGroupBox("Resumo Estatístico")
    stats_layout = QGridLayout(stats_group)
    
    # Labels informativos
    self.total_files_label = QLabel("Arquivos carregados: 0")
    self.total_records_label = QLabel("Total de registros: 0")
    self.date_range_label = QLabel("Período: -")
    self.sensors_info_label = QLabel("Sensores: -")
    
    stats_layout.addWidget(self.total_files_label, 0, 0)
    stats_layout.addWidget(self.total_records_label, 0, 1)
    stats_layout.addWidget(self.date_range_label, 1, 0)
    stats_layout.addWidget(self.sensors_info_label, 1, 1)
    
    files_info_layout.addWidget(stats_group)
    
    # Botões de ação
    files_buttons_layout = QHBoxLayout()
    
    refresh_files_btn = QPushButton("🔄 Atualizar")
    refresh_files_btn.clicked.connect(self.refresh_files_info)
    refresh_files_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
    files_buttons_layout.addWidget(refresh_files_btn)
    
    clear_files_btn = QPushButton("🗑️ Limpar")
    clear_files_btn.clicked.connect(self.clear_files_info)
    clear_files_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
    files_buttons_layout.addWidget(clear_files_btn)
    
    files_buttons_layout.addStretch()
    files_info_layout.addLayout(files_buttons_layout)
    
    self.plots_tab_widget.addTab(files_info_widget, "📁 Arquivos")
```

---

## ⚙️ Diálogos de Configuração

### `SystemConfigDialog`

Diálogo principal para configuração de sistemas:

```python
class SystemConfigDialog(QDialog):
    """Dialog para configuração completa de sistema"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuração do Sistema Hidráulico")
        self.setModal(True)
        self.resize(800, 700)
        self.setup_ui()
        
    def setup_ui(self):
        """Interface do diálogo"""
        layout = QVBoxLayout(self)
        
        # Seções organizadas em grupos
        layout.addWidget(self.create_basic_info_group())
        layout.addWidget(self.create_units_group()) 
        layout.addWidget(self.create_pipe_group())
        layout.addWidget(self.create_fluid_group())
        layout.addWidget(self.create_sensors_group())
        layout.addWidget(self.create_buttons_group())
```

#### 📋 Grupos de Configuração

##### **Informações Básicas**

```python
def create_basic_info_group(self):
    basic_group = QGroupBox("Informações Básicas")
    basic_layout = QGridLayout(basic_group)
    
    basic_layout.addWidget(QLabel("ID do Sistema:"), 0, 0)
    self.system_id_edit = QLineEdit()
    basic_layout.addWidget(self.system_id_edit, 0, 1)
    
    basic_layout.addWidget(QLabel("Nome:"), 0, 2)
    self.name_edit = QLineEdit()
    basic_layout.addWidget(self.name_edit, 0, 3)
    
    basic_layout.addWidget(QLabel("Localização:"), 1, 0)
    self.location_edit = QLineEdit()
    basic_layout.addWidget(self.location_edit, 1, 1, 1, 3)
    
    return basic_group
```

##### **Unidades do Sistema**

```python
def create_units_group(self):
    units_group = QGroupBox("Unidades do Sistema")
    units_layout = QGridLayout(units_group)
    
    units_layout.addWidget(QLabel("Unidade Expedidora:"), 0, 0)
    self.expeditor_unit_edit = QLineEdit()
    self.expeditor_unit_edit.setPlaceholderText("Nome da unidade expedidora")
    units_layout.addWidget(self.expeditor_unit_edit, 0, 1)
    
    units_layout.addWidget(QLabel("Alias Expedidora:"), 0, 2)
    self.expeditor_alias_edit = QLineEdit()
    self.expeditor_alias_edit.setPlaceholderText("Ex: BAR, EXP, UP")
    self.expeditor_alias_edit.setText("BAR")
    units_layout.addWidget(self.expeditor_alias_edit, 0, 3)
    
    units_layout.addWidget(QLabel("Unidade Recebedora:"), 1, 0)
    self.receiver_unit_edit = QLineEdit()
    self.receiver_unit_edit.setPlaceholderText("Nome da unidade recebedora")
    units_layout.addWidget(self.receiver_unit_edit, 1, 1)
    
    units_layout.addWidget(QLabel("Alias Recebedora:"), 1, 2)
    self.receiver_alias_edit = QLineEdit()
    self.receiver_alias_edit.setPlaceholderText("Ex: PLN, REC, DOWN")
    self.receiver_alias_edit.setText("PLN")
    units_layout.addWidget(self.receiver_alias_edit, 1, 3)
    
    return units_group
```

##### **Características do Duto**

```python
def create_pipe_group(self):
    pipe_group = QGroupBox("Características do Duto")
    pipe_layout = QGridLayout(pipe_group)
    
    # Diâmetro com conversão automática
    pipe_layout.addWidget(QLabel("Diâmetro:"), 0, 0)
    diameter_container = QHBoxLayout()
    
    self.diameter_spin = QDoubleSpinBox()
    self.diameter_spin.setRange(0.01, 10.0)
    self.diameter_spin.setDecimals(3)
    self.diameter_spin.setValue(0.5)
    self.diameter_spin.setSuffix(" m")
    diameter_container.addWidget(self.diameter_spin)
    
    self.diameter_inch_spin = QDoubleSpinBox()
    self.diameter_inch_spin.setRange(0.1, 400.0)
    self.diameter_inch_spin.setDecimals(2)
    self.diameter_inch_spin.setSuffix(" in")
    diameter_container.addWidget(QLabel("ou"))
    diameter_container.addWidget(self.diameter_inch_spin)
    
    # Conecta conversões automáticas
    self.diameter_spin.valueChanged.connect(self.diameter_m_to_inch)
    self.diameter_inch_spin.valueChanged.connect(self.diameter_inch_to_m)
    
    pipe_layout.addLayout(diameter_container, 0, 1, 1, 3)
    
    # Material
    pipe_layout.addWidget(QLabel("Material:"), 1, 0)
    self.material_combo = QComboBox()
    self.material_combo.addItems(['steel', 'pvc', 'concrete', 'fiberglass'])
    pipe_layout.addWidget(self.material_combo, 1, 1)
    
    # Perfil
    pipe_layout.addWidget(QLabel("Perfil:"), 1, 2)
    self.profile_combo = QComboBox()
    self.profile_combo.addItems(['circular', 'rectangular', 'oval'])
    pipe_layout.addWidget(self.profile_combo, 1, 3)
    
    return pipe_group
```

#### 🔄 Conversões Automáticas

O diálogo inclui conversões automáticas entre unidades:

```python
def diameter_m_to_inch(self, value_m):
    """Converte diâmetro de metros para polegadas"""
    if not self.diameter_inch_spin.hasFocus():
        inches = value_m * 39.3701
        self.diameter_inch_spin.blockSignals(True)
        self.diameter_inch_spin.setValue(inches)
        self.diameter_inch_spin.blockSignals(False)

def diameter_inch_to_m(self, value_inch):
    """Converte diâmetro de polegadas para metros"""  
    if not self.diameter_spin.hasFocus():
        meters = value_inch * 0.0254
        self.diameter_spin.blockSignals(True)
        self.diameter_spin.setValue(meters)
        self.diameter_spin.blockSignals(False)
```

---

## 📊 Sistema de Status

### Barra de Status

A barra de status na parte inferior mostra informações em tempo real:

```python
def setup_status_bar(self):
    """Configura barra de status"""
    self.status_bar = QStatusBar()
    self.setStatusBar(self.status_bar)
    
    # Widgets da barra de status
    self.status_label = QLabel("Sistema pronto")
    self.progress_label = QLabel("")
    self.time_label = QLabel("")
    
    self.status_bar.addWidget(self.status_label, 1)  # Stretch=1
    self.status_bar.addPermanentWidget(self.progress_label)
    self.status_bar.addPermanentWidget(self.time_label)
    
    # Timer para atualização do horário
    self.time_timer = QTimer()
    self.time_timer.timeout.connect(self.update_time)
    self.time_timer.start(1000)  # Atualiza a cada segundo
    
def update_time(self):
    """Atualiza horário na barra de status"""
    current_time = QDateTime.currentDateTime().toString("dd/MM/yyyy hh:mm:ss")
    self.time_label.setText(current_time)
```

### Método `log_status()`

Sistema centralizado de logging na interface:

```python
def log_status(self, message: str, level: str = "INFO"):
    """
    Log de status na interface
    
    Args:
        message: Mensagem a ser exibida
        level: Nível do log (INFO, WARNING, ERROR, SUCCESS)
    """
    # Timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Formatação por nível
    if level == "ERROR":
        color = "#f44336"  # Vermelho
        icon = "❌"
    elif level == "WARNING":
        color = "#FF9800"  # Laranja  
        icon = "⚠️"
    elif level == "SUCCESS":
        color = "#4CAF50"  # Verde
        icon = "✅"
    else:  # INFO
        color = "#2196F3"  # Azul
        icon = "ℹ️"
    
    # Atualiza status bar
    formatted_message = f"{icon} [{timestamp}] {message}"
    self.status_bar.showMessage(formatted_message, 5000)  # 5 segundos
    
    # Log interno
    logger.info(f"UI Status: {message}")
```

---

## 🔄 Componentes Assíncronos

### `AsyncFileLoader`

Carregador assíncrono de arquivos para manter interface responsiva:

```python
class AsyncFileLoader(QThread):
    """Carregador assíncrono de arquivos"""
    
    # Sinais PyQt6
    progress_updated = pyqtSignal(int)
    data_loaded = pyqtSignal(object)  
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    
    def __init__(self, file_path: str, system_config: SystemConfiguration):
        super().__init__()
        self.file_path = file_path
        self.system_config = system_config
        self.is_cancelled = False
        
    def run(self):
        """Execução principal da thread"""
        try:
            self.status_updated.emit(f"Carregando arquivo: {os.path.basename(self.file_path)}")
            self.progress_updated.emit(10)
            
            # Carrega dados baseado na extensão
            if self.file_path.lower().endswith('.csv'):
                df = pd.read_csv(self.file_path)
            else:
                df = pd.read_excel(self.file_path)
                
            if self.is_cancelled:
                return
                
            self.progress_updated.emit(50)
            
            # Processamento dos dados
            sensor_info = self.system_config.get_sensor_info()
            processed_data = self.process_dataframe(df, sensor_info)
            
            self.progress_updated.emit(90)
            
            if not self.is_cancelled:
                self.data_loaded.emit(processed_data)
                self.progress_updated.emit(100)
                self.status_updated.emit(f"✅ Carregamento concluído: {len(processed_data)} registros")
                
        except Exception as e:
            self.error_occurred.emit(f"Erro ao carregar arquivo: {str(e)}")
            
    def cancel(self):
        """Cancela operação"""
        self.is_cancelled = True
        self.quit()
```

### `ParallelDataProcessor`

Processador paralelo para análises pesadas:

```python
class ParallelDataProcessor(QThread):
    """Processador paralelo de dados"""
    
    analysis_complete = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, snapshots: List[MultiVariableSnapshot], processor: IndustrialHydraulicProcessor):
        super().__init__()
        self.snapshots = snapshots
        self.processor = processor
        
    def run(self):
        """Processamento em thread separada"""
        try:
            # Divisão em lotes para processamento paralelo
            batch_size = max(100, len(self.snapshots) // 4)
            batches = [self.snapshots[i:i+batch_size] for i in range(0, len(self.snapshots), batch_size)]
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for i, batch in enumerate(batches):
                    future = executor.submit(self.processor.process_batch, batch)
                    futures.append(future)
                
                results = []
                for i, future in enumerate(futures):
                    result = future.result()
                    results.append(result)
                    progress = int((i + 1) * 100 / len(futures))
                    self.progress_updated.emit(progress)
            
            # Consolida resultados
            final_result = self.consolidate_results(results)
            self.analysis_complete.emit(final_result)
            
        except Exception as e:
            self.analysis_complete.emit({'error': str(e)})
```

### Conectores de Sinais

Sistema de comunicação thread-safe:

```python
def setup_connections(self):
    """Conecta todos os sinais da interface"""
    
    # Botões principais
    self.new_system_btn.clicked.connect(self.new_system)
    self.load_system_btn.clicked.connect(self.load_system)
    self.load_data_btn.clicked.connect(self.load_data_file)
    
    # Controles de análise  
    self.start_analysis_btn.clicked.connect(self.start_analysis)
    self.stop_analysis_btn.clicked.connect(self.stop_analysis)
    self.analysis_mode_combo.currentTextChanged.connect(self.analysis_mode_changed)
    
    # ML e aprendizado
    self.report_leak_btn.clicked.connect(self.report_confirmed_leak)
    self.retrain_btn.clicked.connect(self.retrain_ml_model)
    
    # Sliders e combos
    self.speed_slider.valueChanged.connect(self.speed_changed)
    self.sensitivity_combo.currentTextChanged.connect(self.sensitivity_changed)
    
def connect_async_loader(self, loader: AsyncFileLoader):
    """Conecta sinais do carregador assíncrono"""
    loader.progress_updated.connect(self.progress_bar.setValue)
    loader.data_loaded.connect(self.on_data_loaded)
    loader.error_occurred.connect(self.on_error_occurred)
    loader.status_updated.connect(self.log_status)
    
def connect_async_processor(self, processor: ParallelDataProcessor):
    """Conecta sinais do processador assíncrono"""
    processor.progress_updated.connect(self.progress_bar.setValue)
    processor.analysis_complete.connect(self.on_analysis_complete)
```

---

## 🎯 Fluxo de Interação

### Sequência Típica de Uso

1. **Inicialização**

   ```
   Aplicação inicia → Janela principal → "Nenhum sistema carregado"
   ```

2. **Configuração de Sistema**

   ```
   "Novo Sistema" → SystemConfigDialog → Configuração → Salvar
   ```

3. **Carregamento de Dados**

   ```
   "Carregar Dados" → AsyncFileLoader → Progress → Dados carregados
   ```

4. **Análise**

   ```
   "Iniciar Análise" → ParallelDataProcessor → Gráficos atualizados
   ```

5. **Monitoramento**

   ```
   Visualização em tempo real → Alertas → Feedback do usuário
   ```

### Estados da Interface

#### 🟥 Estado Inicial

- **Sistema**: Não carregado (label vermelho)
- **Botões Habilitados**: Novo Sistema, Carregar Sistema
- **Botões Desabilitados**: Carregar Dados, Análise, ML
- **Status**: "Sistema pronto - Configure um sistema para começar"

#### 🟨 Estado Sistema Configurado

- **Sistema**: Configurado (label azul)
- **Botões Habilitados**: Todos exceto análise
- **Botões Desabilitados**: Iniciar Análise (até carregar dados)
- **Status**: "Sistema [Nome] carregado"

#### 🟩 Estado Pronto para Análise

- **Sistema**: Configurado com dados
- **Botões Habilitados**: Todos
- **Gráficos**: Populados com dados
- **Status**: "Dados carregados - [N] registros"

#### 🟦 Estado Analisando

- **Sistema**: Em análise
- **Botões Habilitados**: Parar Análise
- **Botões Desabilitados**: Iniciar Análise
- **Progress**: Barra ativa
- **Status**: "Análise em andamento..."

---

## 📱 Responsividade e Performance

### Otimizações de Interface

#### 🎨 Renderização Otimizada

- **PyQtGraph**: Aceleração OpenGL quando disponível
- **Decimation**: Redução automática de pontos em gráficos grandes
- **Lazy Updates**: Atualização apenas quando necessário
- **Double Buffering**: Renderização suave

#### 🧵 Threading Strategy

- **Thread Principal**: Apenas UI
- **Worker Threads**: Carregamento e processamento
- **Thread Pool**: Análises paralelas
- **Signal/Slot**: Comunicação thread-safe

#### 💾 Gestão de Memória

- **Cache Limitado**: Máximo 2000 resultados
- **Garbage Collection**: Limpeza automática
- **Memory Mapping**: Para arquivos grandes
- **Lazy Loading**: Carregamento sob demanda

---

## 🎯 Próximos Manuais Específicos

A interface gráfica será detalhada nos próximos manuais especializados:

1. **MANUAL_03_CORRELACAO_SONICA.md** - Visualizações e análises sônicas
2. **MANUAL_04_MACHINE_LEARNING.md** - Interface ML e algoritmos
3. **MANUAL_05_ANALISE_MULTIVARIAVEL.md** - Gráficos multivariáveis
4. **MANUAL_06_FILTROS_TEMPORAIS.md** - Interface de filtros
5. **MANUAL_07_ANALISE_ESPECTRAL.md** - Visualização espectral
6. **MANUAL_08_VALIDACAO_FISICA.md** - Alertas e validações
7. **MANUAL_09_CONFIGURACAO_SISTEMA.md** - Diálogos e configurações

---

**Sistema de Análise Hidráulica Industrial v2.0**  
*Manual da Interface Gráfica - Agosto 2025*
