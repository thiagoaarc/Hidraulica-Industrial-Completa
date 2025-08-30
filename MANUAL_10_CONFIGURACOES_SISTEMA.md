# Manual de Configurações do Sistema - Sistema Hidráulico Industrial

## 📋 Índice

1. [Visão Geral das Configurações](#visao-geral-das-configuracoes)
2. [Interface da Aba Configurações](#interface-da-aba-configuracoes)
3. [Configurações de Sistema](#configuracoes-de-sistema)
4. [Parâmetros de Análise](#parametros-de-analise)
5. [Configurações de Interface](#configuracoes-de-interface)
6. [Configurações de Dados](#configuracoes-de-dados)
7. [Configurações Avançadas](#configuracoes-avancadas)
8. [Gerenciamento de Perfis](#gerenciamento-de-perfis)

---

## ⚙️ Visão Geral das Configurações

### Sistema de Configuração Centralizada

O **Sistema de Configurações** centraliza todos os **parâmetros operacionais** do sistema hidráulico, permitindo personalização completa para diferentes aplicações e cenários. É fundamental para:

- **Personalização de Análises**: Ajuste de algoritmos por aplicação
- **Calibração de Sensores**: Fatores de correção e offsets
- **Definição de Limites**: Thresholds para alarmes e detecção
- **Otimização de Performance**: Parâmetros de processamento
- **Configuração de Interface**: Layout e preferências do usuário

#### 🎯 Categorias de Configuração

##### **Configurações de Sistema**

- **Hardware**: Configurações de sensores, aquisição, comunicação
- **Software**: Parâmetros de algoritmos, processamento, cache
- **Segurança**: Limites operacionais, alarmes críticos
- **Performance**: Otimização de processamento e memória

##### **Configurações de Análise**

- **Machine Learning**: Parâmetros de modelos, treinamento, validação
- **Filtros**: Especificações de filtros temporais e espaciais
- **Detecção**: Thresholds para vazamentos, anomalias, falhas
- **Correlação**: Parâmetros de análise sônica e espectral

##### **Configurações de Interface**

- **Visualização**: Cores, escalas, layouts de gráficos
- **Interação**: Atalhos, comportamentos, responsividade
- **Relatórios**: Formatos, conteúdo, frequência
- **Exportação**: Tipos de arquivo, compressão, qualidade

---

## 🖥️ Interface da Aba Configurações

### Layout da Interface

A aba **"Configurações"** oferece interface organizada por categorias:

```
┌─────────────────────────────────────────────────────────┐
│                    CONFIGURAÇÕES                        │
├───────────────┬─────────────────────────────────────────┤
│               │                                         │
│   Categorias  │         Painel de Configuração         │
│               │                                         │
│  □ Sistema    │    [Configurações da Categoria]        │
│  □ Análise    │                                         │
│  □ Interface  │    ┌─────────────────────────────────┐  │
│  □ Dados      │    │                                 │  │
│  □ Avançado   │    │     Controles Específicos      │  │
│  □ Perfis     │    │                                 │  │
│               │    └─────────────────────────────────┘  │
│               │                                         │
│               │    [Aplicar] [Resetar] [Salvar]       │
└───────────────┴─────────────────────────────────────────┘
```

#### 🎛️ Configuração da Interface

```python
def setup_config_tab(self):
    """
    Configura a aba de configurações do sistema
    
    Funcionalidades:
    1. Interface organizada por categorias
    2. Validação em tempo real de parâmetros
    3. Sistema de perfis de configuração
    4. Backup e restauração de configurações
    """
    config_widget = QWidget()
    config_layout = QHBoxLayout(config_widget)
    
    # Painel lateral de categorias
    categories_widget = QListWidget()
    categories_widget.setMaximumWidth(200)
    categories_widget.setStyleSheet("""
        QListWidget {
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        QListWidget::item:selected {
            background-color: #4CAF50;
            color: white;
        }
    """)
    
    # Adiciona categorias
    categories = [
        ("🖥️ Sistema", "system"),
        ("📊 Análise", "analysis"), 
        ("🎨 Interface", "interface"),
        ("💾 Dados", "data"),
        ("⚡ Avançado", "advanced"),
        ("👤 Perfis", "profiles")
    ]
    
    for display_name, internal_name in categories:
        item = QListWidgetItem(display_name)
        item.setData(Qt.ItemDataRole.UserRole, internal_name)
        categories_widget.addItem(item)
    
    categories_widget.currentItemChanged.connect(self.on_category_changed)
    config_layout.addWidget(categories_widget)
    
    # Painel principal de configurações
    self.config_stack = QStackedWidget()
    
    # Cria painéis para cada categoria
    self.setup_system_config_panel()
    self.setup_analysis_config_panel()
    self.setup_interface_config_panel()
    self.setup_data_config_panel()
    self.setup_advanced_config_panel()
    self.setup_profiles_config_panel()
    
    config_layout.addWidget(self.config_stack)
    
    # Seleciona primeira categoria
    categories_widget.setCurrentRow(0)
    
    # Adiciona aba
    self.plots_tab_widget.addTab(config_widget, "Configurações")

def on_category_changed(self, current_item, previous_item):
    """
    Muda painel quando categoria é selecionada
    """
    if current_item:
        category = current_item.data(Qt.ItemDataRole.UserRole)
        category_index = {
            'system': 0,
            'analysis': 1,
            'interface': 2,
            'data': 3,
            'advanced': 4,
            'profiles': 5
        }.get(category, 0)
        
        self.config_stack.setCurrentIndex(category_index)
```

---

## 🖥️ Configurações de Sistema

### Parâmetros Fundamentais

#### ⚡ **Hardware e Aquisição**

```python
def setup_system_config_panel(self):
    """
    Configura painel de configurações de sistema
    """
    system_panel = QWidget()
    system_layout = QVBoxLayout(system_panel)
    
    # Grupo: Configurações de Hardware
    hardware_group = QGroupBox("Configurações de Hardware")
    hardware_layout = QFormLayout(hardware_group)
    
    # Taxa de amostragem
    self.sampling_rate_spinbox = QSpinBox()
    self.sampling_rate_spinbox.setRange(1, 100000)
    self.sampling_rate_spinbox.setValue(self.system_config.get('sampling_rate', 1000))
    self.sampling_rate_spinbox.setSuffix(" Hz")
    hardware_layout.addRow("Taxa de Amostragem:", self.sampling_rate_spinbox)
    
    # Número de canais
    self.channels_spinbox = QSpinBox()
    self.channels_spinbox.setRange(1, 64)
    self.channels_spinbox.setValue(self.system_config.get('num_channels', 8))
    hardware_layout.addRow("Número de Canais:", self.channels_spinbox)
    
    # Resolução ADC
    self.adc_resolution_combo = QComboBox()
    self.adc_resolution_combo.addItems(["12 bits", "14 bits", "16 bits", "18 bits", "24 bits"])
    current_resolution = self.system_config.get('adc_resolution', 16)
    self.adc_resolution_combo.setCurrentText(f"{current_resolution} bits")
    hardware_layout.addRow("Resolução ADC:", self.adc_resolution_combo)
    
    # Range de entrada
    self.input_range_combo = QComboBox()
    self.input_range_combo.addItems(["±1V", "±2V", "±5V", "±10V", "0-10V", "4-20mA"])
    self.input_range_combo.setCurrentText(self.system_config.get('input_range', '±10V'))
    hardware_layout.addRow("Range de Entrada:", self.input_range_combo)
    
    system_layout.addWidget(hardware_group)
    
    # Grupo: Configurações de Comunicação
    comm_group = QGroupBox("Configurações de Comunicação")
    comm_layout = QFormLayout(comm_group)
    
    # Protocolo de comunicação
    self.comm_protocol_combo = QComboBox()
    self.comm_protocol_combo.addItems(["TCP/IP", "Serial", "Modbus TCP", "OPC UA", "MQTT"])
    self.comm_protocol_combo.setCurrentText(self.system_config.get('comm_protocol', 'TCP/IP'))
    comm_layout.addRow("Protocolo:", self.comm_protocol_combo)
    
    # Endereço/Porta
    self.comm_address_edit = QLineEdit()
    self.comm_address_edit.setText(self.system_config.get('comm_address', '192.168.1.100:502'))
    comm_layout.addRow("Endereço:", self.comm_address_edit)
    
    # Timeout
    self.comm_timeout_spinbox = QSpinBox()
    self.comm_timeout_spinbox.setRange(100, 30000)
    self.comm_timeout_spinbox.setValue(self.system_config.get('comm_timeout', 5000))
    self.comm_timeout_spinbox.setSuffix(" ms")
    comm_layout.addRow("Timeout:", self.comm_timeout_spinbox)
    
    system_layout.addWidget(comm_group)
    
    # Grupo: Configurações de Performance
    perf_group = QGroupBox("Configurações de Performance")
    perf_layout = QFormLayout(perf_group)
    
    # Buffer size
    self.buffer_size_spinbox = QSpinBox()
    self.buffer_size_spinbox.setRange(1000, 1000000)
    self.buffer_size_spinbox.setValue(self.system_config.get('buffer_size', 10000))
    self.buffer_size_spinbox.setSuffix(" amostras")
    perf_layout.addRow("Tamanho do Buffer:", self.buffer_size_spinbox)
    
    # Threads de processamento
    self.processing_threads_spinbox = QSpinBox()
    self.processing_threads_spinbox.setRange(1, 16)
    self.processing_threads_spinbox.setValue(self.system_config.get('processing_threads', 4))
    perf_layout.addRow("Threads de Processamento:", self.processing_threads_spinbox)
    
    # Cache de dados
    self.data_cache_checkbox = QCheckBox()
    self.data_cache_checkbox.setChecked(self.system_config.get('enable_data_cache', True))
    perf_layout.addRow("Cache de Dados:", self.data_cache_checkbox)
    
    system_layout.addWidget(perf_group)
    
    self.config_stack.addWidget(system_panel)

def get_system_configuration(self) -> Dict[str, Any]:
    """
    Retorna configuração atual do sistema
    """
    return {
        'hardware': {
            'sampling_rate': self.sampling_rate_spinbox.value(),
            'num_channels': self.channels_spinbox.value(),
            'adc_resolution': int(self.adc_resolution_combo.currentText().split()[0]),
            'input_range': self.input_range_combo.currentText()
        },
        'communication': {
            'protocol': self.comm_protocol_combo.currentText(),
            'address': self.comm_address_edit.text(),
            'timeout': self.comm_timeout_spinbox.value()
        },
        'performance': {
            'buffer_size': self.buffer_size_spinbox.value(),
            'processing_threads': self.processing_threads_spinbox.value(),
            'enable_data_cache': self.data_cache_checkbox.isChecked()
        }
    }
```

---

## 📊 Parâmetros de Análise

### Configurações Algorítmicas

#### 🤖 **Machine Learning**

```python
def setup_analysis_config_panel(self):
    """
    Configura painel de parâmetros de análise
    """
    analysis_panel = QWidget()
    analysis_layout = QVBoxLayout(analysis_panel)
    
    # Grupo: Machine Learning
    ml_group = QGroupBox("Configurações de Machine Learning")
    ml_layout = QFormLayout(ml_group)
    
    # Algoritmo de anomaly detection
    self.anomaly_algorithm_combo = QComboBox()
    self.anomaly_algorithm_combo.addItems([
        "Isolation Forest",
        "One-Class SVM", 
        "Local Outlier Factor",
        "DBSCAN",
        "Autoencoder"
    ])
    ml_layout.addRow("Algoritmo de Anomalia:", self.anomaly_algorithm_combo)
    
    # Threshold de anomalia
    self.anomaly_threshold_spinbox = QDoubleSpinBox()
    self.anomaly_threshold_spinbox.setRange(0.01, 1.0)
    self.anomaly_threshold_spinbox.setSingleStep(0.01)
    self.anomaly_threshold_spinbox.setValue(0.1)
    self.anomaly_threshold_spinbox.setSuffix(" (0-1)")
    ml_layout.addRow("Threshold de Anomalia:", self.anomaly_threshold_spinbox)
    
    # Janela de análise
    self.analysis_window_spinbox = QSpinBox()
    self.analysis_window_spinbox.setRange(100, 10000)
    self.analysis_window_spinbox.setValue(1000)
    self.analysis_window_spinbox.setSuffix(" amostras")
    ml_layout.addRow("Janela de Análise:", self.analysis_window_spinbox)
    
    # Retreinamento automático
    self.auto_retrain_checkbox = QCheckBox()
    self.auto_retrain_checkbox.setChecked(True)
    ml_layout.addRow("Retreinamento Automático:", self.auto_retrain_checkbox)
    
    analysis_layout.addWidget(ml_group)
    
    # Grupo: Detecção de Vazamentos
    leak_group = QGroupBox("Detecção de Vazamentos")
    leak_layout = QFormLayout(leak_group)
    
    # Sensibilidade
    self.leak_sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
    self.leak_sensitivity_slider.setRange(1, 10)
    self.leak_sensitivity_slider.setValue(5)
    self.leak_sensitivity_label = QLabel("Média (5)")
    self.leak_sensitivity_slider.valueChanged.connect(
        lambda v: self.leak_sensitivity_label.setText(
            f"{'Baixa' if v <= 3 else 'Média' if v <= 7 else 'Alta'} ({v})"
        )
    )
    
    leak_sensitivity_layout = QHBoxLayout()
    leak_sensitivity_layout.addWidget(self.leak_sensitivity_slider)
    leak_sensitivity_layout.addWidget(self.leak_sensitivity_label)
    
    leak_layout.addRow("Sensibilidade:", leak_sensitivity_layout)
    
    # Frequência de análise
    self.leak_analysis_freq_combo = QComboBox()
    self.leak_analysis_freq_combo.addItems(["Contínua", "1 Hz", "0.1 Hz", "Manual"])
    leak_layout.addRow("Frequência de Análise:", self.leak_analysis_freq_combo)
    
    # Tamanho mínimo de vazamento
    self.min_leak_size_spinbox = QDoubleSpinBox()
    self.min_leak_size_spinbox.setRange(0.1, 10.0)
    self.min_leak_size_spinbox.setValue(1.0)
    self.min_leak_size_spinbox.setSuffix(" % da vazão")
    leak_layout.addRow("Vazamento Mínimo:", self.min_leak_size_spinbox)
    
    analysis_layout.addWidget(leak_group)
    
    # Grupo: Processamento Espectral
    spectral_group = QGroupBox("Processamento Espectral")
    spectral_layout = QFormLayout(spectral_group)
    
    # Janela FFT
    self.fft_window_combo = QComboBox()
    self.fft_window_combo.addItems(["Hann", "Hamming", "Blackman", "Kaiser", "Rectangular"])
    spectral_layout.addRow("Janela FFT:", self.fft_window_combo)
    
    # Overlap
    self.fft_overlap_spinbox = QSpinBox()
    self.fft_overlap_spinbox.setRange(0, 95)
    self.fft_overlap_spinbox.setValue(50)
    self.fft_overlap_spinbox.setSuffix(" %")
    spectral_layout.addRow("Sobreposição:", self.fft_overlap_spinbox)
    
    # Zero padding
    self.zero_padding_checkbox = QCheckBox()
    self.zero_padding_checkbox.setChecked(True)
    spectral_layout.addRow("Zero Padding:", self.zero_padding_checkbox)
    
    analysis_layout.addWidget(spectral_group)
    
    self.config_stack.addWidget(analysis_panel)
```

#### 🔍 **Filtros e Correlação**

```python
def setup_filter_correlation_configs(self, parent_layout):
    """
    Configurações de filtros e correlação
    """
    
    # Grupo: Filtros Digitais
    filter_group = QGroupBox("Configurações de Filtros")
    filter_layout = QFormLayout(filter_group)
    
    # Filtro padrão
    self.default_filter_combo = QComboBox()
    self.default_filter_combo.addItems([
        "Butterworth",
        "Chebyshev Tipo I",
        "Chebyshev Tipo II", 
        "Elíptico",
        "FIR"
    ])
    filter_layout.addRow("Filtro Padrão:", self.default_filter_combo)
    
    # Ordem padrão
    self.default_filter_order_spinbox = QSpinBox()
    self.default_filter_order_spinbox.setRange(1, 12)
    self.default_filter_order_spinbox.setValue(4)
    filter_layout.addRow("Ordem Padrão:", self.default_filter_order_spinbox)
    
    # Frequência de corte automática
    self.auto_cutoff_checkbox = QCheckBox()
    self.auto_cutoff_checkbox.setChecked(True)
    filter_layout.addRow("Corte Automático:", self.auto_cutoff_checkbox)
    
    parent_layout.addWidget(filter_group)
    
    # Grupo: Correlação
    corr_group = QGroupBox("Configurações de Correlação")
    corr_layout = QFormLayout(corr_group)
    
    # Método de correlação
    self.correlation_method_combo = QComboBox()
    self.correlation_method_combo.addItems([
        "Correlação Cruzada",
        "Correlação Normalizada",
        "Coerência Espectral",
        "Mutual Information"
    ])
    corr_layout.addRow("Método:", self.correlation_method_combo)
    
    # Janela de correlação
    self.correlation_window_spinbox = QSpinBox()
    self.correlation_window_spinbox.setRange(100, 100000)
    self.correlation_window_spinbox.setValue(10000)
    self.correlation_window_spinbox.setSuffix(" amostras")
    corr_layout.addRow("Janela de Correlação:", self.correlation_window_spinbox)
    
    # Threshold de correlação
    self.correlation_threshold_spinbox = QDoubleSpinBox()
    self.correlation_threshold_spinbox.setRange(0.1, 1.0)
    self.correlation_threshold_spinbox.setSingleStep(0.05)
    self.correlation_threshold_spinbox.setValue(0.7)
    corr_layout.addRow("Threshold:", self.correlation_threshold_spinbox)
    
    parent_layout.addWidget(corr_group)
```

---

## 🎨 Configurações de Interface

### Personalização Visual

#### 🌈 **Cores e Temas**

```python
def setup_interface_config_panel(self):
    """
    Configura painel de configurações de interface
    """
    interface_panel = QWidget()
    interface_layout = QVBoxLayout(interface_panel)
    
    # Grupo: Aparência
    appearance_group = QGroupBox("Aparência")
    appearance_layout = QFormLayout(appearance_group)
    
    # Tema
    self.theme_combo = QComboBox()
    self.theme_combo.addItems(["Claro", "Escuro", "Azul Industrial", "Verde Matrix", "Personalizado"])
    appearance_layout.addRow("Tema:", self.theme_combo)
    
    # Tamanho da fonte
    self.font_size_spinbox = QSpinBox()
    self.font_size_spinbox.setRange(8, 24)
    self.font_size_spinbox.setValue(10)
    appearance_layout.addRow("Tamanho da Fonte:", self.font_size_spinbox)
    
    # Família da fonte
    self.font_family_combo = QFontComboBox()
    self.font_family_combo.setCurrentFont(QFont("Arial"))
    appearance_layout.addRow("Fonte:", self.font_family_combo)
    
    interface_layout.addWidget(appearance_group)
    
    # Grupo: Gráficos
    plots_group = QGroupBox("Configurações de Gráficos")
    plots_layout = QFormLayout(plots_group)
    
    # Cores dos gráficos
    self.setup_color_selectors(plots_layout)
    
    # Anti-aliasing
    self.antialiasing_checkbox = QCheckBox()
    self.antialiasing_checkbox.setChecked(True)
    plots_layout.addRow("Anti-aliasing:", self.antialiasing_checkbox)
    
    # Grid
    self.grid_checkbox = QCheckBox()
    self.grid_checkbox.setChecked(True)
    plots_layout.addRow("Grade:", self.grid_checkbox)
    
    # Legendas
    self.legends_checkbox = QCheckBox()
    self.legends_checkbox.setChecked(True)
    plots_layout.addRow("Legendas:", self.legends_checkbox)
    
    interface_layout.addWidget(plots_group)
    
    # Grupo: Interação
    interaction_group = QGroupBox("Interação")
    interaction_layout = QFormLayout(interaction_group)
    
    # Zoom automático
    self.auto_zoom_checkbox = QCheckBox()
    self.auto_zoom_checkbox.setChecked(True)
    interaction_layout.addRow("Zoom Automático:", self.auto_zoom_checkbox)
    
    # Tooltips
    self.tooltips_checkbox = QCheckBox()
    self.tooltips_checkbox.setChecked(True)
    interaction_layout.addRow("Dicas de Ferramenta:", self.tooltips_checkbox)
    
    # Taxa de atualização da interface
    self.ui_refresh_rate_spinbox = QSpinBox()
    self.ui_refresh_rate_spinbox.setRange(1, 60)
    self.ui_refresh_rate_spinbox.setValue(30)
    self.ui_refresh_rate_spinbox.setSuffix(" FPS")
    interaction_layout.addRow("Taxa de Atualização:", self.ui_refresh_rate_spinbox)
    
    interface_layout.addWidget(interaction_group)
    
    self.config_stack.addWidget(interface_panel)

def setup_color_selectors(self, layout):
    """
    Cria seletores de cor para diferentes elementos
    """
    color_configs = [
        ("Cor Primária", "#1f77b4"),
        ("Cor Secundária", "#ff7f0e"), 
        ("Cor de Erro", "#d62728"),
        ("Cor de Sucesso", "#2ca02c"),
        ("Cor de Aviso", "#ff9800"),
        ("Cor de Fundo", "#ffffff"),
        ("Cor do Grid", "#cccccc")
    ]
    
    self.color_buttons = {}
    
    for label, default_color in color_configs:
        color_button = QPushButton()
        color_button.setStyleSheet(f"background-color: {default_color}; border: 1px solid #ccc;")
        color_button.setMaximumSize(50, 30)
        color_button.clicked.connect(lambda checked, btn=color_button: self.select_color(btn))
        
        self.color_buttons[label] = color_button
        layout.addRow(f"{label}:", color_button)

def select_color(self, button):
    """
    Abre seletor de cor
    """
    color = QColorDialog.getColor()
    if color.isValid():
        button.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #ccc;")
```

---

## 💾 Configurações de Dados

### Armazenamento e Backup

#### 📁 **Gerenciamento de Arquivos**

```python
def setup_data_config_panel(self):
    """
    Configura painel de configurações de dados
    """
    data_panel = QWidget()
    data_layout = QVBoxLayout(data_panel)
    
    # Grupo: Armazenamento
    storage_group = QGroupBox("Armazenamento de Dados")
    storage_layout = QFormLayout(storage_group)
    
    # Diretório de dados
    self.data_directory_layout = QHBoxLayout()
    self.data_directory_edit = QLineEdit()
    self.data_directory_edit.setText(self.config_manager.get('data_directory', './data'))
    self.data_directory_button = QPushButton("📁")
    self.data_directory_button.clicked.connect(self.select_data_directory)
    
    self.data_directory_layout.addWidget(self.data_directory_edit)
    self.data_directory_layout.addWidget(self.data_directory_button)
    
    storage_layout.addRow("Diretório de Dados:", self.data_directory_layout)
    
    # Formato de arquivo
    self.data_format_combo = QComboBox()
    self.data_format_combo.addItems(["HDF5", "CSV", "JSON", "Parquet", "Binary"])
    storage_layout.addRow("Formato de Arquivo:", self.data_format_combo)
    
    # Compressão
    self.compression_combo = QComboBox()
    self.compression_combo.addItems(["Nenhuma", "gzip", "lz4", "zstd"])
    storage_layout.addRow("Compressão:", self.compression_combo)
    
    # Tamanho máximo de arquivo
    self.max_file_size_spinbox = QSpinBox()
    self.max_file_size_spinbox.setRange(1, 10000)
    self.max_file_size_spinbox.setValue(100)
    self.max_file_size_spinbox.setSuffix(" MB")
    storage_layout.addRow("Tamanho Máx. Arquivo:", self.max_file_size_spinbox)
    
    data_layout.addWidget(storage_group)
    
    # Grupo: Backup
    backup_group = QGroupBox("Backup Automático")
    backup_layout = QFormLayout(backup_group)
    
    # Habilitar backup
    self.backup_enabled_checkbox = QCheckBox()
    self.backup_enabled_checkbox.setChecked(True)
    backup_layout.addRow("Backup Automático:", self.backup_enabled_checkbox)
    
    # Frequência de backup
    self.backup_frequency_combo = QComboBox()
    self.backup_frequency_combo.addItems([
        "A cada hora", "A cada 4 horas", "Diariamente", 
        "Semanalmente", "Mensalmente"
    ])
    backup_layout.addRow("Frequência:", self.backup_frequency_combo)
    
    # Diretório de backup
    self.backup_directory_layout = QHBoxLayout()
    self.backup_directory_edit = QLineEdit()
    self.backup_directory_edit.setText('./backup')
    self.backup_directory_button = QPushButton("📁")
    self.backup_directory_button.clicked.connect(self.select_backup_directory)
    
    self.backup_directory_layout.addWidget(self.backup_directory_edit)
    self.backup_directory_layout.addWidget(self.backup_directory_button)
    
    backup_layout.addRow("Diretório de Backup:", self.backup_directory_layout)
    
    # Retenção de backups
    self.backup_retention_spinbox = QSpinBox()
    self.backup_retention_spinbox.setRange(1, 365)
    self.backup_retention_spinbox.setValue(30)
    self.backup_retention_spinbox.setSuffix(" dias")
    backup_layout.addRow("Retenção:", self.backup_retention_spinbox)
    
    data_layout.addWidget(backup_group)
    
    # Grupo: Cache
    cache_group = QGroupBox("Configurações de Cache")
    cache_layout = QFormLayout(cache_group)
    
    # Tamanho do cache
    self.cache_size_spinbox = QSpinBox()
    self.cache_size_spinbox.setRange(10, 10000)
    self.cache_size_spinbox.setValue(500)
    self.cache_size_spinbox.setSuffix(" MB")
    cache_layout.addRow("Tamanho do Cache:", self.cache_size_spinbox)
    
    # Cache de resultados ML
    self.ml_cache_checkbox = QCheckBox()
    self.ml_cache_checkbox.setChecked(True)
    cache_layout.addRow("Cache de ML:", self.ml_cache_checkbox)
    
    # Limpar cache na inicialização
    self.clear_cache_startup_checkbox = QCheckBox()
    self.clear_cache_startup_checkbox.setChecked(False)
    cache_layout.addRow("Limpar na Inicialização:", self.clear_cache_startup_checkbox)
    
    data_layout.addWidget(cache_group)
    
    self.config_stack.addWidget(data_panel)

def select_data_directory(self):
    """
    Seleciona diretório de dados
    """
    directory = QFileDialog.getExistingDirectory(
        self, "Selecionar Diretório de Dados"
    )
    if directory:
        self.data_directory_edit.setText(directory)

def select_backup_directory(self):
    """
    Seleciona diretório de backup
    """
    directory = QFileDialog.getExistingDirectory(
        self, "Selecionar Diretório de Backup"
    )
    if directory:
        self.backup_directory_edit.setText(directory)
```

---

## ⚡ Configurações Avançadas

### Parâmetros de Sistema

#### 🔧 **Configurações de Desenvolvedor**

```python
def setup_advanced_config_panel(self):
    """
    Configura painel de configurações avançadas
    """
    advanced_panel = QWidget()
    advanced_layout = QVBoxLayout(advanced_panel)
    
    # Aviso
    warning_label = QLabel("⚠️ Atenção: Alterações nesta seção podem afetar a estabilidade do sistema")
    warning_label.setStyleSheet("color: orange; font-weight: bold; padding: 10px;")
    advanced_layout.addWidget(warning_label)
    
    # Grupo: Debug e Logs
    debug_group = QGroupBox("Debug e Logs")
    debug_layout = QFormLayout(debug_group)
    
    # Nível de log
    self.log_level_combo = QComboBox()
    self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    self.log_level_combo.setCurrentText("INFO")
    debug_layout.addRow("Nível de Log:", self.log_level_combo)
    
    # Arquivo de log
    self.log_file_edit = QLineEdit()
    self.log_file_edit.setText("./logs/hydraulic_system.log")
    debug_layout.addRow("Arquivo de Log:", self.log_file_edit)
    
    # Console de debug
    self.debug_console_checkbox = QCheckBox()
    self.debug_console_checkbox.setChecked(False)
    debug_layout.addRow("Console de Debug:", self.debug_console_checkbox)
    
    # Profiler de performance
    self.profiler_checkbox = QCheckBox()
    self.profiler_checkbox.setChecked(False)
    debug_layout.addRow("Profiler de Performance:", self.profiler_checkbox)
    
    advanced_layout.addWidget(debug_group)
    
    # Grupo: Limites do Sistema
    limits_group = QGroupBox("Limites do Sistema")
    limits_layout = QFormLayout(limits_group)
    
    # Uso máximo de memória
    self.max_memory_spinbox = QSpinBox()
    self.max_memory_spinbox.setRange(100, 32000)
    self.max_memory_spinbox.setValue(4000)
    self.max_memory_spinbox.setSuffix(" MB")
    limits_layout.addRow("Memória Máxima:", self.max_memory_spinbox)
    
    # Uso máximo de CPU
    self.max_cpu_spinbox = QSpinBox()
    self.max_cpu_spinbox.setRange(10, 100)
    self.max_cpu_spinbox.setValue(80)
    self.max_cpu_spinbox.setSuffix(" %")
    limits_layout.addRow("CPU Máxima:", self.max_cpu_spinbox)
    
    # Timeout de operações
    self.operation_timeout_spinbox = QSpinBox()
    self.operation_timeout_spinbox.setRange(1, 300)
    self.operation_timeout_spinbox.setValue(30)
    self.operation_timeout_spinbox.setSuffix(" s")
    limits_layout.addRow("Timeout Operações:", self.operation_timeout_spinbox)
    
    advanced_layout.addWidget(limits_group)
    
    # Grupo: Otimizações
    optimization_group = QGroupBox("Otimizações")
    optimization_layout = QFormLayout(optimization_group)
    
    # Processamento paralelo
    self.parallel_processing_checkbox = QCheckBox()
    self.parallel_processing_checkbox.setChecked(True)
    optimization_layout.addRow("Processamento Paralelo:", self.parallel_processing_checkbox)
    
    # Cache de FFT
    self.fft_cache_checkbox = QCheckBox()
    self.fft_cache_checkbox.setChecked(True)
    optimization_layout.addRow("Cache de FFT:", self.fft_cache_checkbox)
    
    # Otimização de memória
    self.memory_optimization_combo = QComboBox()
    self.memory_optimization_combo.addItems(["Desabilitada", "Conservadora", "Agressiva"])
    optimization_layout.addRow("Otimização de Memória:", self.memory_optimization_combo)
    
    advanced_layout.addWidget(optimization_group)
    
    # Grupo: Segurança
    security_group = QGroupBox("Configurações de Segurança")
    security_layout = QFormLayout(security_group)
    
    # Limites de alarme
    self.alarm_limits_button = QPushButton("Configurar Limites de Alarme...")
    self.alarm_limits_button.clicked.connect(self.configure_alarm_limits)
    security_layout.addRow("Alarmes:", self.alarm_limits_button)
    
    # Modo de operação segura
    self.safe_mode_checkbox = QCheckBox()
    self.safe_mode_checkbox.setChecked(True)
    security_layout.addRow("Modo Seguro:", self.safe_mode_checkbox)
    
    advanced_layout.addWidget(security_group)
    
    self.config_stack.addWidget(advanced_panel)

def configure_alarm_limits(self):
    """
    Abre diálogo para configurar limites de alarme
    """
    dialog = AlarmLimitsDialog(self)
    dialog.exec()
```

---

## 👤 Gerenciamento de Perfis

### Sistema de Perfis

#### 📋 **Perfis de Configuração**

```python
def setup_profiles_config_panel(self):
    """
    Configura painel de gerenciamento de perfis
    """
    profiles_panel = QWidget()
    profiles_layout = QVBoxLayout(profiles_panel)
    
    # Lista de perfis
    profiles_list_group = QGroupBox("Perfis Salvos")
    profiles_list_layout = QVBoxLayout(profiles_list_group)
    
    self.profiles_list_widget = QListWidget()
    self.profiles_list_widget.setMaximumHeight(200)
    
    # Carrega perfis existentes
    self.load_configuration_profiles()
    
    profiles_list_layout.addWidget(self.profiles_list_widget)
    
    # Botões de perfil
    profile_buttons_layout = QHBoxLayout()
    
    self.load_profile_button = QPushButton("Carregar Perfil")
    self.load_profile_button.clicked.connect(self.load_selected_profile)
    profile_buttons_layout.addWidget(self.load_profile_button)
    
    self.save_profile_button = QPushButton("Salvar Perfil Atual")
    self.save_profile_button.clicked.connect(self.save_current_profile)
    profile_buttons_layout.addWidget(self.save_profile_button)
    
    self.delete_profile_button = QPushButton("Excluir Perfil")
    self.delete_profile_button.clicked.connect(self.delete_selected_profile)
    profile_buttons_layout.addWidget(self.delete_profile_button)
    
    profiles_list_layout.addLayout(profile_buttons_layout)
    profiles_layout.addWidget(profiles_list_group)
    
    # Perfil atual
    current_profile_group = QGroupBox("Perfil Atual")
    current_profile_layout = QFormLayout(current_profile_group)
    
    self.current_profile_name_edit = QLineEdit()
    self.current_profile_name_edit.setText("Padrão")
    current_profile_layout.addRow("Nome do Perfil:", self.current_profile_name_edit)
    
    self.current_profile_description_edit = QTextEdit()
    self.current_profile_description_edit.setMaximumHeight(80)
    self.current_profile_description_edit.setText("Configuração padrão do sistema")
    current_profile_layout.addRow("Descrição:", self.current_profile_description_edit)
    
    profiles_layout.addWidget(current_profile_group)
    
    # Importar/Exportar
    import_export_group = QGroupBox("Importar/Exportar")
    import_export_layout = QHBoxLayout(import_export_group)
    
    self.import_config_button = QPushButton("Importar Configuração")
    self.import_config_button.clicked.connect(self.import_configuration)
    import_export_layout.addWidget(self.import_config_button)
    
    self.export_config_button = QPushButton("Exportar Configuração")
    self.export_config_button.clicked.connect(self.export_configuration)
    import_export_layout.addWidget(self.export_config_button)
    
    profiles_layout.addWidget(import_export_group)
    
    self.config_stack.addWidget(profiles_panel)

def load_configuration_profiles(self):
    """
    Carrega lista de perfis salvos
    """
    profiles_dir = "./config/profiles"
    if os.path.exists(profiles_dir):
        for filename in os.listdir(profiles_dir):
            if filename.endswith('.json'):
                profile_name = filename[:-5]  # Remove .json
                self.profiles_list_widget.addItem(profile_name)

def save_current_profile(self):
    """
    Salva configuração atual como perfil
    """
    profile_name = self.current_profile_name_edit.text()
    if not profile_name:
        QMessageBox.warning(self, "Erro", "Nome do perfil não pode estar vazio")
        return
    
    # Coleta todas as configurações
    current_config = {
        'profile_info': {
            'name': profile_name,
            'description': self.current_profile_description_edit.toPlainText(),
            'created_date': datetime.now().isoformat(),
            'version': "1.0"
        },
        'system': self.get_system_configuration(),
        'analysis': self.get_analysis_configuration(),
        'interface': self.get_interface_configuration(),
        'data': self.get_data_configuration(),
        'advanced': self.get_advanced_configuration()
    }
    
    # Salva arquivo
    profiles_dir = "./config/profiles"
    os.makedirs(profiles_dir, exist_ok=True)
    
    filename = os.path.join(profiles_dir, f"{profile_name}.json")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(current_config, f, indent=2, ensure_ascii=False)
        
        # Atualiza lista
        self.profiles_list_widget.addItem(profile_name)
        
        QMessageBox.information(self, "Sucesso", f"Perfil '{profile_name}' salvo com sucesso")
        
    except Exception as e:
        QMessageBox.critical(self, "Erro", f"Erro ao salvar perfil: {str(e)}")

def load_selected_profile(self):
    """
    Carrega perfil selecionado
    """
    current_item = self.profiles_list_widget.currentItem()
    if not current_item:
        QMessageBox.warning(self, "Erro", "Selecione um perfil para carregar")
        return
    
    profile_name = current_item.text()
    filename = f"./config/profiles/{profile_name}.json"
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Aplica configurações
        self.apply_configuration(config)
        
        QMessageBox.information(self, "Sucesso", f"Perfil '{profile_name}' carregado com sucesso")
        
    except Exception as e:
        QMessageBox.critical(self, "Erro", f"Erro ao carregar perfil: {str(e)}")
```

---

## 📋 Conclusão - Configurações do Sistema

### Capacidades Implementadas

✅ **Sistema Completo de Configuração** - Interface organizada por categorias  
✅ **Configurações de Hardware** - ADC, comunicação, performance  
✅ **Parâmetros de Análise** - ML, filtros, detecção, correlação  
✅ **Personalização de Interface** - Temas, cores, layouts  
✅ **Gerenciamento de Dados** - Armazenamento, backup, cache  
✅ **Configurações Avançadas** - Debug, limites, segurança  
✅ **Sistema de Perfis** - Salvar, carregar, importar, exportar

### Características do Sistema

- **Validação em Tempo Real**: Parâmetros validados durante entrada
- **Profiles Contextuais**: Configurações específicas por aplicação
- **Backup Automático**: Proteção contra perda de configurações
- **Interface Intuitiva**: Organização clara e navegação simples
- **Importação/Exportação**: Compartilhamento de configurações

O **Sistema de Configurações** permite personalização total do sistema hidráulico, garantindo flexibilidade máxima para diferentes aplicações industriais enquanto mantém segurança e confiabilidade operacional.
