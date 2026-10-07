import sys
import os
import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QTableView,
    QAbstractItemView,
    QMessageBox,
    QHeaderView,
    QDoubleSpinBox,
    QCheckBox,
    QStackedWidget,
    QComboBox,
    QSpinBox,
)

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from scipy import signal
from fractions import Fraction


class DataFrameTableModel(QAbstractTableModel):
    """Lazy Qt model: hiển thị toàn bộ DataFrame mà không tạo QTableWidgetItem cho từng ô."""

    def __init__(self, dataframe=None, parent=None):
        super().__init__(parent)
        self._df = dataframe if dataframe is not None else pd.DataFrame()

    def set_dataframe(self, dataframe):
        self.beginResetModel()
        self._df = dataframe if dataframe is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            if pd.isna(value):
                return ""
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.6f}"
            return str(value)
        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section]).replace("_", "\n")
        return str(section + 1)




class FOGSignalAnalyzer(QMainWindow):

    def __init__(self):
        super().__init__()

        # =====================================================
        # WINDOW
        # =====================================================

        self.setWindowTitle(
            "FOG Signal Analyzer"
        )

        self.resize(
            1500,
            900
        )

        # =====================================================
        # DATA
        # =====================================================

        self.df = None
        self.df_raw = None
        self.df_processed = None
        self.df_selected = None
        self.display_df = None
        self.display_sampling_rate = None
        self.display_processed_label = False
        self.file_path = None

        self.sampling_rate = None
        self.duration = None

        # Sensor hiện tại
        self.current_sensor = "Ankle"

        # Dữ liệu đang hiển thị trên Time Domain
        self.current_time_array = None
        self.current_signal_data = None
        self.current_signal_columns = []

        # Y-axis cố định trong Time Domain
        self.fixed_y_range = None

        # =====================================================
        # PHASE 3 - PREPROCESSING STATE
        # =====================================================

        self.preprocessing_applied = False
        self.processed_sampling_rate = None
        self.processed_start_time = None
        self.processed_end_time = None

        # =====================================================
        # PHASE 4 - WINDOWING STATE
        # =====================================================

        # Data dùng làm đầu vào cho Windowing.
        # Phase 4 ưu tiên dữ liệu đã qua Phase 3.
        self.windowing_input_df = None
        self.windowing_sampling_rate = None

        # Tham số Windowing hiện tại.
        self.window_length_seconds = 2.0
        self.window_overlap_percent = 50.0
        self.window_step_seconds = 1.0
        self.window_samples = 128
        self.window_step_samples = 64

        # Kết quả Windowing.
        self.window_metadata = None
        self.window_signal_data = []
        self.windowing_applied = False

        # =====================================================
        # PHASE 5 - TIME & FREQUENCY ANALYSIS STATE
        # =====================================================
        self.analysis_window_id = None
        self.analysis_signal = None
        self.analysis_time = None
        self.analysis_fs = None
        self.analysis_fft_freq = None
        self.analysis_fft_mag = None
        self.analysis_psd_freq = None
        self.analysis_psd_power = None

        # =====================================================
        # UI
        # =====================================================

        self.create_ui()

    # =========================================================
    # ESC TO CLOSE
    # =========================================================

    def keyPressEvent(self, event):

        if event.key() == Qt.Key_Escape:

            self.close()

            return

        super().keyPressEvent(event)

    # =========================================================
    # CREATE UI
    # =========================================================

    def create_ui(self):

        central_widget = QWidget()

        self.setCentralWidget(
            central_widget
        )

        main_layout = QVBoxLayout(
            central_widget
        )

        main_layout.setContentsMargins(
            12,
            12,
            12,
            12
        )

        main_layout.setSpacing(
            8
        )

        # =====================================================
        # TOP BAR
        # =====================================================

        top_layout = QHBoxLayout()

        top_layout.setSpacing(
            12
        )

        # -----------------------------------------------------
        # TITLE
        # -----------------------------------------------------

        title = QLabel(
            "FOG SIGNAL ANALYZER"
        )

        title.setStyleSheet(
            """
            font-size: 24px;
            font-weight: bold;
            """
        )

        title.setAlignment(
            Qt.AlignLeft |
            Qt.AlignVCenter
        )

        # -----------------------------------------------------
        # PATH
        # -----------------------------------------------------

        self.path_label = QLabel(
            "No dataset selected"
        )

        self.path_label.setAlignment(
            Qt.AlignLeft |
            Qt.AlignVCenter
        )

        self.path_label.setToolTip(
            "Dataset path"
        )

        self.path_label.setStyleSheet(
            """
            color: #555555;
            padding: 4px;
            """
        )

        # -----------------------------------------------------
        # OPEN
        # -----------------------------------------------------

        self.open_button = QPushButton(
            "OPEN DATASET"
        )

        self.open_button.setMinimumWidth(
            160
        )

        self.open_button.setMinimumHeight(
            40
        )

        self.open_button.clicked.connect(
            self.open_dataset
        )

        # -----------------------------------------------------
        # TOP BAR
        # -----------------------------------------------------

        top_layout.addWidget(
            title
        )

        top_layout.addSpacing(110)

        top_layout.addWidget(
            self.open_button
        )

        top_layout.addWidget(
            self.path_label,
            1
        )

        main_layout.addLayout(
            top_layout
        )

        # =====================================================
        # STACKED PAGES
        # =====================================================

        self.pages = QStackedWidget()

        # Dataset page
        self.dataset_page = (
            self.create_dataset_page()
        )

        # Time Domain page
        self.time_domain_page = (
            self.create_time_domain_page()
        )

        self.preprocessing_page = (
            self.create_preprocessing_page()
        )

        # Phase 4: Windowing page.
        self.windowing_page = (
            self.create_windowing_page()
        )

        # Phase 5: Time & Frequency Analysis page.
        self.analysis_page = (
            self.create_analysis_page()
        )

        self.pages.addWidget(
            self.dataset_page
        )

        self.pages.addWidget(
            self.time_domain_page
        )

        self.pages.addWidget(
            self.preprocessing_page
        )

        self.pages.addWidget(
            self.windowing_page
        )

        self.pages.addWidget(
            self.analysis_page
        )

        main_layout.addWidget(
            self.pages
        )

    # =========================================================
    # DATASET PAGE
    # =========================================================

    def create_dataset_page(self):

        page = QWidget()

        content_layout = QHBoxLayout(
            page
        )

        content_layout.setSpacing(
            10
        )

        # =====================================================
        # LEFT PANEL
        # =====================================================

        left_panel = QWidget()

        left_panel.setMinimumWidth(
            360
        )

        left_panel.setMaximumWidth(
            420
        )

        left_layout = QVBoxLayout(
            left_panel
        )

        left_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        left_layout.setSpacing(
            10
        )

        # =====================================================
        # DATASET INFORMATION
        # =====================================================

        dataset_group = QGroupBox(
            "DATASET INFORMATION"
        )

        dataset_layout = QGridLayout(
            dataset_group
        )

        dataset_layout.setHorizontalSpacing(
            18
        )

        dataset_layout.setVerticalSpacing(
            8
        )

        dataset_layout.setColumnStretch(
            0,
            0
        )

        dataset_layout.setColumnStretch(
            1,
            1
        )

        self.file_value = QLabel("-")
        self.columns_value = QLabel("-")
        self.samples_value = QLabel("-")
        self.fs_value = QLabel("-")
        self.duration_value = QLabel("-")

        for label in [
            self.file_value,
            self.columns_value,
            self.samples_value,
            self.fs_value,
            self.duration_value
        ]:

            label.setAlignment(
                Qt.AlignRight |
                Qt.AlignVCenter
            )

        dataset_layout.addWidget(
            QLabel("File:"),
            0,
            0
        )

        dataset_layout.addWidget(
            self.file_value,
            0,
            1
        )

        dataset_layout.addWidget(
            QLabel("Columns:"),
            1,
            0
        )

        dataset_layout.addWidget(
            self.columns_value,
            1,
            1
        )

        dataset_layout.addWidget(
            QLabel("Samples:"),
            2,
            0
        )

        dataset_layout.addWidget(
            self.samples_value,
            2,
            1
        )

        dataset_layout.addWidget(
            QLabel("Sampling Rate:"),
            3,
            0
        )

        dataset_layout.addWidget(
            self.fs_value,
            3,
            1
        )

        dataset_layout.addWidget(
            QLabel("Duration:"),
            4,
            0
        )

        dataset_layout.addWidget(
            self.duration_value,
            4,
            1
        )

        left_layout.addWidget(
            dataset_group
        )

        # =====================================================
        # LABEL CONFIGURATION
        # =====================================================

        label_group = QGroupBox(
            "LABEL CONFIGURATION"
        )

        label_layout = QVBoxLayout(
            label_group
        )

        label_info = QLabel(
            "Detected labels"
        )

        label_info.setAlignment(
            Qt.AlignCenter
        )

        label_layout.addWidget(
            label_info
        )

        self.label_table = QTableWidget()

        self.label_table.setColumnCount(
            3
        )

        self.label_table.setHorizontalHeaderLabels(
            [
                "Original Label",
                "Label Name",
                "Binary"
            ]
        )

        self.label_table.verticalHeader().setVisible(
            False
        )

        self.label_table.setWordWrap(
            True
        )

        self.label_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )

        self.label_table.setSelectionMode(
            QTableWidget.NoSelection
        )

        label_header = (
            self.label_table.horizontalHeader()
        )

        for column in range(3):

            label_header.setSectionResizeMode(
                column,
                QHeaderView.Stretch
            )

        self.label_table.setMinimumHeight(
            170
        )

        label_layout.addWidget(
            self.label_table
        )

        self.apply_label_button = QPushButton(
            "APPLY LABEL MAPPING"
        )

        self.apply_label_button.clicked.connect(
            self.apply_label_mapping
        )

        label_layout.addWidget(
            self.apply_label_button
        )

        left_layout.addWidget(
            label_group
        )

        left_layout.addStretch(
            1
        )

        # =====================================================
        # RIGHT PANEL
        # =====================================================

        right_panel = QWidget()

        right_layout = QVBoxLayout(
            right_panel
        )

        right_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        # =====================================================
        # PREVIEW
        # =====================================================

        preview_group = QGroupBox(
            "DATASET PREVIEW"
        )

        preview_layout = QVBoxLayout(
            preview_group
        )

        preview_layout.setContentsMargins(
            6,
            6,
            6,
            6
        )

        self.preview_table = QTableWidget()

        self.preview_table.setWordWrap(
            True
        )

        self.preview_table.setAlternatingRowColors(
            True
        )

        self.preview_table.verticalHeader().setVisible(
            False
        )

        self.preview_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        preview_layout.addWidget(
            self.preview_table
        )

        # =====================================================
        # TIME DOMAIN BUTTON
        # =====================================================

        self.time_domain_button = QPushButton(
            "TIME DOMAIN"
        )

        self.time_domain_button.setMinimumHeight(
            42
        )

        self.time_domain_button.clicked.connect(
            self.show_time_domain
        )

        preview_layout.addWidget(
            self.time_domain_button
        )

        right_layout.addWidget(
            preview_group
        )

        # =====================================================
        # ADD PANELS
        # =====================================================

        content_layout.addWidget(
            left_panel,
            2
        )

        content_layout.addWidget(
            right_panel,
            5
        )

        return page

    # =========================================================
    # TIME DOMAIN PAGE
    # =========================================================

    def create_time_domain_page(self):

        page = QWidget()

        main_layout = QVBoxLayout(
            page
        )

        main_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        main_layout.setSpacing(
            8
        )

        # =====================================================
        # HEADER
        # =====================================================

        header_layout = QHBoxLayout()

        title = QLabel(
            "TIME DOMAIN"
        )

        title.setStyleSheet(
            """
            font-size: 22px;
            font-weight: bold;
            """
        )

        self.back_dataset_button = QPushButton(
            "BACK TO DATASET"
        )

        self.back_dataset_button.clicked.connect(
            self.show_dataset_page
        )

        self.reset_view_button = QPushButton(
            "RESET VIEW"
        )

        self.reset_view_button.clicked.connect(
            self.reset_view
        )

        self.preprocessing_button = QPushButton(
            "PREPROCESSING"
        )

        self.preprocessing_button.clicked.connect(
            self.show_preprocessing
        )

        header_layout.addWidget(
            title
        )

        header_layout.addStretch()

        header_layout.addWidget(
            self.preprocessing_button
        )

        header_layout.addWidget(
            self.reset_view_button
        )

        header_layout.addWidget(
            self.back_dataset_button
        )

        main_layout.addLayout(
            header_layout
        )

        # =====================================================
        # CONTROL BAR
        # =====================================================

        control_layout = QHBoxLayout()

        # =====================================================
        # SENSOR SELECT
        # =====================================================

        sensor_group = QGroupBox(
            "SIGNAL"
        )

        sensor_layout = QHBoxLayout(
            sensor_group
        )

        sensor_layout.setSpacing(
            5
        )

        self.ankle_button = QPushButton(
            "ANKLE"
        )

        self.thigh_button = QPushButton(
            "THIGH"
        )

        self.trunk_button = QPushButton(
            "TRUNK"
        )

        for button in [
            self.ankle_button,
            self.thigh_button,
            self.trunk_button
        ]:

            button.setCheckable(
                True
            )

            button.setMinimumWidth(
                90
            )

        self.ankle_button.setChecked(
            True
        )

        self.ankle_button.clicked.connect(
            lambda:
            self.select_sensor("Ankle")
        )

        self.thigh_button.clicked.connect(
            lambda:
            self.select_sensor("Thigh")
        )

        self.trunk_button.clicked.connect(
            lambda:
            self.select_sensor("Trunk")
        )

        sensor_layout.addWidget(
            self.ankle_button
        )

        sensor_layout.addWidget(
            self.thigh_button
        )

        sensor_layout.addWidget(
            self.trunk_button
        )

        control_layout.addWidget(
            sensor_group
        )

        # =====================================================
        # TIME RANGE
        # =====================================================

        range_group = QGroupBox(
            "TIME RANGE"
        )

        range_layout = QGridLayout(
            range_group
        )

        range_layout.setHorizontalSpacing(
            8
        )

        # -----------------------------------------------------
        # START
        # -----------------------------------------------------

        range_layout.addWidget(
            QLabel("Start:"),
            0,
            0
        )

        self.start_time_spinbox = (
            QDoubleSpinBox()
        )

        self.start_time_spinbox.setDecimals(
            3
        )

        self.start_time_spinbox.setSuffix(
            " s"
        )

        self.start_time_spinbox.setRange(
            0,
            999999999
        )

        self.start_time_spinbox.setValue(
            0
        )

        range_layout.addWidget(
            self.start_time_spinbox,
            0,
            1
        )

        # -----------------------------------------------------
        # END
        # -----------------------------------------------------

        range_layout.addWidget(
            QLabel("End:"),
            0,
            2
        )

        self.end_time_spinbox = (
            QDoubleSpinBox()
        )

        self.end_time_spinbox.setDecimals(
            3
        )

        self.end_time_spinbox.setSuffix(
            " s"
        )

        self.end_time_spinbox.setRange(
            0,
            999999999
        )

        range_layout.addWidget(
            self.end_time_spinbox,
            0,
            3
        )

        # -----------------------------------------------------
        # ENTIRE DATASET
        # -----------------------------------------------------

        self.entire_dataset_checkbox = (
            QCheckBox(
                "Entire dataset"
            )
        )

        self.entire_dataset_checkbox.setChecked(
            True
        )

        self.entire_dataset_checkbox.stateChanged.connect(
            self.toggle_time_range
        )

        range_layout.addWidget(
            self.entire_dataset_checkbox,
            1,
            0,
            1,
            2
        )

        # -----------------------------------------------------
        # LOAD
        # -----------------------------------------------------

        self.load_range_button = QPushButton(
            "LOAD TIME RANGE"
        )

        self.load_range_button.clicked.connect(
            self.load_time_range
        )

        range_layout.addWidget(
            self.load_range_button,
            1,
            2,
            1,
            2
        )

        control_layout.addWidget(
            range_group,
            1
        )

        main_layout.addLayout(
            control_layout
        )

        # =====================================================
        # PLOT
        # =====================================================

        self.plot_widget = pg.PlotWidget()

        self.plot_widget.setBackground(
            "w"
        )

        self.plot_widget.showGrid(
            x=True,
            y=True,
            alpha=0.25
        )

        self.plot_widget.setLabel(
            "bottom",
            "Time",
            units="s"
        )

        self.plot_widget.setLabel(
            "left",
            "Acceleration",
            units="g"
        )

        self.plot_widget.setTitle(
            "Ankle Signal"
        )

        # =====================================================
        # LEGEND
        # =====================================================

        self.legend = (
            self.plot_widget.addLegend(
                offset=(10, 10)
            )
        )

        # =====================================================
        # SECOND Y AXIS - FOG
        # =====================================================

        plot_item = (
            self.plot_widget.getPlotItem()
        )

        plot_item.showAxis(
            "right"
        )

        self.fog_axis = (
            plot_item.getAxis(
                "right"
            )
        )

        self.fog_axis.setLabel(
            "FOG State",
            color="#d62728"
        )

        # -----------------------------------------------------
        # FOG VIEWBOX
        # -----------------------------------------------------

        self.fog_view = pg.ViewBox()

        plot_item.scene().addItem(
            self.fog_view
        )

        self.fog_axis.linkToView(
            self.fog_view
        )

        self.fog_view.setXLink(
            plot_item.vb
        )

        plot_item.vb.sigResized.connect(
            self.update_fog_view
        )

        self.fog_view.setYRange(
            -0.1,
            1.1,
            padding=0
        )

        # =====================================================
        # PHASE 2 - SMOOTH ZOOM / PAN
        # =====================================================

        # Chỉ cho phép Zoom / Pan theo trục X.
        # Y-axis được cố định nên không tự động thay đổi khi zoom/pan.
        plot_item.vb.setMouseEnabled(
            x=True,
            y=False
        )

        self.plot_widget.enableAutoRange(
            x=False,
            y=False
        )
        # Bật chế độ Pan
        plot_item.vb.setMouseMode(pg.ViewBox.PanMode)
        plot_item.vb.setMenuEnabled(False)

        # Tắt antialias để giảm tải khi vẽ dataset dài.
        self.plot_widget.setAntialiasing(False)

        # =====================================================
        # ADD PLOT
        # =====================================================

        main_layout.addWidget(
            self.plot_widget,
            1
        )

        # =====================================================
        # STATUS
        # =====================================================

        self.time_domain_status = QLabel(
            "No signal loaded."
        )

        self.time_domain_status.setAlignment(
            Qt.AlignCenter
        )

        self.time_domain_status.setStyleSheet(
            """
            color: #555555;
            padding: 4px;
            """
        )

        main_layout.addWidget(
            self.time_domain_status
        )

        return page

    # =========================================================
    # UPDATE FOG VIEW
    # =========================================================

    def update_fog_view(self):

        plot_item = (
            self.plot_widget.getPlotItem()
        )

        self.fog_view.setGeometry(
            plot_item.vb.sceneBoundingRect()
        )

        self.fog_view.linkedViewChanged(
            plot_item.vb,
            self.fog_view.XAxis
        )

        self.fog_view.setYRange(
            -0.1,
            1.1,
            padding=0
        )

    # =========================================================
    # FIXED Y SCALE
    # =========================================================

    def update_fixed_y_scale(self):

        if self.current_signal_data is None:

            return

        try:

            finite_mask = np.isfinite(
                self.current_signal_data
            )

            if not finite_mask.any():

                return

            y_min = np.min(
                self.current_signal_data[
                    finite_mask
                ]
            )

            y_max = np.max(
                self.current_signal_data[
                    finite_mask
                ]
            )

            if not np.isfinite(y_min) or not np.isfinite(y_max):

                return

            amplitude = y_max - y_min

            # Tránh trường hợp tín hiệu gần như phẳng.
            min_amplitude = 0.01

            if amplitude < min_amplitude:

                center = (y_max + y_min) / 2.0
                amplitude = min_amplitude
                y_min = center - amplitude / 2.0
                y_max = center + amplitude / 2.0

            # 5% khoảng đệm để waveform không chạm biên.
            padding = amplitude * 0.05

            self.fixed_y_range = (
                y_min - padding,
                y_max + padding
            )

            self.plot_widget.setYRange(
                self.fixed_y_range[0],
                self.fixed_y_range[1],
                padding=0
            )

        except Exception:

            self.fixed_y_range = None


    # =========================================================
    # RESET VIEW
    # =========================================================

    def _force_time_domain_antialias(self, *args):
        if not hasattr(self, "plot_widget"):
            return
        try:
            self.plot_widget.setAntialiasing(True)
            for item in list(self.plot_widget.getPlotItem().listDataItems()):
                if hasattr(item, "setAntialias"):
                    item.setAntialias(True)
        except Exception:
            pass

    def reset_view(self):

        if self.current_time_array is None:

            return

        time_array = (
            self.current_time_array
        )

        if len(time_array) == 0:

            return

        # Reset X
        self.plot_widget.setXRange(
            float(time_array[0]),
            float(time_array[-1]),
            padding=0
        )

        # Giữ nguyên Y-range cố định.
        if self.fixed_y_range is not None:

            self.plot_widget.setYRange(
                self.fixed_y_range[0],
                self.fixed_y_range[1],
                padding=0
            )

    # =========================================================
    # SELECT SENSOR
    # =========================================================

    def select_sensor(
        self,
        sensor
    ):

        self.current_sensor = sensor

        self.ankle_button.setChecked(
            sensor == "Ankle"
        )

        self.thigh_button.setChecked(
            sensor == "Thigh"
        )

        self.trunk_button.setChecked(
            sensor == "Trunk"
        )

        if self.df is not None:

            self.load_time_range()

    # =========================================================
    # SHOW TIME DOMAIN
    # =========================================================

    def show_time_domain(self):

        if self.df is None:

            QMessageBox.warning(
                self,
                "No Dataset",
                "Hãy mở dataset trước."
            )

            return

        self.display_df = self.df.copy()
        self.display_sampling_rate = self.sampling_rate
        self.display_processed_label = False

        self.pages.setCurrentWidget(
            self.time_domain_page
        )

        self.update_time_controls()

        self.load_time_range()

    # =========================================================
    # PHASE 3 - PREPROCESSING PAGE
    # =========================================================

    def create_preprocessing_page(self):

        page = QWidget()

        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        header_layout = QHBoxLayout()

        title = QLabel("PREPROCESSING")
        title.setStyleSheet(
            """
            font-size: 22px;
            font-weight: bold;
            """
        )

        self.preprocessing_back_button = QPushButton(
            "BACK TO TIME DOMAIN"
        )
        self.preprocessing_back_button.clicked.connect(
            self.show_time_domain_from_preprocessing
        )

        self.preprocessing_reset_button = QPushButton("RESET")
        self.preprocessing_reset_button.clicked.connect(
            self.reset_preprocessing
        )

        self.preprocessing_apply_button = QPushButton(
            "APPLY PREPROCESSING"
        )
        self.preprocessing_apply_button.clicked.connect(
            self.apply_preprocessing
        )

        # Chuyển sang Phase 4 sau khi đã có dữ liệu preprocessing.
        self.windowing_button = QPushButton(
            "WINDOWING"
        )
        self.windowing_button.clicked.connect(
            self.show_windowing
        )

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.preprocessing_reset_button)
        header_layout.addWidget(self.preprocessing_apply_button)
        header_layout.addWidget(self.windowing_button)
        header_layout.addWidget(self.preprocessing_back_button)
        main_layout.addLayout(header_layout)

        pipeline_label = QLabel(
            "Pipeline:  Resample  →  Filter  →  Detrending  →  Select Region"
        )
        pipeline_label.setAlignment(Qt.AlignCenter)
        pipeline_label.setStyleSheet(
            "font-weight: bold; padding: 6px;"
        )
        main_layout.addWidget(pipeline_label)

        content_layout = QHBoxLayout()

        settings_panel = QWidget()
        settings_panel.setMinimumWidth(360)
        settings_panel.setMaximumWidth(460)
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(10)

        # RESAMPLE
        resample_group = QGroupBox("RESAMPLE")
        resample_layout = QGridLayout(resample_group)

        self.resample_checkbox = QCheckBox("Enable resampling")
        self.resample_checkbox.setChecked(False)
        self.resample_checkbox.stateChanged.connect(
            self.toggle_resample_controls
        )
        resample_layout.addWidget(self.resample_checkbox, 0, 0, 1, 2)

        resample_layout.addWidget(QLabel("Original Rate:"), 1, 0)
        self.preprocessing_original_fs_label = QLabel("-")
        self.preprocessing_original_fs_label.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        resample_layout.addWidget(
            self.preprocessing_original_fs_label, 1, 1
        )

        resample_layout.addWidget(QLabel("Target Rate:"), 2, 0)
        self.target_fs_spinbox = QDoubleSpinBox()
        self.target_fs_spinbox.setDecimals(2)
        self.target_fs_spinbox.setRange(1, 1000)
        self.target_fs_spinbox.setValue(64)
        self.target_fs_spinbox.setSuffix(" Hz")
        self.target_fs_spinbox.setEnabled(False)
        resample_layout.addWidget(self.target_fs_spinbox, 2, 1)
        settings_layout.addWidget(resample_group)

        # FILTER
        filter_group = QGroupBox("FILTER")
        filter_layout = QGridLayout(filter_group)

        self.filter_checkbox = QCheckBox("Enable filter")
        self.filter_checkbox.setChecked(False)
        self.filter_checkbox.stateChanged.connect(
            self.toggle_filter_controls
        )
        filter_layout.addWidget(self.filter_checkbox, 0, 0, 1, 2)

        filter_layout.addWidget(QLabel("Type:"), 1, 0)
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems([
            "Low-pass",
            "High-pass",
            "Band-pass"
        ])
        self.filter_type_combo.setEnabled(False)
        self.filter_type_combo.currentTextChanged.connect(
            self.update_filter_cutoff_controls
        )
        filter_layout.addWidget(self.filter_type_combo, 1, 1)

        filter_layout.addWidget(QLabel("Cutoff 1:"), 2, 0)
        self.cutoff_1_spinbox = QDoubleSpinBox()
        self.cutoff_1_spinbox.setDecimals(2)
        self.cutoff_1_spinbox.setRange(0.01, 500)
        self.cutoff_1_spinbox.setValue(6.0)
        self.cutoff_1_spinbox.setSuffix(" Hz")
        self.cutoff_1_spinbox.setEnabled(False)
        filter_layout.addWidget(self.cutoff_1_spinbox, 2, 1)

        filter_layout.addWidget(QLabel("Cutoff 2:"), 3, 0)
        self.cutoff_2_spinbox = QDoubleSpinBox()
        self.cutoff_2_spinbox.setDecimals(2)
        self.cutoff_2_spinbox.setRange(0.01, 500)
        self.cutoff_2_spinbox.setValue(12.0)
        self.cutoff_2_spinbox.setSuffix(" Hz")
        self.cutoff_2_spinbox.setEnabled(False)
        filter_layout.addWidget(self.cutoff_2_spinbox, 3, 1)

        filter_layout.addWidget(QLabel("Order:"), 4, 0)
        self.filter_order_spinbox = QDoubleSpinBox()
        self.filter_order_spinbox.setDecimals(0)
        self.filter_order_spinbox.setRange(1, 12)
        self.filter_order_spinbox.setValue(4)
        self.filter_order_spinbox.setEnabled(False)
        filter_layout.addWidget(self.filter_order_spinbox, 4, 1)
        settings_layout.addWidget(filter_group)

        # DETRENDING
        detrend_group = QGroupBox("DETRENDING")
        detrend_layout = QGridLayout(detrend_group)

        self.detrend_checkbox = QCheckBox("Enable detrending")
        self.detrend_checkbox.setChecked(False)
        self.detrend_checkbox.stateChanged.connect(
            self.toggle_detrend_controls
        )
        detrend_layout.addWidget(self.detrend_checkbox, 0, 0, 1, 2)

        detrend_layout.addWidget(QLabel("Method:"), 1, 0)
        self.detrend_type_combo = QComboBox()
        self.detrend_type_combo.addItems(["Linear", "Constant"])
        self.detrend_type_combo.setEnabled(False)
        detrend_layout.addWidget(self.detrend_type_combo, 1, 1)
        settings_layout.addWidget(detrend_group)

        # SELECT REGION
        region_group = QGroupBox("SELECT REGION")
        region_layout = QGridLayout(region_group)

        self.preprocessing_entire_dataset_checkbox = QCheckBox(
            "Entire dataset"
        )
        self.preprocessing_entire_dataset_checkbox.setChecked(True)
        self.preprocessing_entire_dataset_checkbox.stateChanged.connect(
            self.toggle_preprocessing_region_controls
        )
        region_layout.addWidget(
            self.preprocessing_entire_dataset_checkbox, 0, 0, 1, 2
        )

        region_layout.addWidget(QLabel("Start:"), 1, 0)
        self.preprocessing_start_spinbox = QDoubleSpinBox()
        self.preprocessing_start_spinbox.setDecimals(3)
        self.preprocessing_start_spinbox.setRange(0, 999999999)
        self.preprocessing_start_spinbox.setSuffix(" s")
        self.preprocessing_start_spinbox.setEnabled(False)
        region_layout.addWidget(
            self.preprocessing_start_spinbox, 1, 1
        )

        region_layout.addWidget(QLabel("End:"), 2, 0)
        self.preprocessing_end_spinbox = QDoubleSpinBox()
        self.preprocessing_end_spinbox.setDecimals(3)
        self.preprocessing_end_spinbox.setRange(0, 999999999)
        self.preprocessing_end_spinbox.setSuffix(" s")
        self.preprocessing_end_spinbox.setEnabled(False)
        region_layout.addWidget(
            self.preprocessing_end_spinbox, 2, 1
        )
        settings_layout.addWidget(region_group)
        settings_layout.addStretch(1)
        content_layout.addWidget(settings_panel)

        # RESULT / PREVIEW
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)

        info_group = QGroupBox("PROCESSING INFORMATION")
        info_layout = QGridLayout(info_group)

        self.preprocessing_status_value = QLabel("Not applied")
        self.preprocessing_fs_value = QLabel("-")
        self.preprocessing_samples_value = QLabel("-")
        self.preprocessing_duration_value = QLabel("-")
        self.preprocessing_operations_value = QLabel("None")

        for label in [
            self.preprocessing_status_value,
            self.preprocessing_fs_value,
            self.preprocessing_samples_value,
            self.preprocessing_duration_value,
            self.preprocessing_operations_value
        ]:
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        for row, name, widget in [
            (0, "Status:", self.preprocessing_status_value),
            (1, "Sampling Rate:", self.preprocessing_fs_value),
            (2, "Samples:", self.preprocessing_samples_value),
            (3, "Duration:", self.preprocessing_duration_value),
            (4, "Operations:", self.preprocessing_operations_value)
        ]:
            info_layout.addWidget(QLabel(name), row, 0)
            info_layout.addWidget(widget, row, 1)

        result_layout.addWidget(info_group)

        preview_group = QGroupBox("PROCESSED DATA PREVIEW")
        preview_layout = QVBoxLayout(preview_group)
        self.preprocessing_preview_table = QTableWidget()
        self.preprocessing_preview_table.setWordWrap(True)
        self.preprocessing_preview_table.setAlternatingRowColors(True)
        self.preprocessing_preview_table.verticalHeader().setVisible(False)
        self.preprocessing_preview_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        preview_layout.addWidget(self.preprocessing_preview_table)
        result_layout.addWidget(preview_group, 1)

        self.view_processed_button = QPushButton(
            "VIEW PROCESSED TIME DOMAIN"
        )
        self.view_processed_button.setMinimumHeight(42)
        self.view_processed_button.clicked.connect(
            self.view_processed_time_domain
        )
        result_layout.addWidget(self.view_processed_button)

        content_layout.addWidget(result_panel, 1)
        main_layout.addLayout(content_layout, 1)

        self.update_filter_cutoff_controls()
        return page

    def toggle_resample_controls(self):

        self.target_fs_spinbox.setEnabled(
            self.resample_checkbox.isChecked()
        )

    def toggle_filter_controls(self):

        enabled = self.filter_checkbox.isChecked()
        self.filter_type_combo.setEnabled(enabled)
        self.filter_order_spinbox.setEnabled(enabled)
        self.cutoff_1_spinbox.setEnabled(enabled)
        self.update_filter_cutoff_controls()

    def update_filter_cutoff_controls(self):

        enabled = self.filter_checkbox.isChecked()
        band_pass = (
            self.filter_type_combo.currentText() == "Band-pass"
        )
        self.cutoff_1_spinbox.setEnabled(enabled)
        self.cutoff_2_spinbox.setEnabled(
            enabled and band_pass
        )

    def toggle_detrend_controls(self):

        self.detrend_type_combo.setEnabled(
            self.detrend_checkbox.isChecked()
        )

    def toggle_preprocessing_region_controls(self):

        enabled = not self.preprocessing_entire_dataset_checkbox.isChecked()
        self.preprocessing_start_spinbox.setEnabled(enabled)
        self.preprocessing_end_spinbox.setEnabled(enabled)

    def show_preprocessing(self):

        if self.df is None:
            QMessageBox.warning(
                self,
                "No Dataset",
                "Hãy mở dataset trước."
            )
            return

        self.pages.setCurrentWidget(self.preprocessing_page)
        self.update_preprocessing_controls()

    def show_time_domain_from_preprocessing(self):

        if self.display_df is None and self.df is not None:
            self.display_df = self.df.copy()
            self.display_sampling_rate = self.sampling_rate
            self.display_processed_label = False

        self.pages.setCurrentWidget(self.time_domain_page)
        self.update_time_controls()
        self.load_time_range()

    def update_preprocessing_controls(self):

        if self.df_raw is None:
            return

        if self.sampling_rate is not None:
            self.preprocessing_original_fs_label.setText(
                f"{self.sampling_rate:g} Hz"
            )
            self.target_fs_spinbox.setValue(self.sampling_rate)
        else:
            self.preprocessing_original_fs_label.setText("Unknown")

        duration = self.get_dataframe_duration(
            self.df_raw,
            self.sampling_rate
        )

        if duration is not None:
            self.preprocessing_start_spinbox.setRange(0, duration)
            self.preprocessing_end_spinbox.setRange(0, duration)
            self.preprocessing_start_spinbox.setValue(0)
            self.preprocessing_end_spinbox.setValue(duration)

        self.preprocessing_entire_dataset_checkbox.setChecked(True)
        self.toggle_preprocessing_region_controls()

    def get_sensor_columns(self):

        return [
            "Ankle_Forward_mg",
            "Ankle_Vertical_mg",
            "Ankle_Lateral_mg",
            "Thigh_Forward_mg",
            "Thigh_Vertical_mg",
            "Thigh_Lateral_mg",
            "Trunk_Forward_mg",
            "Trunk_Vertical_mg",
            "Trunk_Lateral_mg"
        ]

    def get_dataframe_time_array(self, dataframe, sampling_rate):

        if "Time_ms" in dataframe.columns:
            return (
                dataframe["Time_ms"]
                .to_numpy(dtype=float) / 1000.0
            )

        if sampling_rate:
            return (
                np.arange(len(dataframe), dtype=float)
                / sampling_rate
            )

        return None

    def get_dataframe_duration(self, dataframe, sampling_rate):

        time_array = self.get_dataframe_time_array(
            dataframe,
            sampling_rate
        )

        if time_array is None or len(time_array) == 0:
            return None

        if len(time_array) == 1:
            return 0.0

        return float(time_array[-1] - time_array[0])

    def nearest_label_values(self, original_time, original_values, new_time):

        indices = np.searchsorted(
            original_time,
            new_time,
            side="left"
        )

        indices = np.clip(
            indices,
            0,
            len(original_time) - 1
        )

        left_indices = np.maximum(indices - 1, 0)

        choose_left = (
            np.abs(new_time - original_time[left_indices])
            <=
            np.abs(new_time - original_time[indices])
        )

        nearest_indices = np.where(
            choose_left,
            left_indices,
            indices
        )

        return original_values[nearest_indices]

    def resample_dataframe(self, dataframe, source_fs, target_fs):

        if source_fs is None or source_fs <= 0:
            raise ValueError(
                "Không xác định được Sampling Rate để Resample."
            )

        if target_fs <= 0:
            raise ValueError(
                "Target Sampling Rate phải lớn hơn 0."
            )

        if len(dataframe) < 2:
            return dataframe.copy()

        original_time = self.get_dataframe_time_array(
            dataframe, source_fs
        )

        if original_time is None:
            raise ValueError("Không tạo được Time array.")

        if np.isclose(source_fs, target_fs):
            return dataframe.copy()

        start_time = float(original_time[0])
        end_time = float(original_time[-1])
        duration = end_time - start_time

        new_count = max(
            2,
            int(round(duration * target_fs)) + 1
        )

        new_time = np.linspace(
            start_time,
            end_time,
            new_count
        )

        ratio = (
            Fraction(float(target_fs)).limit_denominator(1000)
            /
            Fraction(float(source_fs)).limit_denominator(1000)
        )

        up = ratio.numerator
        down = ratio.denominator

        output = pd.DataFrame(
            index=np.arange(new_count)
        )
        output["Time_ms"] = new_time * 1000.0

        for column in [
            c for c in self.get_sensor_columns()
            if c in dataframe.columns
        ]:
            values = dataframe[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Signal '{column}' chứa NaN/Inf."
                )

            resampled_values = signal.resample_poly(
                values,
                up,
                down
            )

            old_index = np.linspace(
                0, 1, len(resampled_values)
            )
            new_index = np.linspace(0, 1, new_count)

            output[column] = np.interp(
                new_index,
                old_index,
                resampled_values
            )

        for column in [
            "Label",
            "Original_Label",
            "Binary_Label"
        ]:
            if column in dataframe.columns:
                output[column] = self.nearest_label_values(
                    original_time,
                    dataframe[column].to_numpy(),
                    new_time
                )

        return output

    def apply_filter_to_dataframe(self, dataframe, fs):

        if fs is None or fs <= 0:
            raise ValueError(
                "Sampling Rate không hợp lệ cho Filter."
            )

        nyquist = fs / 2.0
        filter_type = self.filter_type_combo.currentText()
        order = int(self.filter_order_spinbox.value())
        cutoff_1 = self.cutoff_1_spinbox.value()

        if filter_type == "Low-pass":
            if cutoff_1 <= 0 or cutoff_1 >= nyquist:
                raise ValueError(
                    "Low-pass cutoff phải thỏa 0 < cutoff < Nyquist."
                )
            b, a = signal.butter(
                order, cutoff_1, btype="lowpass", fs=fs
            )

        elif filter_type == "High-pass":
            if cutoff_1 <= 0 or cutoff_1 >= nyquist:
                raise ValueError(
                    "High-pass cutoff phải thỏa 0 < cutoff < Nyquist."
                )
            b, a = signal.butter(
                order, cutoff_1, btype="highpass", fs=fs
            )

        else:
            cutoff_2 = self.cutoff_2_spinbox.value()
            if (
                cutoff_1 <= 0
                or cutoff_2 <= cutoff_1
                or cutoff_2 >= nyquist
            ):
                raise ValueError(
                    "Band-pass cần Cutoff 1 < Cutoff 2 < Nyquist."
                )
            b, a = signal.butter(
                order, [cutoff_1, cutoff_2],
                btype="bandpass", fs=fs
            )

        processed = dataframe.copy()

        for column in self.get_sensor_columns():
            if column not in processed.columns:
                continue

            values = processed[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Signal '{column}' chứa NaN/Inf."
                )

            minimum_length = max(len(a), len(b)) * 3

            if len(values) <= minimum_length:
                filtered = signal.lfilter(b, a, values)
            else:
                filtered = signal.filtfilt(b, a, values)

            processed[column] = filtered

        return processed

    def apply_detrend_to_dataframe(self, dataframe):

        processed = dataframe.copy()
        detrend_type = (
            "linear"
            if self.detrend_type_combo.currentText() == "Linear"
            else "constant"
        )

        for column in self.get_sensor_columns():
            if column not in processed.columns:
                continue

            values = processed[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Signal '{column}' chứa NaN/Inf."
                )

            processed[column] = signal.detrend(
                values,
                type=detrend_type
            )

        return processed

    def select_region_from_dataframe(
        self, dataframe, start_time, end_time, fs
    ):

        time_array = self.get_dataframe_time_array(
            dataframe, fs
        )

        if time_array is None:
            raise ValueError(
                "Không có Time array để Select Region."
            )

        mask = (
            (time_array >= start_time)
            &
            (time_array <= end_time)
        )

        selected = dataframe.loc[mask].copy()

        if selected.empty:
            raise ValueError(
                "Không có dữ liệu trong region đã chọn."
            )

        selected.reset_index(
            drop=True,
            inplace=True
        )

        return selected

    def show_preprocessing_preview(self, dataframe):

        if dataframe is None:
            self.preprocessing_preview_table.clear()
            self.preprocessing_preview_table.setRowCount(0)
            return

        preview_df = dataframe.head(12)

        self.preprocessing_preview_table.clear()
        self.preprocessing_preview_table.setRowCount(
            len(preview_df)
        )
        self.preprocessing_preview_table.setColumnCount(
            len(preview_df.columns)
        )

        headers = [
            str(column).replace("_", "\n")
            for column in preview_df.columns
        ]

        self.preprocessing_preview_table.setHorizontalHeaderLabels(
            headers
        )

        for row in range(len(preview_df)):
            for column in range(len(preview_df.columns)):
                item = QTableWidgetItem(
                    str(preview_df.iloc[row, column])
                )
                item.setTextAlignment(
                    Qt.AlignCenter | Qt.AlignVCenter
                )
                self.preprocessing_preview_table.setItem(
                    row, column, item
                )

        header = self.preprocessing_preview_table.horizontalHeader()
        for column in range(len(preview_df.columns)):
            header.setSectionResizeMode(
                column, QHeaderView.Stretch
            )

        self.preprocessing_preview_table.horizontalHeader().setMinimumHeight(65)
        self.preprocessing_preview_table.verticalHeader().setDefaultSectionSize(27)

    def apply_preprocessing(self):

        if self.df_raw is None:
            QMessageBox.warning(
                self,
                "No Dataset",
                "Hãy mở dataset trước."
            )
            return

        try:
            # Always start from RAW.
            working_df = self.df_raw.copy()
            source_fs = self.sampling_rate

            if source_fs is None:
                raise ValueError(
                    "Sampling Rate hiện chưa xác định."
                )

            current_fs = float(source_fs)
            operations = []

            # 1. RESAMPLE
            if self.resample_checkbox.isChecked():
                target_fs = float(
                    self.target_fs_spinbox.value()
                )
                working_df = self.resample_dataframe(
                    working_df,
                    current_fs,
                    target_fs
                )
                current_fs = target_fs
                operations.append(
                    f"Resample {target_fs:g} Hz"
                )

            # 2. FILTER
            if self.filter_checkbox.isChecked():
                working_df = self.apply_filter_to_dataframe(
                    working_df,
                    current_fs
                )

                filter_type = self.filter_type_combo.currentText()
                cutoff_1 = self.cutoff_1_spinbox.value()

                if filter_type == "Band-pass":
                    cutoff_2 = self.cutoff_2_spinbox.value()
                    operations.append(
                        f"{filter_type} "
                        f"{cutoff_1:g}-{cutoff_2:g} Hz"
                    )
                else:
                    operations.append(
                        f"{filter_type} {cutoff_1:g} Hz"
                    )

                operations.append(
                    f"Order {int(self.filter_order_spinbox.value())}"
                )

            # 3. DETRENDING
            if self.detrend_checkbox.isChecked():
                working_df = self.apply_detrend_to_dataframe(
                    working_df
                )
                operations.append(
                    f"Detrend {self.detrend_type_combo.currentText()}"
                )

            # 4. SELECT REGION
            time_array = self.get_dataframe_time_array(
                working_df, current_fs
            )

            if time_array is None or len(time_array) == 0:
                raise ValueError(
                    "Không có Time array sau preprocessing."
                )

            if self.preprocessing_entire_dataset_checkbox.isChecked():
                start_time = float(time_array[0])
                end_time = float(time_array[-1])
            else:
                start_time = self.preprocessing_start_spinbox.value()
                end_time = self.preprocessing_end_spinbox.value()

                if end_time <= start_time:
                    raise ValueError(
                        "End Time phải lớn hơn Start Time."
                    )

                if (
                    start_time < time_array[0]
                    or end_time > time_array[-1]
                ):
                    raise ValueError(
                        "Region nằm ngoài phạm vi dữ liệu."
                    )

            working_df = self.select_region_from_dataframe(
                working_df,
                start_time,
                end_time,
                current_fs
            )

            operations.append(
                f"Region {start_time:.3f}-{end_time:.3f} s"
            )

            self.df_processed = working_df.copy()
            self.df_selected = working_df.copy()
            self.processed_sampling_rate = current_fs
            self.processed_start_time = start_time
            self.processed_end_time = end_time
            self.preprocessing_applied = True

            self.preprocessing_status_value.setText("Applied")
            self.preprocessing_fs_value.setText(
                f"{current_fs:g} Hz"
            )
            self.preprocessing_samples_value.setText(
                f"{len(working_df):,}"
            )

            processed_duration = self.get_dataframe_duration(
                working_df, current_fs
            )

            self.preprocessing_duration_value.setText(
                f"{processed_duration:.3f} sec"
                if processed_duration is not None
                else "-"
            )

            self.preprocessing_operations_value.setText(
                " → ".join(operations)
            )

            self.show_preprocessing_preview(working_df)

            QMessageBox.information(
                self,
                "Preprocessing",
                "Preprocessing đã được áp dụng."
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "Preprocessing Error",
                str(error)
            )

    def reset_preprocessing(self):

        self.preprocessing_applied = False
        self.df_processed = None
        self.df_selected = None
        self.processed_sampling_rate = None
        self.processed_start_time = None
        self.processed_end_time = None

        self.resample_checkbox.setChecked(False)
        self.filter_checkbox.setChecked(False)
        self.detrend_checkbox.setChecked(False)

        self.target_fs_spinbox.setValue(
            self.sampling_rate
            if self.sampling_rate is not None
            else 64
        )

        self.preprocessing_status_value.setText("Not applied")
        self.preprocessing_fs_value.setText("-")
        self.preprocessing_samples_value.setText("-")
        self.preprocessing_duration_value.setText("-")
        self.preprocessing_operations_value.setText("None")
        self.show_preprocessing_preview(None)
        self.update_preprocessing_controls()

    def view_processed_time_domain(self):

        if (
            not self.preprocessing_applied
            or self.df_selected is None
        ):
            QMessageBox.warning(
                self,
                "No Processed Data",
                "Hãy APPLY PREPROCESSING trước."
            )
            return

        self.display_df = self.df_selected.copy()
        self.display_sampling_rate = self.processed_sampling_rate
        self.display_processed_label = True

        self.pages.setCurrentWidget(
            self.time_domain_page
        )

        self.update_time_controls()
        self.load_time_range()

    # =========================================================
    # PHASE 4 - WINDOWING PAGE
    # =========================================================

    def create_windowing_page(self):
        """
        PHASE 4 UI:
        - Chọn dữ liệu đầu vào.
        - Thiết lập Window Length / Overlap.
        - Tự động tính Step Size và Samples / Window.
        - Chọn cách gán Label cho từng window.
        - Hiển thị metadata của các window sau khi Generate.

        LƯU Ý:
        Phase 4 chỉ làm Windowing + Label Assignment.
        Feature Extraction sẽ được xây dựng ở bước tiếp theo.
        """

        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # ---------------------------------------------------------
        # HEADER
        # ---------------------------------------------------------
        header_layout = QHBoxLayout()

        title = QLabel("WINDOWING")
        title.setStyleSheet(
            "font-size: 22px; font-weight: bold;"
        )

        self.windowing_reset_button = QPushButton("RESET")
        self.windowing_reset_button.clicked.connect(
            self.reset_windowing
        )

        self.windowing_apply_button = QPushButton(
            "APPLY WINDOWING"
        )
        self.windowing_apply_button.clicked.connect(
            self.apply_windowing
        )

        # Chuyển sang Phase 5 để phân tích một window đã tạo.
        self.analysis_button = QPushButton(
            "TIME & FREQUENCY ANALYSIS"
        )
        self.analysis_button.clicked.connect(
            self.show_analysis
        )

        self.windowing_back_button = QPushButton(
            "BACK TO PREPROCESSING"
        )
        self.windowing_back_button.clicked.connect(
            self.show_preprocessing_from_windowing
        )

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.windowing_reset_button)
        header_layout.addWidget(self.windowing_apply_button)
        header_layout.addWidget(self.analysis_button)
        header_layout.addWidget(self.windowing_back_button)

        main_layout.addLayout(header_layout)

        # ---------------------------------------------------------
        # PIPELINE DESCRIPTION
        # ---------------------------------------------------------
        pipeline_label = QLabel(
            "Pipeline:  Processed Data  →  Window Length  →  Overlap  →  Label Assignment  →  Windows"
        )
        pipeline_label.setAlignment(Qt.AlignCenter)
        pipeline_label.setStyleSheet(
            "font-weight: bold; padding: 6px;"
        )
        main_layout.addWidget(pipeline_label)

        content_layout = QHBoxLayout()

        # =========================================================
        # LEFT PANEL - INPUT + WINDOW PARAMETERS
        # =========================================================
        left_panel = QWidget()
        left_panel.setMinimumWidth(360)
        left_panel.setMaximumWidth(470)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # ---------------------------------------------------------
        # INPUT DATA INFORMATION
        # ---------------------------------------------------------
        input_group = QGroupBox("INPUT DATA")
        input_layout = QGridLayout(input_group)

        self.windowing_source_value = QLabel("-")
        self.windowing_sensor_value = QLabel("-")
        self.windowing_fs_value = QLabel("-")
        self.windowing_samples_input_value = QLabel("-")
        self.windowing_duration_input_value = QLabel("-")

        for label in [
            self.windowing_source_value,
            self.windowing_sensor_value,
            self.windowing_fs_value,
            self.windowing_samples_input_value,
            self.windowing_duration_input_value,
        ]:
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        for row, name, widget in [
            (0, "Source:", self.windowing_source_value),
            (1, "Sensor:", self.windowing_sensor_value),
            (2, "Sampling Rate:", self.windowing_fs_value),
            (3, "Samples:", self.windowing_samples_input_value),
            (4, "Duration:", self.windowing_duration_input_value),
        ]:
            input_layout.addWidget(QLabel(name), row, 0)
            input_layout.addWidget(widget, row, 1)

        left_layout.addWidget(input_group)

        # ---------------------------------------------------------
        # SIGNAL SELECTION
        # ---------------------------------------------------------
        signal_group = QGroupBox("SIGNAL")
        signal_layout = QGridLayout(signal_group)

        signal_layout.addWidget(QLabel("Sensor:"), 0, 0)

        self.windowing_sensor_combo = QComboBox()
        self.windowing_sensor_combo.addItems([
            "Ankle",
            "Thigh",
            "Trunk",
        ])
        self.windowing_sensor_combo.setCurrentText(
            self.current_sensor
        )
        self.windowing_sensor_combo.currentTextChanged.connect(
            self.update_windowing_sensor_info
        )
        signal_layout.addWidget(
            self.windowing_sensor_combo, 0, 1
        )

        self.windowing_axis_x_checkbox = QCheckBox("X / Forward")
        self.windowing_axis_y_checkbox = QCheckBox("Y / Vertical")
        self.windowing_axis_z_checkbox = QCheckBox("Z / Lateral")

        self.windowing_axis_x_checkbox.setChecked(True)
        self.windowing_axis_y_checkbox.setChecked(True)
        self.windowing_axis_z_checkbox.setChecked(True)

        signal_layout.addWidget(
            self.windowing_axis_x_checkbox, 1, 0, 1, 2
        )
        signal_layout.addWidget(
            self.windowing_axis_y_checkbox, 2, 0, 1, 2
        )
        signal_layout.addWidget(
            self.windowing_axis_z_checkbox, 3, 0, 1, 2
        )

        left_layout.addWidget(signal_group)

        # ---------------------------------------------------------
        # WINDOW PARAMETERS
        # ---------------------------------------------------------
        window_group = QGroupBox("WINDOW PARAMETERS")
        window_layout = QGridLayout(window_group)

        window_layout.addWidget(
            QLabel("Window Length:"), 0, 0
        )

        self.window_length_spinbox = QDoubleSpinBox()
        self.window_length_spinbox.setDecimals(3)
        self.window_length_spinbox.setRange(0.010, 3600.0)
        self.window_length_spinbox.setSingleStep(0.25)
        self.window_length_spinbox.setValue(2.0)
        self.window_length_spinbox.setSuffix(" s")
        self.window_length_spinbox.valueChanged.connect(
            self.update_window_parameter_calculation
        )
        window_layout.addWidget(
            self.window_length_spinbox, 0, 1
        )

        window_layout.addWidget(
            QLabel("Overlap:"), 1, 0
        )

        self.window_overlap_spinbox = QDoubleSpinBox()
        self.window_overlap_spinbox.setDecimals(1)
        self.window_overlap_spinbox.setRange(0.0, 99.0)
        self.window_overlap_spinbox.setSingleStep(5.0)
        self.window_overlap_spinbox.setValue(50.0)
        self.window_overlap_spinbox.setSuffix(" %")
        self.window_overlap_spinbox.valueChanged.connect(
            self.update_window_parameter_calculation
        )
        window_layout.addWidget(
            self.window_overlap_spinbox, 1, 1
        )

        window_layout.addWidget(
            QLabel("Step Size:"), 2, 0
        )
        self.window_step_value = QLabel("-")
        self.window_step_value.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        window_layout.addWidget(
            self.window_step_value, 2, 1
        )

        window_layout.addWidget(
            QLabel("Samples / Window:"), 3, 0
        )
        self.window_samples_value = QLabel("-")
        self.window_samples_value.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        window_layout.addWidget(
            self.window_samples_value, 3, 1
        )

        window_layout.addWidget(
            QLabel("Step Samples:"), 4, 0
        )
        self.window_step_samples_value = QLabel("-")
        self.window_step_samples_value.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        window_layout.addWidget(
            self.window_step_samples_value, 4, 1
        )

        window_layout.addWidget(
            QLabel("Number of Windows:"), 5, 0
        )
        self.window_count_calculated_value = QLabel("-")
        self.window_count_calculated_value.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        window_layout.addWidget(
            self.window_count_calculated_value, 5, 1
        )

        left_layout.addWidget(window_group)

        # ---------------------------------------------------------
        # LABEL ASSIGNMENT
        # ---------------------------------------------------------
        label_group = QGroupBox("LABEL ASSIGNMENT")
        label_layout = QGridLayout(label_group)

        label_layout.addWidget(QLabel("Label Mode:"), 0, 0)
        self.window_label_mode_combo = QComboBox()
        self.window_label_mode_combo.addItems([
            "Majority",
            "FOG Threshold",
            "Strict FOG",
        ])
        self.window_label_mode_combo.setCurrentText("Majority")
        self.window_label_mode_combo.currentTextChanged.connect(
            self.update_window_label_controls
        )
        label_layout.addWidget(
            self.window_label_mode_combo, 0, 1
        )

        label_layout.addWidget(
            QLabel("FOG Threshold:"), 1, 0
        )
        self.window_fog_threshold_spinbox = QDoubleSpinBox()
        self.window_fog_threshold_spinbox.setDecimals(1)
        self.window_fog_threshold_spinbox.setRange(0.0, 100.0)
        self.window_fog_threshold_spinbox.setSingleStep(5.0)
        self.window_fog_threshold_spinbox.setValue(50.0)
        self.window_fog_threshold_spinbox.setSuffix(" %")
        label_layout.addWidget(
            self.window_fog_threshold_spinbox, 1, 1
        )

        self.window_label_rule_value = QLabel(
            "Majority: label chiếm đa số quyết định window."
        )
        self.window_label_rule_value.setWordWrap(True)
        label_layout.addWidget(
            self.window_label_rule_value, 2, 0, 1, 2
        )

        left_layout.addWidget(label_group)
        left_layout.addStretch(1)

        content_layout.addWidget(left_panel)

        # =========================================================
        # RIGHT PANEL - RESULT / WINDOW TABLE
        # =========================================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        result_group = QGroupBox("WINDOWING RESULT")
        result_layout = QGridLayout(result_group)

        self.windowing_status_value = QLabel("Not applied")
        self.windowing_result_fs_value = QLabel("-")
        self.windowing_length_result_value = QLabel("-")
        self.windowing_overlap_result_value = QLabel("-")
        self.windowing_step_result_value = QLabel("-")
        self.windowing_count_result_value = QLabel("-")

        for label in [
            self.windowing_status_value,
            self.windowing_result_fs_value,
            self.windowing_length_result_value,
            self.windowing_overlap_result_value,
            self.windowing_step_result_value,
            self.windowing_count_result_value,
        ]:
            label.setAlignment(
                Qt.AlignRight | Qt.AlignVCenter
            )

        for row, name, widget in [
            (0, "Status:", self.windowing_status_value),
            (1, "Sampling Rate:", self.windowing_result_fs_value),
            (2, "Window Length:", self.windowing_length_result_value),
            (3, "Overlap:", self.windowing_overlap_result_value),
            (4, "Step:", self.windowing_step_result_value),
            (5, "Windows:", self.windowing_count_result_value),
        ]:
            result_layout.addWidget(QLabel(name), row, 0)
            result_layout.addWidget(widget, row, 1)

        right_layout.addWidget(result_group)

        table_group = QGroupBox("WINDOW PREVIEW")
        table_layout = QVBoxLayout(table_group)

        self.window_preview_table = QTableWidget()
        self.window_preview_table.setColumnCount(7)
        self.window_preview_table.setHorizontalHeaderLabels([
            "Window ID",
            "Start (s)",
            "End (s)",
            "Samples",
            "Binary Label",
            "FOG Ratio",
            "Label Name",
        ])
        self.window_preview_table.verticalHeader().setVisible(False)
        self.window_preview_table.setAlternatingRowColors(True)
        self.window_preview_table.setSelectionBehavior(
            QTableWidget.SelectRows
        )
        self.window_preview_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )
        self.window_preview_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        table_layout.addWidget(self.window_preview_table)

        right_layout.addWidget(table_group, 1)

        self.windowing_status_message = QLabel(
            "Configure Window Length / Overlap, then press APPLY WINDOWING."
        )
        self.windowing_status_message.setAlignment(
            Qt.AlignCenter
        )
        self.windowing_status_message.setStyleSheet(
            "color: #555555; padding: 4px;"
        )
        right_layout.addWidget(
            self.windowing_status_message
        )

        content_layout.addWidget(right_panel, 1)
        main_layout.addLayout(content_layout, 1)

        self.update_window_label_controls()
        self.update_window_parameter_calculation()

        return page

    # =========================================================
    # PHASE 4 - OPEN WINDOWING
    # =========================================================

    def show_windowing(self):
        """
        Mở Phase 4.

        Ưu tiên df_selected từ Phase 3 vì đây là dữ liệu đã
        được preprocessing + select region.
        Nếu Phase 3 chưa Apply, vẫn cho phép dùng raw data
        để kiểm tra Windowing.
        """

        if self.df_selected is not None and self.processed_sampling_rate:
            self.windowing_input_df = self.df_selected.copy()
            self.windowing_sampling_rate = self.processed_sampling_rate
            source_text = "Phase 3 Processed Data"
        elif self.df_raw is not None and self.sampling_rate:
            self.windowing_input_df = self.df_raw.copy()
            self.windowing_sampling_rate = self.sampling_rate
            source_text = "Raw Data (Phase 3 not applied)"
        else:
            QMessageBox.warning(
                self,
                "No Input Data",
                "Hãy mở dataset trước hoặc APPLY PREPROCESSING để tạo dữ liệu đầu vào cho Windowing."
            )
            return

        self.windowing_applied = False
        self.window_metadata = None
        self.window_signal_data = []
        self.windowing_source_value.setText(source_text)
        self.windowing_result_fs_value.setText(
            f"{self.windowing_sampling_rate:g} Hz"
        )
        self.windowing_sensor_combo.setCurrentText(
            self.current_sensor
        )
        self.update_windowing_input_info()
        self.update_window_label_controls()
        self.update_window_parameter_calculation()

        self.pages.setCurrentWidget(self.windowing_page)

    # =========================================================
    # PHASE 4 - BACK TO PHASE 3
    # =========================================================

    def show_preprocessing_from_windowing(self):
        self.pages.setCurrentWidget(self.preprocessing_page)
        self.update_preprocessing_controls()

    # =========================================================
    # PHASE 4 - UPDATE INPUT INFORMATION
    # =========================================================

    def update_windowing_input_info(self):
        """Cập nhật thông tin dữ liệu đang đưa vào Windowing."""

        if self.windowing_input_df is None:
            return

        fs = self.windowing_sampling_rate
        duration = self.get_dataframe_duration(
            self.windowing_input_df,
            fs
        )

        self.windowing_sensor_value.setText(
            self.windowing_sensor_combo.currentText()
        )
        self.windowing_fs_value.setText(
            f"{fs:g} Hz" if fs else "Unknown"
        )
        self.windowing_samples_input_value.setText(
            f"{len(self.windowing_input_df):,}"
        )
        self.windowing_duration_input_value.setText(
            f"{duration:.3f} sec"
            if duration is not None
            else "Unknown"
        )

    # =========================================================
    # PHASE 4 - SENSOR INFO
    # =========================================================

    def update_windowing_sensor_info(self):
        """Đồng bộ Sensor selection với thông tin trên UI."""
        self.windowing_sensor_value.setText(
            self.windowing_sensor_combo.currentText()
            if hasattr(self, "windowing_sensor_combo")
            else "-"
        )

    # =========================================================
    # PHASE 4 - CALCULATE WINDOW PARAMETERS
    # =========================================================

    def update_window_parameter_calculation(self):
        """
        Tự động tính:
            Window samples = Window Length × Fs
            Step = Window Length × (1 - Overlap)
            Step samples = Window samples × (1 - Overlap)
            Number of windows = floor((N - W) / S) + 1
        """

        if not hasattr(self, "window_length_spinbox"):
            return

        length_seconds = float(
            self.window_length_spinbox.value()
        )
        overlap_percent = float(
            self.window_overlap_spinbox.value()
        )

        fs = self.windowing_sampling_rate
        if fs is None or fs <= 0:
            self.window_step_value.setText("-")
            self.window_samples_value.setText("-")
            self.window_step_samples_value.setText("-")
            self.window_count_calculated_value.setText("-")
            return

        # Chuyển Window Length sang số sample.
        window_samples = max(
            1,
            int(round(length_seconds * fs))
        )

        # Overlap không được bằng 100%, do đó step luôn >= 1 sample.
        overlap_ratio = overlap_percent / 100.0
        step_samples = max(
            1,
            int(round(window_samples * (1.0 - overlap_ratio)))
        )

        step_seconds = step_samples / fs

        number_windows = 0
        if self.windowing_input_df is not None:
            total_samples = len(self.windowing_input_df)
            if total_samples >= window_samples:
                number_windows = (
                    (total_samples - window_samples)
                    // step_samples
                ) + 1

        self.window_length_seconds = length_seconds
        self.window_overlap_percent = overlap_percent
        self.window_step_seconds = step_seconds
        self.window_samples = window_samples
        self.window_step_samples = step_samples

        self.window_step_value.setText(
            f"{step_seconds:.3f} s"
        )
        self.window_samples_value.setText(
            f"{window_samples:,}"
        )
        self.window_step_samples_value.setText(
            f"{step_samples:,}"
        )
        self.window_count_calculated_value.setText(
            f"{number_windows:,}"
        )

    # =========================================================
    # PHASE 4 - LABEL CONTROL
    # =========================================================

    def update_window_label_controls(self):
        """Bật/tắt threshold và mô tả rule theo Label Mode."""

        if not hasattr(self, "window_label_mode_combo"):
            return

        mode = self.window_label_mode_combo.currentText()

        threshold_enabled = mode == "FOG Threshold"
        self.window_fog_threshold_spinbox.setEnabled(
            threshold_enabled
        )

        if mode == "Majority":
            rule = (
                "Majority: label chiếm đa số sample trong window "
                "sẽ trở thành label của window."
            )
        elif mode == "FOG Threshold":
            rule = (
                "FOG Threshold: nếu FOG Ratio >= threshold thì "
                "window được gán FOG (1), ngược lại Normal (0)."
            )
        else:
            rule = (
                "Strict FOG: window chỉ được gán FOG khi toàn bộ "
                "sample trong window đều là FOG."
            )

        self.window_label_rule_value.setText(rule)

    # =========================================================
    # PHASE 4 - GET SENSOR COLUMNS
    # =========================================================

    def get_windowing_sensor_columns(self):
        """Trả về 3 cột X/Y/Z của sensor đang chọn."""

        sensor = self.windowing_sensor_combo.currentText()

        sensor_columns = {
            "Ankle": [
                "Ankle_Forward_mg",
                "Ankle_Vertical_mg",
                "Ankle_Lateral_mg",
            ],
            "Thigh": [
                "Thigh_Forward_mg",
                "Thigh_Vertical_mg",
                "Thigh_Lateral_mg",
            ],
            "Trunk": [
                "Trunk_Forward_mg",
                "Trunk_Vertical_mg",
                "Trunk_Lateral_mg",
            ],
        }

        return sensor_columns[sensor]

    # =========================================================
    # PHASE 4 - ASSIGN WINDOW LABEL
    # =========================================================

    def assign_window_label(self, binary_values):
        """
        Gán một Binary Label cho một window.

        binary_values phải là mảng 0/1.
        Hàm đồng thời trả về FOG Ratio để hiển thị/debug.
        """

        values = np.asarray(binary_values, dtype=float)
        values = values[np.isfinite(values)]

        if values.size == 0:
            return 0, 0.0

        fog_ratio = float(np.mean(values >= 0.5))
        mode = self.window_label_mode_combo.currentText()

        if mode == "Majority":
            label = 1 if fog_ratio > 0.5 else 0

        elif mode == "FOG Threshold":
            threshold = (
                self.window_fog_threshold_spinbox.value()
                / 100.0
            )
            label = 1 if fog_ratio >= threshold else 0

        else:
            # Strict FOG: mọi sample phải là FOG.
            label = 1 if fog_ratio >= 1.0 else 0

        return label, fog_ratio

    # =========================================================
    # PHASE 4 - CLEAR WINDOW RESULT
    # =========================================================

    def clear_windowing_result(self):
        """Xóa toàn bộ kết quả Windowing nhưng không xóa input data."""

        self.window_metadata = None
        self.window_signal_data = []
        self.windowing_applied = False

        if hasattr(self, "window_preview_table"):
            self.window_preview_table.clearContents()
            self.window_preview_table.setRowCount(0)

        self.windowing_status_value.setText("Not applied")
        self.windowing_result_fs_value.setText(
            f"{self.windowing_sampling_rate:g} Hz"
            if self.windowing_sampling_rate
            else "-"
        )
        self.windowing_length_result_value.setText("-")
        self.windowing_overlap_result_value.setText("-")
        self.windowing_step_result_value.setText("-")
        self.windowing_count_result_value.setText("-")
        self.windowing_status_message.setText(
            "Configure Window Length / Overlap, then press APPLY WINDOWING."
        )

    # =========================================================
    # PHASE 4 - APPLY WINDOWING
    # =========================================================

    def apply_windowing(self):
        """
        Tạo các window từ dữ liệu đầu vào.

        Mỗi window gồm:
            - Metadata: ID / Start / End / Samples / Label / FOG Ratio
            - Signal arrays của các axis đã chọn

        Kết quả được lưu riêng để Phase 4.2 Feature Extraction
        có thể dùng trực tiếp mà không cần tạo lại window.
        """

        if self.windowing_input_df is None:
            QMessageBox.warning(
                self,
                "No Input Data",
                "Chưa có dữ liệu đầu vào cho Windowing."
            )
            return

        fs = self.windowing_sampling_rate
        if fs is None or fs <= 0:
            QMessageBox.warning(
                self,
                "Sampling Rate Error",
                "Sampling Rate không hợp lệ."
            )
            return

        if not any([
            self.windowing_axis_x_checkbox.isChecked(),
            self.windowing_axis_y_checkbox.isChecked(),
            self.windowing_axis_z_checkbox.isChecked(),
        ]):
            QMessageBox.warning(
                self,
                "No Axis Selected",
                "Hãy chọn ít nhất một axis X, Y hoặc Z."
            )
            return

        try:
            dataframe = self.windowing_input_df
            time_array = self.get_dataframe_time_array(
                dataframe,
                fs
            )

            if time_array is None or len(time_array) == 0:
                raise ValueError(
                    "Không tạo được Time array cho Windowing."
                )

            window_samples = self.window_samples
            step_samples = self.window_step_samples

            if len(dataframe) < window_samples:
                raise ValueError(
                    "Window Length lớn hơn số sample hiện có."
                )

            signal_columns = self.get_windowing_sensor_columns()

            selected_flags = [
                self.windowing_axis_x_checkbox.isChecked(),
                self.windowing_axis_y_checkbox.isChecked(),
                self.windowing_axis_z_checkbox.isChecked(),
            ]
            selected_columns = [
                column
                for column, selected in zip(
                    signal_columns,
                    selected_flags
                )
                if selected
            ]

            missing = [
                column
                for column in selected_columns
                if column not in dataframe.columns
            ]
            if missing:
                raise ValueError(
                    "Không tìm thấy các cột signal:\n\n"
                    + "\n".join(missing)
                )

            # Binary Label là nguồn label chuẩn của Phase 4.
            if "Binary_Label" in dataframe.columns:
                binary_array = dataframe[
                    "Binary_Label"
                ].to_numpy()
            elif "Label" in dataframe.columns:
                binary_array = (
                    dataframe["Label"]
                    .to_numpy(dtype=float) == 2
                ).astype(int)
            else:
                raise ValueError(
                    "Dataset không có Binary_Label hoặc Label để gán label cho window."
                )

            self.window_metadata = []
            self.window_signal_data = []

            # -----------------------------------------------------
            # GENERATE EACH WINDOW
            # -----------------------------------------------------
            start_index = 0
            window_id = 1

            while start_index + window_samples <= len(dataframe):
                end_index = (
                    start_index + window_samples
                )

                window_time = time_array[
                    start_index:end_index
                ]

                label, fog_ratio = self.assign_window_label(
                    binary_array[start_index:end_index]
                )

                # Lưu signal theo đơn vị g để đồng nhất với Time Domain.
                signal_dict = {}
                for column in selected_columns:
                    values = dataframe[column].to_numpy(
                        dtype=float
                    )[start_index:end_index] / 1000.0
                    signal_dict[column] = values.copy()

                self.window_signal_data.append({
                    "Window_ID": window_id,
                    "Time": window_time.copy(),
                    "Signals": signal_dict,
                })

                self.window_metadata.append({
                    "Window_ID": window_id,
                    "Start_Index": start_index,
                    "End_Index": end_index - 1,
                    "Start_Time": float(window_time[0]),
                    "End_Time": float(window_time[-1]),
                    "Samples": int(len(window_time)),
                    "Binary_Label": int(label),
                    "FOG_Ratio": float(fog_ratio),
                })

                start_index += step_samples
                window_id += 1

            if not self.window_metadata:
                raise ValueError(
                    "Không tạo được window từ dữ liệu hiện tại."
                )

            self.windowing_applied = True

            self.windowing_status_value.setText("Applied")
            self.windowing_result_fs_value.setText(
                f"{fs:g} Hz"
            )
            self.windowing_length_result_value.setText(
                f"{self.window_length_seconds:.3f} s"
            )
            self.windowing_overlap_result_value.setText(
                f"{self.window_overlap_percent:.1f} %"
            )
            self.windowing_step_result_value.setText(
                f"{self.window_step_seconds:.3f} s"
            )
            self.windowing_count_result_value.setText(
                f"{len(self.window_metadata):,}"
            )

            self.show_window_preview()

            self.windowing_status_message.setText(
                f"Generated {len(self.window_metadata):,} windows | "
                f"{self.window_samples:,} samples/window | "
                f"step = {self.window_step_samples:,} samples"
            )

        except Exception as error:
            self.clear_windowing_result()
            QMessageBox.critical(
                self,
                "Windowing Error",
                str(error)
            )

    # =========================================================
    # PHASE 4 - WINDOW PREVIEW TABLE
    # =========================================================

    def show_window_preview(self):
        """Hiển thị metadata của tất cả window đã tạo."""

        self.window_preview_table.clearContents()
        self.window_preview_table.setRowCount(
            len(self.window_metadata)
        )

        for row, metadata in enumerate(
            self.window_metadata
        ):
            label = metadata["Binary_Label"]
            label_name = "FOG" if label == 1 else "Normal"

            values = [
                metadata["Window_ID"],
                f'{metadata["Start_Time"]:.3f}',
                f'{metadata["End_Time"]:.3f}',
                f'{metadata["Samples"]:,}',
                label,
                f'{metadata["FOG_Ratio"] * 100:.1f} %',
                label_name,
            ]

            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(
                    Qt.AlignCenter | Qt.AlignVCenter
                )
                self.window_preview_table.setItem(
                    row,
                    column,
                    item
                )

        self.window_preview_table.resizeRowsToContents()

    # =========================================================
    # PHASE 4 - RESET
    # =========================================================

    def reset_windowing(self):
        """Reset thông số + kết quả Windowing về mặc định."""

        self.window_length_spinbox.setValue(2.0)
        self.window_overlap_spinbox.setValue(50.0)
        self.window_label_mode_combo.setCurrentText("Majority")
        self.window_fog_threshold_spinbox.setValue(50.0)

        self.windowing_axis_x_checkbox.setChecked(True)
        self.windowing_axis_y_checkbox.setChecked(True)
        self.windowing_axis_z_checkbox.setChecked(True)

        self.clear_windowing_result()
        self.update_window_parameter_calculation()
        self.update_window_label_controls()

    # =========================================================
    # PHASE 5 - TIME & FREQUENCY ANALYSIS PAGE
    # =========================================================

    def create_analysis_page(self):
        """PHASE 5: phân tích một window bằng Time Analysis + FFT + PSD."""

        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # ---------------------------------------------------------
        # HEADER
        # ---------------------------------------------------------
        header_layout = QHBoxLayout()

        title = QLabel("TIME & FREQUENCY ANALYSIS")
        title.setStyleSheet(
            "font-size: 22px; font-weight: bold;"
        )

        self.analysis_back_button = QPushButton("BACK TO WINDOWING")
        self.analysis_back_button.clicked.connect(
            self.show_windowing_from_analysis
        )

        self.analysis_run_button = QPushButton("ANALYZE WINDOW")
        self.analysis_run_button.clicked.connect(
            self.analyze_selected_window
        )

        self.analysis_reset_button = QPushButton("RESET ANALYSIS")
        self.analysis_reset_button.clicked.connect(
            self.reset_analysis
        )

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.analysis_reset_button)
        header_layout.addWidget(self.analysis_run_button)
        header_layout.addWidget(self.analysis_back_button)
        main_layout.addLayout(header_layout)

        pipeline_label = QLabel(
            "Pipeline:  Select Window  →  Time Analysis  +  FFT  +  PSD"
        )
        pipeline_label.setAlignment(Qt.AlignCenter)
        pipeline_label.setStyleSheet(
            "font-weight: bold; padding: 6px;"
        )
        main_layout.addWidget(pipeline_label)

        # ---------------------------------------------------------
        # SELECTION BAR
        # ---------------------------------------------------------
        selection_group = QGroupBox("WINDOW SELECTION")
        selection_layout = QGridLayout(selection_group)

        selection_layout.addWidget(QLabel("Window:"), 0, 0)
        self.analysis_window_combo = QComboBox()
        self.analysis_window_combo.currentIndexChanged.connect(
            self.on_analysis_window_changed
        )
        selection_layout.addWidget(self.analysis_window_combo, 0, 1)

        selection_layout.addWidget(QLabel("Sensor:"), 0, 2)
        self.analysis_sensor_combo = QComboBox()
        self.analysis_sensor_combo.addItems(["Ankle", "Thigh", "Trunk"])
        selection_layout.addWidget(self.analysis_sensor_combo, 0, 3)

        selection_layout.addWidget(QLabel("Axis:"), 0, 4)
        self.analysis_axis_combo = QComboBox()
        self.analysis_axis_combo.addItems(["X", "Y", "Z"])
        selection_layout.addWidget(self.analysis_axis_combo, 0, 5)

        selection_layout.addWidget(QLabel("Label:"), 1, 0)
        self.analysis_label_value = QLabel("-")
        selection_layout.addWidget(self.analysis_label_value, 1, 1)

        selection_layout.addWidget(QLabel("Time:"), 1, 2)
        self.analysis_time_value = QLabel("-")
        selection_layout.addWidget(self.analysis_time_value, 1, 3)

        selection_layout.addWidget(QLabel("Samples:"), 1, 4)
        self.analysis_samples_value = QLabel("-")
        selection_layout.addWidget(self.analysis_samples_value, 1, 5)

        main_layout.addWidget(selection_group)

        # ---------------------------------------------------------
        # MAIN CONTENT
        # ---------------------------------------------------------
        content_layout = QHBoxLayout()

        # LEFT: time-domain plot + time features
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        time_group = QGroupBox("TIME ANALYSIS")
        time_layout = QVBoxLayout(time_group)

        self.analysis_time_plot = pg.PlotWidget()
        self.analysis_time_plot.setBackground("w")
        self.analysis_time_plot.showGrid(x=True, y=True, alpha=0.25)
        self.analysis_time_plot.setLabel("bottom", "Time", units="s")
        self.analysis_time_plot.setLabel("left", "Acceleration", units="g")
        self.analysis_time_plot.setMinimumHeight(300)
        time_layout.addWidget(self.analysis_time_plot)

        self.analysis_time_stats = QLabel(
            "Mean: -    RMS: -    Std: -    Variance: -    Min: -    Max: -"
        )
        self.analysis_time_stats.setWordWrap(True)
        self.analysis_time_stats.setStyleSheet("padding: 6px;")
        time_layout.addWidget(self.analysis_time_stats)

        left_layout.addWidget(time_group, 1)

        # RIGHT: FFT + PSD
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        fft_group = QGroupBox("FFT SPECTRUM")
        fft_layout = QVBoxLayout(fft_group)
        self.analysis_fft_plot = pg.PlotWidget()
        self.analysis_fft_plot.setBackground("w")
        self.analysis_fft_plot.showGrid(x=True, y=True, alpha=0.25)
        self.analysis_fft_plot.setLabel("bottom", "Frequency", units="Hz")
        self.analysis_fft_plot.setLabel("left", "Magnitude")
        self.analysis_fft_plot.setMinimumHeight(240)
        fft_layout.addWidget(self.analysis_fft_plot)
        right_layout.addWidget(fft_group, 1)

        psd_group = QGroupBox("POWER SPECTRAL DENSITY (PSD)")
        psd_layout = QVBoxLayout(psd_group)
        self.analysis_psd_plot = pg.PlotWidget()
        self.analysis_psd_plot.setBackground("w")
        self.analysis_psd_plot.showGrid(x=True, y=True, alpha=0.25)
        self.analysis_psd_plot.setLabel("bottom", "Frequency", units="Hz")
        self.analysis_psd_plot.setLabel("left", "PSD", units="g²/Hz")
        self.analysis_psd_plot.setMinimumHeight(240)
        psd_layout.addWidget(self.analysis_psd_plot)
        right_layout.addWidget(psd_group, 1)

        self.analysis_frequency_stats = QLabel(
            "Dominant Frequency: -    Total Power: -"
        )
        self.analysis_frequency_stats.setWordWrap(True)
        self.analysis_frequency_stats.setStyleSheet("padding: 6px;")
        right_layout.addWidget(self.analysis_frequency_stats)

        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(right_panel, 1)
        main_layout.addLayout(content_layout, 1)

        self.analysis_status = QLabel(
            "Apply Windowing, then select a window and press ANALYZE WINDOW."
        )
        self.analysis_status.setAlignment(Qt.AlignCenter)
        self.analysis_status.setStyleSheet(
            "color: #555555; padding: 4px;"
        )
        main_layout.addWidget(self.analysis_status)

        return page

    # =========================================================
    # PHASE 5 - OPEN / NAVIGATION
    # =========================================================

    def show_analysis(self):
        """Mở Phase 5 sau khi Phase 4 đã tạo window."""

        if not self.windowing_applied or not self.window_metadata:
            QMessageBox.warning(
                self,
                "Analysis Not Available",
                "Hãy APPLY WINDOWING trước khi thực hiện Time & Frequency Analysis."
            )
            return

        self.update_analysis_window_list()
        self.pages.setCurrentWidget(self.analysis_page)

    def show_windowing_from_analysis(self):
        self.pages.setCurrentWidget(self.windowing_page)

    def update_analysis_window_list(self):
        """Nạp danh sách window từ kết quả Phase 4."""

        if not hasattr(self, "analysis_window_combo"):
            return

        self.analysis_window_combo.blockSignals(True)
        self.analysis_window_combo.clear()

        if self.window_metadata:
            for metadata in self.window_metadata:
                label_name = (
                    "FOG" if int(metadata["Binary_Label"]) == 1 else "Non-FOG"
                )
                text = (
                    f'Window {metadata["Window_ID"]} | '
                    f'{metadata["Start_Time"]:.3f}–{metadata["End_Time"]:.3f} s | '
                    f'{label_name} | FOG {metadata["FOG_Ratio"] * 100:.1f}%'
                )
                self.analysis_window_combo.addItem(
                    text,
                    metadata["Window_ID"]
                )

        self.analysis_window_combo.blockSignals(False)

        if self.analysis_window_combo.count() > 0:
            self.analysis_window_combo.setCurrentIndex(0)
            self.on_analysis_window_changed(0)

    def on_analysis_window_changed(self, index):
        if index < 0 or not self.window_metadata:
            return

        window_id = self.analysis_window_combo.itemData(index)
        self.analysis_window_id = int(window_id)

        metadata = next(
            (
                item for item in self.window_metadata
                if int(item["Window_ID"]) == self.analysis_window_id
            ),
            None
        )

        if metadata is None:
            return

        label_name = (
            "FOG" if int(metadata["Binary_Label"]) == 1 else "Non-FOG"
        )

        self.analysis_label_value.setText(label_name)
        self.analysis_time_value.setText(
            f'{metadata["Start_Time"]:.3f} → {metadata["End_Time"]:.3f} s'
        )
        self.analysis_samples_value.setText(
            f'{metadata["Samples"]:,}'
        )

        # Chọn lại signal chưa chạy analysis.
        self.analysis_status.setText(
            f'Window {self.analysis_window_id} selected | {label_name} | Press ANALYZE WINDOW.'
        )

    # =========================================================
    # PHASE 5 - GET SIGNAL
    # =========================================================

    def get_analysis_signal(self):
        """Lấy một axis của sensor từ window đang chọn."""

        if not self.window_signal_data:
            raise ValueError("Chưa có dữ liệu window. Hãy APPLY WINDOWING trước.")

        if self.analysis_window_id is None:
            raise ValueError("Chưa chọn window để phân tích.")

        selected_window = next(
            (
                item for item in self.window_signal_data
                if int(item["Window_ID"]) == self.analysis_window_id
            ),
            None
        )

        if selected_window is None:
            raise ValueError(
                f"Không tìm thấy dữ liệu của Window {self.analysis_window_id}."
            )

        sensor = self.analysis_sensor_combo.currentText()
        axis = self.analysis_axis_combo.currentText()

        sensor_columns = {
            "Ankle": [
                "Ankle_Forward_mg",
                "Ankle_Vertical_mg",
                "Ankle_Lateral_mg",
            ],
            "Thigh": [
                "Thigh_Forward_mg",
                "Thigh_Vertical_mg",
                "Thigh_Lateral_mg",
            ],
            "Trunk": [
                "Trunk_Forward_mg",
                "Trunk_Vertical_mg",
                "Trunk_Lateral_mg",
            ],
        }

        axis_index = {"X": 0, "Y": 1, "Z": 2}[axis]
        column = sensor_columns[sensor][axis_index]

        if column not in selected_window["Signals"]:
            raise ValueError(
                f"Windowing hiện tại chưa lưu {sensor} Axis {axis}. "
                "Hãy chọn axis tương ứng trong Phase 4."
            )

        time_array = np.asarray(
            selected_window["Time"],
            dtype=float
        )
        signal_array = np.asarray(
            selected_window["Signals"][column],
            dtype=float
        )

        if len(signal_array) < 2:
            raise ValueError("Window phải có ít nhất 2 sample để phân tích.")

        return time_array, signal_array

    # =========================================================
    # PHASE 5 - ANALYSIS
    # =========================================================

    def analyze_selected_window(self):
        """Tính Time Analysis, FFT và PSD cho đúng một window."""

        try:
            time_array, x = self.get_analysis_signal()

            fs = self.windowing_sampling_rate
            if fs is None or fs <= 0:
                raise ValueError("Sampling rate không hợp lệ.")

            x = np.asarray(x, dtype=float)
            x = x - np.mean(x)

            # -----------------------------------------------------
            # TIME FEATURES
            # -----------------------------------------------------
            #mean
            mean_value = float(np.mean(x))
            #RMS
            rms_value = float(np.sqrt(np.mean(x ** 2)))
            #STD
            std_value = float(np.std(x))
            #variance
            variance_value = float(np.var(x))
            #range
            min_value = float(np.min(x))
            max_value = float(np.max(x))
            #SMA
            sma_value = float(np.sum(np.abs(x)) / fs)

            # -----------------------------------------------------
            # FFT
            # -----------------------------------------------------
            n = len(x)
            fft_freq = np.fft.rfftfreq(n, d=1.0 / fs)
            fft_complex = np.fft.rfft(x)
            fft_mag = (2.0 / n) * np.abs(fft_complex)

            if len(fft_mag) > 1:
                fft_mag[0] *= 0.5

            # Ignore DC when selecting dominant frequency.
            if len(fft_mag) > 1:
                dominant_index = int(np.argmax(fft_mag[1:]) + 1)
            else:
                dominant_index = 0

            dominant_frequency = float(
                fft_freq[dominant_index]
            )

            # -----------------------------------------------------
            # PSD - Welch
            # -----------------------------------------------------
            nperseg = min(256, n)
            psd_freq, psd_power = signal.welch(
                x,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=nperseg // 2 if nperseg >= 2 else 0,
                detrend=False,
                scaling="density"
            )
            # -----------------------------------------------------
            # FREQUENCY FEATURES (Từ PSD)
            # -----------------------------------------------------
            #Total_Power
            total_power = float(
                np.trapezoid(psd_power, psd_freq)
                if hasattr(np, "trapezoid")
                else np.trapz(psd_power, psd_freq)
            )

            #Band Power Calculation
            loco_mask = (psd_freq >= 0.5) & (psd_freq <= 3.0)
            freeze_mask = (psd_freq > 3.0) & (psd_freq <= 8.0)
            def integrate_area(y, x_arr):
                if len(y) < 2: return 0.0
                return float(np.trapezoid(y, x_arr) if hasattr(np, "trapezoid") else np.trapz(y, x_arr))
            loco_power = integrate_area(psd_power[loco_mask], psd_freq[loco_mask])
            freeze_power = integrate_area(psd_power[freeze_mask], psd_freq[freeze_mask])

            #Freezing Index (FI)
            fi_value = (freeze_power / loco_power) if loco_power > 0 else 0.0

            #Low Frequency Ratio
            low_freq_ratio = (loco_power / total_power) if total_power > 0 else 0.0

            #Power Spectral Entropy (PSE)
            psd_sum = np.sum(psd_power)
            if psd_sum > 0:
                psd_norm = psd_power / psd_sum
                psd_norm = psd_norm[psd_norm > 0] # Chặn lỗi log2(0)
                pse_value = float(-np.sum(psd_norm * np.log2(psd_norm)))
            else:
                pse_value = 0.0

            # -----------------------------------------------------
            # STORE RESULTS
            # -----------------------------------------------------
            self.analysis_signal = x
            self.analysis_time = time_array
            self.analysis_fs = fs
            self.analysis_fft_freq = fft_freq
            self.analysis_fft_mag = fft_mag
            self.analysis_psd_freq = psd_freq
            self.analysis_psd_power = psd_power

            # -----------------------------------------------------
            # TIME PLOT
            # -----------------------------------------------------
            self.analysis_time_plot.clear()
            self.analysis_time_plot.plot(
                time_array,
                x,
                pen=pg.mkPen(
                    color="#1f77b4",
                    width=1.4
                ),
                antialias=False
            )
            self.analysis_time_plot.setTitle(
                f"Time Domain | Window {self.analysis_window_id} | "
                f"{self.analysis_sensor_combo.currentText()} "
                f"{self.analysis_axis_combo.currentText()}"
            )


            self.analysis_time_stats.setText(
                f"Mean: {mean_value:.6f} g    |    "
                f"RMS: {rms_value:.6f} g    |    "
                f"Std: {std_value:.6f} g    |    "
                f"Variance: {variance_value:.6f} g²    |    "
                f"Min: {min_value:.6f} g    |    Max: {max_value:.6f} g    |    "
                f"SMA: {sma_value:.5f} g·s"
            )
            

            # -----------------------------------------------------
            # FFT PLOT
            # -----------------------------------------------------
            self.analysis_fft_plot.clear()
            self.analysis_fft_plot.plot(
                fft_freq,
                fft_mag,
                pen=pg.mkPen(
                    color="#ff7f0e",
                    width=1.4
                ),
                antialias=False
            )
            self.analysis_fft_plot.setTitle(
                f"FFT Spectrum | Window {self.analysis_window_id}"
            )

            # -----------------------------------------------------
            # PSD PLOT
            # -----------------------------------------------------
            self.analysis_psd_plot.clear()
            self.analysis_psd_plot.plot(
                psd_freq,
                psd_power,
                pen=pg.mkPen(
                    color="#2ca02c",
                    width=1.4
                ),
                antialias=False
            )
            self.analysis_psd_plot.setTitle(
                f"PSD (Welch) | Window {self.analysis_window_id}"
            )

            self.analysis_frequency_stats.setText(
                f"Dominant Freq: {dominant_frequency:.2f} Hz  |  Total Power: {total_power:.6f} g²  |  PSE: {pse_value:.4f}\n"
                f"Loco Power (0.5-3Hz): {loco_power:.6f}  |  Freeze Power (3-8Hz): {freeze_power:.6f}\n"
                f"FI Ratio: {fi_value:.4f}  |  Low Freq Ratio: {low_freq_ratio:.4f}"
            )

            label_name = self.analysis_label_value.text()
            self.analysis_status.setText(
                f"Analyzed Window {self.analysis_window_id} | "
                f"{label_name} | "
                f"{self.analysis_sensor_combo.currentText()} "
                f"{self.analysis_axis_combo.currentText()} | "
                f"Fs = {fs:g} Hz | FFT + PSD ready"
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "Analysis Error",
                str(error)
            )

    # =========================================================
    # PHASE 5 - RESET
    # =========================================================

    def reset_analysis(self):
        self.analysis_window_id = None
        self.analysis_signal = None
        self.analysis_time = None
        self.analysis_fs = None
        self.analysis_fft_freq = None
        self.analysis_fft_mag = None
        self.analysis_psd_freq = None
        self.analysis_psd_power = None

        if hasattr(self, "analysis_time_plot"):
            self.analysis_time_plot.clear()
            self.analysis_time_plot.setTitle("Time Domain")

        if hasattr(self, "analysis_fft_plot"):
            self.analysis_fft_plot.clear()
            self.analysis_fft_plot.setTitle("FFT Spectrum")

        if hasattr(self, "analysis_psd_plot"):
            self.analysis_psd_plot.clear()
            self.analysis_psd_plot.setTitle("PSD (Welch)")

        if hasattr(self, "analysis_label_value"):
            self.analysis_label_value.setText("-")
            self.analysis_time_value.setText("-")
            self.analysis_samples_value.setText("-")
            self.analysis_time_stats.setText(
                "Mean: -    RMS: -    Std: -    Variance: -    Min: -    Max: -"
            )
            self.analysis_frequency_stats.setText(
                "Dominant Frequency: -    Total Power: -"
            )
            self.analysis_status.setText(
                "Analysis reset. Select a window and press ANALYZE WINDOW."
            )

    # =========================================================
    # BACK TO DATASET
    # =========================================================

    def show_dataset_page(self):

        if self.df is not None:
            self.display_df = self.df.copy()
            self.display_sampling_rate = self.sampling_rate
            self.display_processed_label = False

        self.pages.setCurrentWidget(
            self.dataset_page
        )

    # =========================================================
    # TIME RANGE ENABLE
    # =========================================================

    def toggle_time_range(self):

        entire = (
            self.entire_dataset_checkbox.isChecked()
        )

        self.start_time_spinbox.setEnabled(
            not entire
        )

        self.end_time_spinbox.setEnabled(
            not entire
        )

    # =========================================================
    # UPDATE TIME CONTROLS
    # =========================================================

    def update_time_controls(self):

        if (
            self.display_df is not None
            and self.display_sampling_rate is not None
        ):
            display_duration = self.get_dataframe_duration(
                self.display_df,
                self.display_sampling_rate
            )
            if display_duration is not None:
                self.duration = display_duration

        if self.duration is None:

            return

        self.start_time_spinbox.setRange(
            0,
            self.duration
        )

        self.end_time_spinbox.setRange(
            0,
            self.duration
        )

        self.start_time_spinbox.setValue(
            0
        )

        self.end_time_spinbox.setValue(
            self.duration
        )

        self.entire_dataset_checkbox.setChecked(
            True
        )

        self.toggle_time_range()

    # =========================================================
    # OPEN DATASET
    # =========================================================

    def open_dataset(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Dataset",
            "",
            "Dataset Files (*.txt *.csv);;All Files (*)"
        )

        if not file_path:

            return

        try:

            # =================================================
            # DAPHNET TXT
            # =================================================

            if file_path.lower().endswith(
                ".txt"
            ):

                column_names = [

                    "Time_ms",

                    "Ankle_Forward_mg",
                    "Ankle_Vertical_mg",
                    "Ankle_Lateral_mg",

                    "Thigh_Forward_mg",
                    "Thigh_Vertical_mg",
                    "Thigh_Lateral_mg",

                    "Trunk_Forward_mg",
                    "Trunk_Vertical_mg",
                    "Trunk_Lateral_mg",

                    "Label"
                ]

                df = pd.read_csv(
                    file_path,
                    sep=r"\s+",
                    header=None,
                    names=column_names
                )

                self.sampling_rate = 64

            # =================================================
            # CSV
            # =================================================

            else:

                df = pd.read_csv(
                    file_path
                )

                self.sampling_rate = None

            # =================================================
            # STORE
            # =================================================

            self.df = df
            self.df_raw = df.copy()
            self.df_processed = None
            self.df_selected = None
            self.display_df = df.copy()
            self.display_sampling_rate = self.sampling_rate
            self.display_processed_label = False
            self.preprocessing_applied = False
            self.processed_sampling_rate = None
            self.processed_start_time = None
            self.processed_end_time = None

            # Dataset mới => xóa toàn bộ Windowing result cũ.
            self.windowing_input_df = None
            self.windowing_sampling_rate = None
            self.window_metadata = None
            self.window_signal_data = []
            self.windowing_applied = False

            self.file_path = file_path

            # =================================================
            # FILE
            # =================================================

            file_name = os.path.basename(
                file_path
            )

            self.file_value.setText(
                file_name
            )

            self.file_value.setToolTip(
                file_path
            )

            self.path_label.setText(
                file_path
            )

            self.path_label.setToolTip(
                file_path
            )

            # =================================================
            # COLUMNS
            # =================================================

            self.columns_value.setText(
                str(len(df.columns))
            )

            # =================================================
            # SAMPLES
            # =================================================

            self.samples_value.setText(
                f"{len(df):,}"
            )

            # =================================================
            # SAMPLING RATE
            # =================================================

            if self.sampling_rate is not None:

                self.fs_value.setText(
                    f"{self.sampling_rate} Hz"
                )

            else:

                self.fs_value.setText(
                    "Unknown"
                )

            # =================================================
            # DURATION
            # =================================================

            self.calculate_duration()

            # =================================================
            # PREVIEW
            # =================================================

            self.show_preview()

            # =================================================
            # LABEL
            # =================================================

            self.detect_labels()

            # =================================================
            # TIME
            # =================================================

            self.update_time_controls()

        except Exception as error:

            QMessageBox.critical(
                self,
                "Error",
                "Không thể đọc dataset.\n\n"
                f"{error}"
            )

    # =========================================================
    # CALCULATE DURATION
    # =========================================================

    def calculate_duration(self):

        if self.df is None:

            return

        if "Time_ms" in self.df.columns:

            try:

                first_time = float(
                    self.df[
                        "Time_ms"
                    ].iloc[0]
                )

                last_time = float(
                    self.df[
                        "Time_ms"
                    ].iloc[-1]
                )

                self.duration = (
                    last_time - first_time
                ) / 1000.0

                self.duration_value.setText(
                    f"{self.duration:.3f} sec"
                )

                return

            except Exception:

                pass

        if self.sampling_rate:

            self.duration = (
                len(self.df)
                / self.sampling_rate
            )

            self.duration_value.setText(
                f"{self.duration:.3f} sec"
            )

        else:

            self.duration = None

            self.duration_value.setText(
                "Unknown"
            )

    # =========================================================
    # PREVIEW
    # =========================================================

    def show_preview(self):

        if self.df is None:

            return

        preview_df = self.df.head(
            12
        )

        self.preview_table.clear()

        self.preview_table.setRowCount(
            len(preview_df)
        )

        self.preview_table.setColumnCount(
            len(preview_df.columns)
        )

        headers = []

        for column in preview_df.columns:

            header = str(
                column
            )

            header = header.replace(
                "_",
                "\n"
            )

            headers.append(
                header
            )

        self.preview_table.setHorizontalHeaderLabels(
            headers
        )

        for row in range(
            len(preview_df)
        ):

            for column in range(
                len(preview_df.columns)
            ):

                value = preview_df.iloc[
                    row,
                    column
                ]

                item = QTableWidgetItem(
                    str(value)
                )

                item.setTextAlignment(
                    Qt.AlignCenter |
                    Qt.AlignVCenter
                )

                self.preview_table.setItem(
                    row,
                    column,
                    item
                )

        header = (
            self.preview_table.horizontalHeader()
        )

        for column in range(
            len(preview_df.columns)
        ):

            header.setSectionResizeMode(
                column,
                QHeaderView.Stretch
            )

        self.preview_table.horizontalHeader().setMinimumHeight(
            65
        )

        self.preview_table.verticalHeader().setDefaultSectionSize(
            27
        )

    # =========================================================
    # DETECT LABELS
    # =========================================================

    def detect_labels(self):

        if self.df is None:

            return

        if "Label" in self.df.columns:

            label_column = "Label"

        else:

            possible_columns = [

                column

                for column in self.df.columns

                if str(column).lower()
                in [
                    "label",
                    "class",
                    "target",
                    "state"
                ]
            ]

            if not possible_columns:

                self.label_table.setRowCount(
                    0
                )

                return

            label_column = (
                possible_columns[0]
            )

        try:

            unique_labels = sorted(
                self.df[
                    label_column
                ]
                .dropna()
                .unique()
            )

            self.label_table.clear()

            self.label_table.setRowCount(
                len(unique_labels)
            )

            self.label_table.setColumnCount(
                3
            )

            self.label_table.setHorizontalHeaderLabels(
                [
                    "Original Label",
                    "Label Name",
                    "Binary"
                ]
            )

            for row, label in enumerate(
                unique_labels
            ):

                original_item = (
                    QTableWidgetItem(
                        str(label)
                    )
                )

                original_item.setTextAlignment(
                    Qt.AlignCenter |
                    Qt.AlignVCenter
                )

                self.label_table.setItem(
                    row,
                    0,
                    original_item
                )

                name_item = (
                    QTableWidgetItem(
                        self.get_default_label_name(
                            label
                        )
                    )
                )

                name_item.setTextAlignment(
                    Qt.AlignCenter |
                    Qt.AlignVCenter
                )

                self.label_table.setItem(
                    row,
                    1,
                    name_item
                )

                if label == 2:

                    binary = "1"

                else:

                    binary = "0"

                binary_item = (
                    QTableWidgetItem(
                        binary
                    )
                )

                binary_item.setTextAlignment(
                    Qt.AlignCenter |
                    Qt.AlignVCenter
                )

                self.label_table.setItem(
                    row,
                    2,
                    binary_item
                )

            self.label_table.resizeRowsToContents()

        except Exception as error:

            QMessageBox.warning(
                self,
                "Label Error",
                str(error)
            )

    # =========================================================
    # LABEL NAME
    # =========================================================

    def get_default_label_name(
        self,
        label
    ):

        if label == 0:

            return "Before Experiment"

        if label == 1:

            return "Normal"

        if label == 2:

            return "FOG"

        return "Unknown"

    # =========================================================
    # APPLY LABEL MAPPING
    # =========================================================

    def apply_label_mapping(self):

        if self.df is None:

            return

        if "Label" in self.df.columns:

            label_column = "Label"

        else:

            possible_columns = [

                column

                for column in self.df.columns

                if str(column).lower()
                in [
                    "label",
                    "class",
                    "target",
                    "state"
                ]
            ]

            if not possible_columns:

                QMessageBox.warning(
                    self,
                    "Label Mapping",
                    "Không tìm thấy cột label."
                )

                return

            label_column = (
                possible_columns[0]
            )

        # Original
        self.df["Original_Label"] = (
            self.df[
                label_column
            ]
        )

        # Binary
        self.df["Binary_Label"] = (
            self.df[
                label_column
            ]
            .apply(
                lambda x:
                    1 if x == 2 else 0
            )
        )

        # Nếu đang ở Time Domain,
        # cập nhật lại plot để dùng Binary_Label
        if (
            self.pages.currentWidget()
            == self.time_domain_page
        ):

            self.load_time_range()

    # =========================================================
    # LOAD TIME RANGE
    # =========================================================

    def load_time_range(self, dataframe=None, sampling_rate=None):

        if dataframe is None:
            dataframe = (
                self.display_df
                if self.display_df is not None
                else self.df
            )

        if sampling_rate is None:
            sampling_rate = (
                self.display_sampling_rate
                if self.display_sampling_rate is not None
                else self.sampling_rate
            )

        if dataframe is None:
            return

        # =====================================================
        # TIME ARRAY
        # =====================================================

        if "Time_ms" in dataframe.columns:

            time_array = (
                dataframe[
                    "Time_ms"
                ]
                .to_numpy(
                    dtype=float
                )
                / 1000.0
            )

        else:

            if not sampling_rate:

                QMessageBox.warning(
                    self,
                    "Time Error",
                    "Dataset không có Time_ms "
                    "và chưa có Sampling Rate."
                )

                return

            time_array = (
                np.arange(
                    len(dataframe),
                    dtype=float
                )
                / sampling_rate
            )

        # =====================================================
        # RANGE
        # =====================================================

        if self.entire_dataset_checkbox.isChecked():

            start_time = float(
                time_array[0]
            )

            end_time = float(
                time_array[-1]
            )

        else:

            start_time = (
                self.start_time_spinbox.value()
            )

            end_time = (
                self.end_time_spinbox.value()
            )

            if end_time <= start_time:

                QMessageBox.warning(
                    self,
                    "Invalid Time Range",
                    "End Time phải lớn hơn Start Time."
                )

                return

        # =====================================================
        # MASK
        # =====================================================

        mask = (
            (time_array >= start_time)
            &
            (time_array <= end_time)
        )

        selected_df = (
            dataframe.loc[
                mask
            ].copy()
        )

        selected_time = (
            time_array[mask]
        )

        if selected_df.empty:

            QMessageBox.warning(
                self,
                "No Data",
                "Không có dữ liệu trong khoảng thời gian này."
            )

            return

        # =====================================================
        # SENSOR COLUMNS
        # =====================================================

        sensor_columns = {

            "Ankle": [
                "Ankle_Forward_mg",
                "Ankle_Vertical_mg",
                "Ankle_Lateral_mg"
            ],

            "Thigh": [
                "Thigh_Forward_mg",
                "Thigh_Vertical_mg",
                "Thigh_Lateral_mg"
            ],

            "Trunk": [
                "Trunk_Forward_mg",
                "Trunk_Vertical_mg",
                "Trunk_Lateral_mg"
            ]
        }

        columns = sensor_columns[
            self.current_sensor
        ]

        # =====================================================
        # CHECK
        # =====================================================

        missing_columns = [

            column

            for column in columns

            if column not in selected_df.columns
        ]

        if missing_columns:

            QMessageBox.warning(
                self,
                "Signal Error",
                "Không tìm thấy các cột:\n\n"
                + "\n".join(
                    missing_columns
                )
            )

            return

        # =====================================================
        # CLEAR
        # =====================================================

        self.plot_widget.clear()

        self.legend = (
            self.plot_widget.addLegend(
                offset=(10, 10)
            )
        )

        # =====================================================
        # STORE CURRENT DATA
        # FOR DYNAMIC Y SCALE
        # =====================================================

        self.current_time_array = (
            selected_time.copy()
        )

        # Data ở đơn vị g
        signal_matrix = np.column_stack(

            [

                selected_df[
                    column
                ]
                .to_numpy(
                    dtype=float
                )
                / 1000.0

                for column in columns
            ]
        )

        self.current_signal_data = (
            signal_matrix
        )

        self.current_signal_columns = (
            columns
        )

        # =====================================================
        # COLORS
        # =====================================================

        blue = "#1f77b4"
        orange = "#ff7f0e"
        green = "#2ca02c"
        red = "#d62728"

        colors = [
            blue,
            orange,
            green
        ]

        axis_names = [
            "X / Forward",
            "Y / Vertical",
            "Z / Lateral"
        ]

        # =====================================================
        # PLOT X / Y / Z
        # =====================================================

        for index, column in enumerate(
            columns
        ):

            signal = (
                signal_matrix[
                    :,
                    index
                ]
            )

            curve = (
                self.plot_widget.plot(
                    selected_time,
                    signal,
                    name=axis_names[index],
                    pen=pg.mkPen(
                        color=colors[index],
                        width=1.2
                    ),
                    antialias=False
                )
            )

            # Dataset rất dài:
            # giảm số điểm hiển thị nhưng
            # vẫn bảo toàn peak.
            curve.setDownsampling(
                auto=True,
                method="peak"
            )

            curve.setClipToView(
                True
            )

        # =====================================================
        # BINARY LABEL
        # =====================================================

        if "Binary_Label" in selected_df.columns:

            binary_label = (
                selected_df[
                    "Binary_Label"
                ]
                .to_numpy(
                    dtype=float
                )
            )

        elif "Label" in selected_df.columns:

            binary_label = (
                selected_df[
                    "Label"
                ]
                .to_numpy(
                    dtype=float
                )
                == 2
            ).astype(float)

        else:

            binary_label = None

        # =====================================================
        # FOG
        # =====================================================

        if binary_label is not None:

            fog_curve = (
                pg.PlotDataItem(
                    selected_time,
                    binary_label,
                    pen=pg.mkPen(
                        color=red,
                        width=2
                    ),
                    stepMode="left"
                )
            )

            self.fog_view.addItem(
                fog_curve
            )

            self.legend.addItem(
                fog_curve,
                "FOG State"
            )

        # =====================================================
        # AXES
        # =====================================================

        self.plot_widget.setLabel(
            "bottom",
            "Time",
            units="s"
        )

        self.plot_widget.setLabel(
            "left",
            "Acceleration",
            units="g"
        )

        self.fog_axis.setLabel(
            "FOG State",
            color=red
        )

        self.fog_view.setYRange(
            -0.1,
            1.1,
            padding=0
        )

        # =====================================================
        # TITLE
        # =====================================================

        signal_title = (
            f"{self.current_sensor} Signal"
            + (
                " (Processed)"
                if self.display_processed_label
                else " (Raw)"
            )
        )

        self.plot_widget.setTitle(
            signal_title
        )

        # =====================================================
        # SET INITIAL X RANGE
        # =====================================================

        self.plot_widget.setXRange(
            float(selected_time[0]),
            float(selected_time[-1]),
            padding=0
        )

        # =====================================================
        # FIXED Y RANGE
        # =====================================================

        self.update_fixed_y_scale()

        # =====================================================
        # STATUS
        # =====================================================

        self.time_domain_status.setText(
            f"{self.current_sensor} | "
            f"{len(selected_df):,} samples | "
            f"{start_time:.3f} → "
            f"{end_time:.3f} s | "
            f"X = Blue | "
            f"Y = Orange | "
            f"Z = Green | "
            f"FOG = Red | "
            f"Wheel = Zoom X | "
            f"Drag = Pan X | "
            f"Fixed Y-scale"
        )


    # =========================================================
    # UI REDESIGN - 1365 x 720
    # =========================================================

    def _apply_app_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #ffffff; color: #222222; }
            QGroupBox {
                border: 1px solid #b9b9b9;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QPushButton {
                min-height: 30px;
                padding: 4px 10px;
                border: 1px solid #b7b7b7;
                border-radius: 3px;
                background: #f7f7f7;
            }
            QPushButton:hover { background: #eeeeee; }
            QPushButton:checked { background: #4d78c2; color: white; border-color: #3e67aa; }
            QComboBox, QDoubleSpinBox, QSpinBox {
                min-height: 27px;
                border: 1px solid #b7b7b7;
                border-radius: 3px;
                padding: 2px 6px;
                background: white;
            }
            QTableView, QTableWidget {
                gridline-color: #d7d7d7;
                selection-background-color: #dce8f8;
                selection-color: #111111;
                alternate-background-color: #fafafa;
            }
            QHeaderView::section {
                background: #f2f2f2;
                border: 1px solid #d2d2d2;
                padding: 4px;
                font-weight: 600;
            }
        """)

    def create_ui(self):
        """Bố cục mới; toàn bộ logic Phase 1-4 cũ vẫn được giữ lại."""
        self._apply_app_style()
        self.setMinimumSize(1100, 620)
        self.resize(1365, 720)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # TOP BAR - giống bố cục ảnh tham chiếu.
        top = QHBoxLayout()
        top.setSpacing(12)
        title = QLabel("FOG SIGNAL ANALYZER")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        self.open_button = QPushButton("OPEN DATASET")
        self.open_button.setMinimumWidth(145)
        self.open_button.clicked.connect(self.open_dataset)
        self.path_label = QLabel("No dataset selected")
        self.path_label.setStyleSheet("color:#555; padding:3px;")
        self.path_label.setToolTip("Dataset path")
        top.addWidget(title)
        top.addSpacing(50)
        top.addWidget(self.open_button)
        top.addWidget(self.path_label, 1)
        root.addLayout(top)

        body = QHBoxLayout()
        body.setSpacing(6)

        # LEFT NAVIGATION
        nav = QWidget()
        nav.setFixedWidth(170)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 6, 0, 0)
        nav_layout.setSpacing(0)
        nav_title = QPushButton("Dataset")
        nav_title.setCheckable(True)
        nav_title.setChecked(True)
        nav_title.setMinimumHeight(38)
        nav_title.clicked.connect(self.show_dataset_page)
        self.nav_dataset = nav_title
        nav_layout.addWidget(nav_title)

        self.nav_buttons = []
        nav_items = [
            ("Time domain", self.show_time_domain),
            ("Preprocessing", self.show_preprocessing),
            ("Windowing", self.show_windowing),
            ("Time Analysis", self.show_analysis),
            ("Frequency Analysis", self.show_frequency_analysis),
            ("Publish data", self.show_publish_page),
        ]
        for text, slot in nav_items:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setMinimumHeight(46)
            b.clicked.connect(slot)
            nav_layout.addWidget(b)
            self.nav_buttons.append(b)
        nav_layout.addStretch(1)
        body.addWidget(nav)

        self.pages = QStackedWidget()
        self.dataset_page = self.create_dataset_page_redesigned()
        self.time_domain_page = self.create_time_domain_page()
        self.plot_widget.setAntialiasing(True)
        self.plot_widget.getPlotItem().sigRangeChanged.connect(self._force_time_domain_antialias)
        self.preprocessing_page = self.create_preprocessing_page()
        self.windowing_page = self.create_windowing_page()
        self.time_analysis_page = self.create_time_analysis_page()
        self.frequency_analysis_page = self.create_frequency_analysis_page()
        self.publish_page = self.create_publish_page()
        # Alias để Phase 5 cũ vẫn không bị phá vỡ.
        self.analysis_page = self.time_analysis_page

        for page in [
            self.dataset_page,
            self.time_domain_page,
            self.preprocessing_page,
            self.windowing_page,
            self.time_analysis_page,
            self.frequency_analysis_page,
            self.publish_page,
        ]:
            self.pages.addWidget(page)
        body.addWidget(self.pages, 1)
        root.addLayout(body, 1)
        self.pages.setCurrentWidget(self.dataset_page)
        self._set_nav_active(0)

    def _set_nav_active(self, index):
        if not hasattr(self, "nav_buttons"):
            return
        self.nav_dataset.setChecked(index == 0)
        for i, button in enumerate(self.nav_buttons, start=1):
            button.setChecked(i == index)

    def _plot_common(self, plot, x_label, y_label, antialias=True):
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.20)
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        plot.setAntialiasing(bool(antialias))
        vb = plot.getPlotItem().vb
        vb.setMouseEnabled(x=True, y=False)
        plot.enableAutoRange(x=False, y=False)

    def _reset_plot_x(self, plot, x):
        if x is None or len(x) < 2:
            return
        plot.setXRange(float(x[0]), float(x[-1]), padding=0)

    def create_dataset_page_redesigned(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        left = QWidget()
        left.setFixedWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(7)

        info = QGroupBox("DATASET INFORMATION")
        il = QGridLayout(info)
        self.file_value = QLabel("-")
        self.columns_value = QLabel("-")
        self.samples_value = QLabel("-")
        self.fs_value = QLabel("-")
        self.duration_value = QLabel("-")
        for r, name, w in [
            (0,"File:",self.file_value),(1,"Columns:",self.columns_value),
            (2,"Samples:",self.samples_value),(3,"Sampling Rate:",self.fs_value),
            (4,"Duration:",self.duration_value)]:
            il.addWidget(QLabel(name), r, 0)
            il.addWidget(w, r, 1)
            w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ll.addWidget(info)

        label_group = QGroupBox("LABEL CONFIGURATION")
        lv = QVBoxLayout(label_group)
        self.label_table = QTableWidget()
        self.label_table.setColumnCount(3)
        self.label_table.setHorizontalHeaderLabels(["Original Label", "Label Name", "Binary"])
        self.label_table.verticalHeader().setVisible(False)
        self.label_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.label_table.setSelectionMode(QTableWidget.NoSelection)
        self.label_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.label_table.setMinimumHeight(165)
        lv.addWidget(self.label_table)
        ll.addWidget(label_group)
        ll.addStretch(1)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        preview_group = QGroupBox("PREVIEW DATA — ALL ROWS")
        pv = QVBoxLayout(preview_group)
        self.preview_table = QTableView()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.preview_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.preview_table.setModel(DataFrameTableModel(parent=self.preview_table))
        self.preview_model = self.preview_table.model()
        pv.addWidget(self.preview_table, 1)
        self.preview_status = QLabel("0 rows")
        self.preview_status.setAlignment(Qt.AlignCenter)
        pv.addWidget(self.preview_status)
        rl.addWidget(preview_group, 1)

        self.time_domain_button = QPushButton("TIME DOMAIN")
        self.time_domain_button.setMinimumHeight(38)
        self.time_domain_button.clicked.connect(self.show_time_domain)
        rl.addWidget(self.time_domain_button)
        layout.addWidget(left)
        layout.addWidget(right, 1)
        return page

    def show_preview(self):
        """Hiển thị toàn bộ DataFrame bằng model/view, tránh tạo hàng nghìn QTableWidgetItem."""
        if self.df is None:
            return
        self.preview_model.set_dataframe(self.df)
        self.preview_table.resizeColumnsToContents()
        self.preview_status.setText(f"{len(self.df):,} rows × {len(self.df.columns):,} columns — full dataset")

    def set_label_mapping_enabled(self, checked=True):
        if self.df_raw is None:
            if hasattr(self, "label_mapping_toggle"):
                self.label_mapping_toggle.setChecked(False)
            return
        checked = bool(checked)
        self.label_mapping_toggle.blockSignals(True)
        self.label_mapping_toggle.setChecked(checked)
        self.label_mapping_toggle.setText("BINARY MAPPING: ON" if checked else "BINARY MAPPING: OFF")
        self.label_mapping_toggle.blockSignals(False)

        # Không sửa df_raw. Mapping chỉ tạo trên bản hiển thị/đầu vào hiện tại.
        base = self.df_raw.copy()
        label_column = None
        if "Label" in base.columns:
            label_column = "Label"
        else:
            for c in base.columns:
                if str(c).lower() in ["label", "class", "target", "state"]:
                    label_column = c
                    break
        if label_column is None:
            QMessageBox.warning(self, "Label Mapping", "Không tìm thấy cột label.")
            return
        base["Original_Label"] = base[label_column]
        if checked:
            base["Binary_Label"] = base[label_column].apply(lambda x: 1 if x == 2 else 0)
        elif "Binary_Label" in base.columns:
            base.drop(columns=["Binary_Label"], inplace=True)
        self.df = base
        self.display_df = self.df.copy()
        self.display_sampling_rate = self.sampling_rate
        self.show_preview()
        self.detect_labels()
        self._update_label_mapping_status()

    def _update_label_mapping_status(self):
        if hasattr(self, "label_mapping_status"):
            state = "ON" if self.label_mapping_toggle.isChecked() else "OFF"
            self.label_mapping_status.setText(f"Binary mapping: {state}")

    # =========================================================
    # PHASE 5 - SHARED SELECTOR / ANALYSIS PAGES
    # =========================================================

    def _make_analysis_selector(self, parent_layout, frequency=False):
        group = QGroupBox("WINDOW SELECTION")
        g = QGridLayout(group)
        prefix = "frequency" if frequency else "analysis"
        combo = QComboBox()
        sensor = QComboBox(); sensor.addItems(["Ankle", "Thigh", "Trunk"])
        axis = QComboBox(); axis.addItems(["X", "Y", "Z"])
        if frequency:
            self.frequency_window_combo = combo
            self.frequency_sensor_combo = sensor
            self.frequency_axis_combo = axis
            combo.currentIndexChanged.connect(self.on_frequency_window_changed)
            sensor.currentTextChanged.connect(self.on_frequency_selection_changed)
            axis.currentTextChanged.connect(self.on_frequency_selection_changed)
        else:
            self.analysis_window_combo = combo
            self.analysis_sensor_combo = sensor
            self.analysis_axis_combo = axis
            combo.currentIndexChanged.connect(self.on_analysis_window_changed)
            sensor.currentTextChanged.connect(self.on_analysis_selection_changed)
            axis.currentTextChanged.connect(self.on_analysis_selection_changed)
        g.addWidget(QLabel("Window:"), 0, 0); g.addWidget(combo, 0, 1, 1, 3)
        g.addWidget(QLabel("Sensor:"), 0, 4); g.addWidget(sensor, 0, 5)
        g.addWidget(QLabel("Axis:"), 0, 6); g.addWidget(axis, 0, 7)
        label = QLabel("-"); timev = QLabel("-"); samples = QLabel("-")
        if frequency:
            self.frequency_label_value=label; self.frequency_time_value=timev; self.frequency_samples_value=samples
        else:
            self.analysis_label_value=label; self.analysis_time_value=timev; self.analysis_samples_value=samples
        g.addWidget(QLabel("Label:"),1,0); g.addWidget(label,1,1,1,3)
        g.addWidget(QLabel("Time:"),1,4); g.addWidget(timev,1,5)
        g.addWidget(QLabel("Samples:"),1,6); g.addWidget(samples,1,7)
        parent_layout.addWidget(group)

    def create_time_analysis_page(self):
        page=QWidget(); main=QVBoxLayout(page); main.setContentsMargins(0,0,0,0); main.setSpacing(5)
        h=QHBoxLayout(); title=QLabel("TIME ANALYSIS"); title.setStyleSheet("font-size:21px;font-weight:700;")
        self.time_analysis_run_button=QPushButton("ANALYZE WINDOW"); self.time_analysis_run_button.clicked.connect(self.analyze_selected_window)
        reset=QPushButton("RESET"); reset.clicked.connect(self.reset_analysis)
        back=QPushButton("BACK TO WINDOWING"); back.clicked.connect(self.show_windowing_from_analysis)
        h.addWidget(title); h.addStretch(); h.addWidget(reset); h.addWidget(self.time_analysis_run_button); h.addWidget(back); main.addLayout(h)
        self._make_analysis_selector(main, False)

        plots=QHBoxLayout()
        raw_group=QGroupBox("TIME DOMAIN — RAW DATA"); raw_l=QVBoxLayout(raw_group)
        self.analysis_raw_time_plot=pg.PlotWidget(); self._plot_common(self.analysis_raw_time_plot,"Time (s)","Acceleration (g)",True)
        raw_l.addWidget(self.analysis_raw_time_plot,1); plots.addWidget(raw_group,1)
        proc_group=QGroupBox("TIME DOMAIN — AFTER PREPROCESSING") ; proc_l=QVBoxLayout(proc_group)
        self.analysis_time_plot=pg.PlotWidget(); self._plot_common(self.analysis_time_plot,"Time (s)","Acceleration (g)",True)
        proc_l.addWidget(self.analysis_time_plot,1); plots.addWidget(proc_group,1)
        main.addLayout(plots,1)
        feature_group=QGroupBox("TIME-DOMAIN FEATURES") ; fl=QGridLayout(feature_group)
        self.time_feature_labels={}
        names=["Mean","RMS","Std","Variance","Min","Max","SMA"]
        for i,name in enumerate(names):
            lab=QLabel(f"{name}: -"); self.time_feature_labels[name]=lab; fl.addWidget(lab,i//4,i%4)
        main.addWidget(feature_group)
        self.analysis_time_stats=QLabel("Select a window and press ANALYZE WINDOW."); self.analysis_time_stats.setStyleSheet("padding:2px;color:#555;")
        main.addWidget(self.analysis_time_stats)
        self.analysis_status=QLabel("Apply Windowing, then select a window."); self.analysis_status.setAlignment(Qt.AlignCenter); self.analysis_status.setStyleSheet("color:#555;padding:2px;")
        main.addWidget(self.analysis_status)
        return page

    def create_frequency_analysis_page(self):
        page=QWidget(); main=QVBoxLayout(page); main.setContentsMargins(0,0,0,0); main.setSpacing(5)
        h=QHBoxLayout(); title=QLabel("FREQUENCY ANALYSIS"); title.setStyleSheet("font-size:21px;font-weight:700;")
        run=QPushButton("ANALYZE FREQUENCY"); run.clicked.connect(self.analyze_frequency_selected_window)
        reset=QPushButton("RESET"); reset.clicked.connect(self.reset_frequency_analysis)
        h.addWidget(title); h.addStretch(); h.addWidget(reset); h.addWidget(run); main.addLayout(h)
        self._make_analysis_selector(main, True)
        plots=QHBoxLayout()
        fftg=QGroupBox("FFT SPECTRUM"); fftl=QVBoxLayout(fftg)
        self.analysis_fft_plot=pg.PlotWidget(); self._plot_common(self.analysis_fft_plot,"Frequency (Hz)","Magnitude",True); fftl.addWidget(self.analysis_fft_plot); plots.addWidget(fftg,1)
        psdg=QGroupBox("POWER SPECTRAL DENSITY (PSD)"); psdl=QVBoxLayout(psdg)
        self.analysis_psd_plot=pg.PlotWidget(); self._plot_common(self.analysis_psd_plot,"Frequency (Hz)","PSD (g²/Hz)",True); psdl.addWidget(self.analysis_psd_plot); plots.addWidget(psdg,1)
        main.addLayout(plots,1)
        fg=QGroupBox("FREQUENCY-DOMAIN FEATURES"); fgl=QGridLayout(fg)
        self.frequency_feature_labels={}
        names=["Dominant Frequency","Total Power","Loco Power","Freeze Power","Freeze Index","Low Freq Ratio","PSE"]
        for i,name in enumerate(names):
            lab=QLabel(f"{name}: -"); self.frequency_feature_labels[name]=lab; fgl.addWidget(lab,i//4,i%4)
        main.addWidget(fg)
        self.analysis_frequency_stats=QLabel("FFT / PSD features: -"); self.analysis_frequency_stats.setWordWrap(True); main.addWidget(self.analysis_frequency_stats)
        self.frequency_status=QLabel("Select a window and press ANALYZE FREQUENCY."); self.frequency_status.setAlignment(Qt.AlignCenter); self.frequency_status.setStyleSheet("color:#555;padding:2px;"); main.addWidget(self.frequency_status)
        return page

    def create_publish_page(self):
        page=QWidget(); main=QVBoxLayout(page); main.setContentsMargins(0,0,0,0)
        h=QHBoxLayout(); title=QLabel("PUBLISH DATA"); title.setStyleSheet("font-size:21px;font-weight:700;")
        h.addWidget(title); h.addStretch(); main.addLayout(h)
        info=QGroupBox("ML FEATURE DATASET"); il=QVBoxLayout(info)
        self.publish_info=QLabel("Windowing + Analysis features will be exported as CSV.")
        self.publish_info.setWordWrap(True); il.addWidget(self.publish_info)
        self.publish_table=QTableWidget(); self.publish_table.setColumnCount(8); self.publish_table.setHorizontalHeaderLabels(["Window","Sensor","Axis","Label","Start","End","RMS","Freeze Index"]); self.publish_table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.publish_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); il.addWidget(self.publish_table,1)
        self.publish_button=QPushButton("PUBLISH FEATURES → CSV"); self.publish_button.setMinimumHeight(40); self.publish_button.clicked.connect(self.publish_features_csv); il.addWidget(self.publish_button)
        main.addWidget(info,1); return page

    # =========================================================
    # PHASE 5 - NAVIGATION / SELECTION
    # =========================================================

    def show_analysis(self):
        if not self.windowing_applied or not self.window_metadata:
            QMessageBox.warning(self,"Analysis Not Available","Hãy APPLY WINDOWING trước khi phân tích.")
            return
        self.update_analysis_window_list()
        self.pages.setCurrentWidget(self.time_analysis_page)
        self._set_nav_active(4)

    def show_windowing_from_analysis(self):
        self.pages.setCurrentWidget(self.windowing_page); self._set_nav_active(3)

    def show_frequency_analysis(self):
        if not self.windowing_applied or not self.window_metadata:
            QMessageBox.warning(self,"Frequency Analysis","Hãy APPLY WINDOWING trước khi phân tích tần số.")
            return
        self.update_analysis_window_list()
        self.pages.setCurrentWidget(self.frequency_analysis_page); self._set_nav_active(5)

    def show_publish_page(self):
        self.pages.setCurrentWidget(self.publish_page); self._set_nav_active(6); self.update_publish_preview()

    def show_dataset_page(self):
        if self.df is not None:
            self.display_df=self.df.copy(); self.display_sampling_rate=self.sampling_rate; self.display_processed_label=False
            self.show_preview()
        self.pages.setCurrentWidget(self.dataset_page); self._set_nav_active(0)

    def show_time_domain(self):
        if self.df is None:
            QMessageBox.warning(self,"No Dataset","Hãy mở dataset trước."); return
        self.display_df=self.df.copy(); self.display_sampling_rate=self.sampling_rate; self.display_processed_label=False
        self.pages.setCurrentWidget(self.time_domain_page); self.update_time_controls(); self.load_time_range(); self._set_nav_active(1)

    def show_preprocessing(self):
        if self.df is None:
            QMessageBox.warning(self,"No Dataset","Hãy mở dataset trước."); return
        self.pages.setCurrentWidget(self.preprocessing_page); self.update_preprocessing_controls(); self._set_nav_active(2)

    def show_windowing(self):
        # Giữ nguyên logic Phase 4, sau đó chỉ đổi nav.
        if self.df_selected is not None and self.processed_sampling_rate:
            self.windowing_input_df=self.df_selected.copy(); self.windowing_sampling_rate=self.processed_sampling_rate; source_text="Phase 3 Processed Data"
        elif self.df_raw is not None and self.sampling_rate:
            self.windowing_input_df=self.df_raw.copy(); self.windowing_sampling_rate=self.sampling_rate; source_text="Raw Data (Phase 3 not applied)"
        else:
            QMessageBox.warning(self,"No Input Data","Hãy mở dataset trước hoặc APPLY PREPROCESSING."); return
        self.windowing_applied=False; self.window_metadata=None; self.window_signal_data=[]
        self.windowing_source_value.setText(source_text); self.windowing_result_fs_value.setText(f"{self.windowing_sampling_rate:g} Hz"); self.windowing_sensor_combo.setCurrentText(self.current_sensor)
        self.update_windowing_input_info(); self.update_window_label_controls(); self.update_window_parameter_calculation()
        self.pages.setCurrentWidget(self.windowing_page); self._set_nav_active(3)

    def update_analysis_window_list(self):
        if not self.window_metadata: return
        combos=[getattr(self,"analysis_window_combo",None),getattr(self,"frequency_window_combo",None)]
        for combo in combos:
            if combo is None: continue
            combo.blockSignals(True); combo.clear()
            for m in self.window_metadata:
                label="FOG" if int(m["Binary_Label"])==1 else "Non-FOG"
                combo.addItem(f"Window {m['Window_ID']} | {m['Start_Time']:.3f}–{m['End_Time']:.3f} s | {label} | FOG {m['FOG_Ratio']*100:.1f}%",m["Window_ID"])
            combo.blockSignals(False)
        if self.analysis_window_combo.count(): self.analysis_window_combo.setCurrentIndex(0); self.on_analysis_window_changed(0)
        if self.frequency_window_combo.count(): self.frequency_window_combo.setCurrentIndex(0); self.on_frequency_window_changed(0)

    def _update_analysis_meta(self, prefix, index):
        combo=getattr(self,f"{prefix}_window_combo")
        if index<0 or not self.window_metadata: return
        wid=combo.itemData(index)
        if wid is None: return
        wid=int(wid)
        meta=next((m for m in self.window_metadata if int(m["Window_ID"])==wid),None)
        if meta is None: return
        label="FOG" if int(meta["Binary_Label"])==1 else "Non-FOG"
        getattr(self,f"{prefix}_label_value").setText(label)
        getattr(self,f"{prefix}_time_value").setText(f"{meta['Start_Time']:.3f} → {meta['End_Time']:.3f} s")
        getattr(self,f"{prefix}_samples_value").setText(f"{meta['Samples']:,}")
        self.analysis_window_id=wid
        self._clear_analysis_plots_only()
        self._reset_analysis_feature_labels()

    def on_analysis_window_changed(self,index): self._update_analysis_meta("analysis",index)
    def on_frequency_window_changed(self,index): self._update_analysis_meta("frequency",index)
    def on_analysis_selection_changed(self,*args):
        self._clear_analysis_plots_only()
    def on_frequency_selection_changed(self,*args):
        self._clear_analysis_plots_only()

    def _clear_analysis_plots_only(self):
        for name in ["analysis_raw_time_plot","analysis_time_plot","analysis_fft_plot","analysis_psd_plot"]:
            w=getattr(self,name,None)
            if w is not None:
                w.clear()
                w.enableAutoRange(x=False,y=False)

    def _reset_analysis_feature_labels(self):
        for d in [getattr(self,"time_feature_labels",{}),getattr(self,"frequency_feature_labels",{})]:
            for k,w in d.items(): w.setText(f"{k}: -")
        if hasattr(self,"analysis_time_stats"): self.analysis_time_stats.setText("Select a window and press ANALYZE WINDOW.")
        if hasattr(self,"analysis_frequency_stats"): self.analysis_frequency_stats.setText("FFT / PSD features: -")

    def reset_analysis(self):
        self.analysis_signal=None; self.analysis_time=None; self.analysis_fft_freq=None; self.analysis_fft_mag=None; self.analysis_psd_freq=None; self.analysis_psd_power=None
        self._clear_analysis_plots_only(); self._reset_analysis_feature_labels()
        if hasattr(self,"analysis_status"): self.analysis_status.setText("Analysis reset. Select a window and press ANALYZE WINDOW.")

    def reset_frequency_analysis(self):
        self._clear_analysis_plots_only();
        for d in [getattr(self,"frequency_feature_labels",{})]:
            for k,w in d.items(): w.setText(f"{k}: -")
        self.analysis_frequency_stats.setText("FFT / PSD features: -")
        self.frequency_status.setText("Frequency analysis reset.")

    # =========================================================
    # PHASE 5 - DATA EXTRACTION / FEATURES
    # =========================================================

    def _analysis_selected_window(self, combo=None):
        if not self.window_signal_data: raise ValueError("Chưa có window. Hãy APPLY WINDOWING trước.")
        if combo is None: combo=self.analysis_window_combo
        idx=combo.currentIndex()
        if idx<0: raise ValueError("Chưa chọn window.")
        wid=int(combo.itemData(idx))
        item=next((x for x in self.window_signal_data if int(x["Window_ID"])==wid),None)
        meta=next((x for x in self.window_metadata if int(x["Window_ID"])==wid),None)
        if item is None or meta is None: raise ValueError(f"Không tìm thấy Window {wid}.")
        return wid,item,meta

    def _get_processed_signal_for(self, item, sensor, axis):
        cols={"Ankle":["Ankle_Forward_mg","Ankle_Vertical_mg","Ankle_Lateral_mg"],"Thigh":["Thigh_Forward_mg","Thigh_Vertical_mg","Thigh_Lateral_mg"],"Trunk":["Trunk_Forward_mg","Trunk_Vertical_mg","Trunk_Lateral_mg"]}
        col=cols[sensor][{"X":0,"Y":1,"Z":2}[axis]]
        if col not in item["Signals"]: raise ValueError(f"Windowing chưa lưu {sensor} Axis {axis}. Hãy chọn axis tương ứng trong Phase 4.")
        return np.asarray(item["Time"],float),np.asarray(item["Signals"][col],float),col

    def _get_raw_signal_for(self, time_array, column):
        if self.df_raw is None or column not in self.df_raw.columns:
            return None
        raw_t=self.get_dataframe_time_array(self.df_raw,self.sampling_rate)
        if raw_t is None or len(raw_t)<2: return None
        raw_x=self.df_raw[column].to_numpy(dtype=float)/1000.0
        return np.interp(time_array,raw_t,raw_x)

    def _compute_features(self,x,fs):
        x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
        if len(x)<2: raise ValueError("Window phải có ít nhất 2 sample.")
        mean=float(np.mean(x)); rms=float(np.sqrt(np.mean(x*x))); std=float(np.std(x)); var=float(np.var(x)); mn=float(np.min(x)); mx=float(np.max(x)); sma=float(np.mean(np.abs(x)))
        xc=x-mean; n=len(xc)
        freq=np.fft.rfftfreq(n,1.0/fs); mag=(2.0/n)*np.abs(np.fft.rfft(xc));
        if len(mag)>1: mag[0]*=0.5
        di=int(np.argmax(mag[1:])+1) if len(mag)>1 else 0
        dom=float(freq[di])
        nperseg=min(256,n); noverlap=nperseg//2 if nperseg>=2 else 0
        pf,pp=signal.welch(xc,fs=fs,window="hann",nperseg=nperseg,noverlap=noverlap,detrend=False,scaling="density")
        integ=lambda y,z: float(np.trapezoid(y,z) if hasattr(np,"trapezoid") else np.trapz(y,z)) if len(y)>1 else 0.0
        total=integ(pp,pf); lm=(pf>=0.5)&(pf<=3.0); fm=(pf>3.0)&(pf<=8.0); loco=integ(pp[lm],pf[lm]); freeze=integ(pp[fm],pf[fm]); fi=freeze/loco if loco>0 else 0.0; low=loco/total if total>0 else 0.0
        ps=pp/np.sum(pp) if np.sum(pp)>0 else pp; ps=ps[ps>0]; pse=float(-np.sum(ps*np.log2(ps))) if len(ps) else 0.0
        return {"Mean":mean,"RMS":rms,"Std":std,"Variance":var,"Min":mn,"Max":mx,"SMA":sma,"Dominant Frequency":dom,"Total Power":total,"Loco Power":loco,"Freeze Power":freeze,"Freeze Index":fi,"Low Freq Ratio":low,"PSE":pse,"FFT_Freq":freq,"FFT_Mag":mag,"PSD_Freq":pf,"PSD_Power":pp}

    def analyze_selected_window(self):
        try:
            wid,item,meta=self._analysis_selected_window(self.analysis_window_combo)
            sensor=self.analysis_sensor_combo.currentText(); axis=self.analysis_axis_combo.currentText()
            t,processed,column=self._get_processed_signal_for(item,sensor,axis); fs=float(self.windowing_sampling_rate)
            raw=self._get_raw_signal_for(t,column); raw=processed.copy() if raw is None else raw
            proc_feat=self._compute_features(processed,fs); raw_feat=self._compute_features(raw,fs)
            self.analysis_window_id=wid; self.analysis_signal=processed; self.analysis_time=t; self.analysis_fs=fs
            self.analysis_fft_freq=proc_feat["FFT_Freq"]; self.analysis_fft_mag=proc_feat["FFT_Mag"]; self.analysis_psd_freq=proc_feat["PSD_Freq"]; self.analysis_psd_power=proc_feat["PSD_Power"]

            for plot,data,title in [(self.analysis_raw_time_plot,raw,f"Raw | Window {wid} | {sensor} {axis}"),(self.analysis_time_plot,processed,f"After Preprocessing | Window {wid} | {sensor} {axis}")]:
                plot.clear(); plot.plot(t,data,pen=pg.mkPen(width=1.25),antialias=True); self._reset_plot_x(plot,t)
                finite=data[np.isfinite(data)];
                if len(finite):
                    lo,hi=float(np.min(finite)),float(np.max(finite)); amp=max(hi-lo,0.01); pad=amp*0.08; plot.setYRange(lo-pad,hi+pad,padding=0)
                plot.setTitle(title)

            for name in ["Mean","RMS","Std","Variance","Min","Max","SMA"]: self.time_feature_labels[name].setText(f"{name}: {proc_feat[name]:.6f} g" if name not in ["Variance"] else f"{name}: {proc_feat[name]:.6f} g²")
            self.analysis_time_stats.setText(f"Processed features | Mean {proc_feat['Mean']:.6f} g | RMS {proc_feat['RMS']:.6f} g | Std {proc_feat['Std']:.6f} g | SMA {proc_feat['SMA']:.6f} g")
            self.analysis_status.setText(f"Window {wid} | {('FOG' if int(meta['Binary_Label']) else 'Non-FOG')} | {sensor} {axis} | Fs={fs:g} Hz | Raw + Preprocessed ready")
            self._plot_frequency_from_features(proc_feat,wid,meta,sensor,axis)
            self.update_publish_preview()
        except Exception as error:
            QMessageBox.critical(self,"Time Analysis Error",str(error))

    def _plot_frequency_from_features(self,feat,wid,meta,sensor,axis):
        self.analysis_fft_plot.clear(); self.analysis_psd_plot.clear()
        self.analysis_fft_plot.plot(feat["FFT_Freq"],feat["FFT_Mag"],pen=pg.mkPen(width=1.25),antialias=True); self._reset_plot_x(self.analysis_fft_plot,feat["FFT_Freq"])
        self.analysis_psd_plot.plot(feat["PSD_Freq"],feat["PSD_Power"],pen=pg.mkPen(width=1.25),antialias=True); self._reset_plot_x(self.analysis_psd_plot,feat["PSD_Freq"])
        for plot,y in [(self.analysis_fft_plot,feat["FFT_Mag"]),(self.analysis_psd_plot,feat["PSD_Power"])]:
            finite=y[np.isfinite(y)];
            if len(finite):
                lo=max(0.0,float(np.min(finite))); hi=float(np.max(finite)); plot.setYRange(lo,hi*1.08 if hi>0 else 1.0,padding=0)
        self.analysis_fft_plot.setTitle(f"FFT Spectrum | Window {wid}")
        self.analysis_psd_plot.setTitle(f"PSD (Welch) | Window {wid}")
        vals=[("Dominant Frequency",feat["Dominant Frequency"]), ("Total Power",feat["Total Power"]),("Loco Power",feat["Loco Power"]),("Freeze Power",feat["Freeze Power"]),("Freeze Index",feat["Freeze Index"]),("Low Freq Ratio",feat["Low Freq Ratio"]),("PSE",feat["PSE"])]
        for name,v in vals: self.frequency_feature_labels[name].setText(f"{name}: {v:.6f}" if name!="Dominant Frequency" else f"{name}: {v:.3f} Hz")
        self.analysis_frequency_stats.setText(f"Window {wid} | {sensor} {axis} | Dominant={feat['Dominant Frequency']:.3f} Hz | Total Power={feat['Total Power']:.6g} | FI={feat['Freeze Index']:.6f} | PSE={feat['PSE']:.4f}")

    def analyze_frequency_selected_window(self):
        try:
            self.analysis_window_combo.setCurrentIndex(self.frequency_window_combo.currentIndex())
            self.analysis_sensor_combo.setCurrentText(self.frequency_sensor_combo.currentText())
            self.analysis_axis_combo.setCurrentText(self.frequency_axis_combo.currentText())
            self.analyze_selected_window()
            self.pages.setCurrentWidget(self.frequency_analysis_page); self._set_nav_active(5)
            self.frequency_status.setText(self.analysis_status.text())
            # Đồng bộ selector sau khi phân tích.
            self.frequency_window_combo.setCurrentIndex(self.analysis_window_combo.currentIndex())
        except Exception as error:
            QMessageBox.critical(self,"Frequency Analysis Error",str(error))

    # =========================================================
    # PUBLISH - FEATURE CSV FOR ML
    # =========================================================

    def _all_feature_rows(self):
        if not self.windowing_applied or not self.window_metadata or not self.window_signal_data:
            raise ValueError("Hãy APPLY WINDOWING trước khi Publish.")
        fs=float(self.windowing_sampling_rate)
        rows=[]
        cols_by_sensor={"Ankle":["Ankle_Forward_mg","Ankle_Vertical_mg","Ankle_Lateral_mg"],"Thigh":["Thigh_Forward_mg","Thigh_Vertical_mg","Thigh_Lateral_mg"],"Trunk":["Trunk_Forward_mg","Trunk_Vertical_mg","Trunk_Lateral_mg"]}
        for item,meta in zip(self.window_signal_data,self.window_metadata):
            for sensor,cols in cols_by_sensor.items():
                for axis,col in zip(["X","Y","Z"],cols):
                    if col not in item["Signals"]: continue
                    f=self._compute_features(np.asarray(item["Signals"][col],float),fs)
                    row={"Window_ID":int(meta["Window_ID"]),"Sensor":sensor,"Axis":axis,"Start_Time":float(meta["Start_Time"]),"End_Time":float(meta["End_Time"]),"Binary_Label":int(meta["Binary_Label"]),"FOG_Ratio":float(meta["FOG_Ratio"])}
                    for k in ["Mean","RMS","Std","Variance","Min","Max","SMA","Dominant Frequency","Total Power","Loco Power","Freeze Power","Freeze Index","Low Freq Ratio","PSE"]: row[k.replace(" ","_")]=f[k]
                    rows.append(row)
        return pd.DataFrame(rows)

    def update_publish_preview(self):
        if not hasattr(self,"publish_table"): return
        try:
            df=self._all_feature_rows()
            self.publish_info.setText(f"Ready: {len(df):,} feature rows. CSV contains time/frequency features + Binary_Label for ML.")
            preview=df.head(1000)
            self.publish_table.setRowCount(len(preview)); self.publish_table.setColumnCount(8)
            self.publish_table.setHorizontalHeaderLabels(["Window","Sensor","Axis","Label","Start","End","RMS","Freeze Index"])
            for r,(_,row) in enumerate(preview.iterrows()):
                vals=[row["Window_ID"],row["Sensor"],row["Axis"],row["Binary_Label"],f"{row['Start_Time']:.3f}",f"{row['End_Time']:.3f}",f"{row['RMS']:.6f}",f"{row['Freeze_Index']:.6f}"]
                for c,v in enumerate(vals):
                    it=QTableWidgetItem(str(v)); it.setTextAlignment(Qt.AlignCenter); self.publish_table.setItem(r,c,it)
        except Exception:
            self.publish_info.setText("No feature data yet. APPLY WINDOWING first."); self.publish_table.setRowCount(0)

    def publish_features_csv(self):
        try:
            df=self._all_feature_rows()
            if df.empty: raise ValueError("Không có feature để xuất.")
            default=os.path.splitext(os.path.basename(self.file_path or "fog_dataset"))[0]+"_ML_features.csv"
            path,_=QFileDialog.getSaveFileName(self,"Publish ML Feature CSV",default,"CSV Files (*.csv)")
            if not path: return
            df.to_csv(path,index=False)
            self.publish_info.setText(f"Published {len(df):,} rows → {path}")
            QMessageBox.information(self,"Publish Data",f"Đã xuất CSV cho ML:\n\n{path}\n\nRows: {len(df):,}\nColumns: {len(df.columns):,}")
        except Exception as error:
            QMessageBox.critical(self,"Publish Error",str(error))

    # =========================================================
    # FIX GRAPH BEHAVIOR FOR PHASE 2 / 5
    # =========================================================

    def reset_view(self):
        if self.current_time_array is None or len(self.current_time_array)<2: return
        self.plot_widget.setXRange(float(self.current_time_array[0]),float(self.current_time_array[-1]),padding=0)
        if self.fixed_y_range is not None: self.plot_widget.setYRange(*self.fixed_y_range,padding=0)

    def show_time_domain_from_preprocessing(self):
        if self.display_df is None and self.df is not None:
            self.display_df=self.df.copy(); self.display_sampling_rate=self.sampling_rate; self.display_processed_label=False
        self.pages.setCurrentWidget(self.time_domain_page); self.update_time_controls(); self.load_time_range(); self._set_nav_active(1)

    def show_preprocessing_from_windowing(self):
        self.pages.setCurrentWidget(self.preprocessing_page); self.update_preprocessing_controls(); self._set_nav_active(2)

    def view_processed_time_domain(self):
        if not self.preprocessing_applied or self.df_selected is None:
            QMessageBox.warning(self,"No Processed Data","Hãy APPLY PREPROCESSING trước."); return
        self.display_df=self.df_selected.copy(); self.display_sampling_rate=self.processed_sampling_rate; self.display_processed_label=True
        self.pages.setCurrentWidget(self.time_domain_page); self.update_time_controls(); self.load_time_range(); self._set_nav_active(1)



# =============================================================
# MAIN
# =============================================================


# ============================================================================
# GUI EXTENSION: original FOGSignalAnalyzer above is retained as CoreFOGAnalyzer.
# The extension changes presentation and adds selection/batch orchestration.
# Original numerical methods are called unchanged through LegacyEngine.
# ============================================================================
import json
import re
import hashlib
import tempfile
from pathlib import Path
from types import MethodType
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QListWidget, QProgressBar, QSizePolicy

pg.setConfigOptions(antialias=False, background='w', foreground='#26364d', leftButtonPan=True)
CoreFOGAnalyzer = FOGSignalAnalyzer
GUI_SENSORS = ('Ankle', 'Thigh', 'Trunk')
GUI_AXES = ('Forward', 'Vertical', 'Lateral')
GUI_FEATURES = ('Mean','RMS','Std','Variance','Min','Max','SMA','Dominant Frequency',
                'Total Power','Loco Power','Freeze Power','Freeze Index','Low Freq Ratio','PSE')
GUI_COLUMNS = ['Time_ms'] + [f'{s}_{a}_mg' for s in GUI_SENSORS for a in GUI_AXES] + ['Label']

class FrozenControl:
    """Read-only values; numerical workers never access Qt widgets."""
    def __init__(self, value): self._value = value
    def value(self): return self._value
    def currentText(self): return self._value
    def isChecked(self): return bool(self._value)

class LegacyEngine:
    """Adapter to the original, unchanged numerical methods."""
    def __init__(self, config):
        self.config = config
        for name,value in config['controls'].items(): setattr(self,name,FrozenControl(value))
    def __getattr__(self,name):
        method = getattr(CoreFOGAnalyzer,name,None)
        if callable(method): return MethodType(method,self)
        raise AttributeError(name)


def gui_read_raw(path):
    path=Path(path)
    if path.suffix.lower()=='.txt':
        frame=pd.read_csv(path,sep=r'\s+',header=None)
        if frame.shape[1]!=11: raise ValueError('DAPHNET TXT cần đúng 11 cột.')
        frame.columns=GUI_COLUMNS
    else:
        frame=pd.read_csv(path)
        missing=set(GUI_COLUMNS)-set(frame.columns)
        if missing: raise ValueError('CSV raw thiếu cột: '+', '.join(sorted(missing)))
        frame=frame[GUI_COLUMNS].copy()
    if len(frame)<2: raise ValueError('Bản ghi cần ít nhất 2 mẫu.')
    for col in GUI_COLUMNS: frame[col]=pd.to_numeric(frame[col],errors='raise')
    if not np.isfinite(frame.to_numpy(float)).all(): raise ValueError('Raw chứa NaN/Inf.')
    if not frame.Label.isin([0,1,2]).all(): raise ValueError('Nhãn DAPHNET phải là 0, 1 hoặc 2.')
    dt=np.diff(frame.Time_ms.to_numpy(float)/1000)
    if not np.all(dt>0): raise ValueError('Timestamp phải tăng nghiêm ngặt.')
    if not np.isclose(np.median(dt),1/64,rtol=.04): raise ValueError('File raw không có bước thời gian DAPHNET 64 Hz.')
    return frame


def gui_labels(frame):
    frame=frame.copy();frame['Original_Label']=frame.Label
    frame['Binary_Label']=frame.Label.map({0:'Outside Experiment', 1:0., 2:1.})
    frame['Valid_Experiment']=frame.Label.isin([1,2])
    return frame


def gui_split(frame,fs=64.,valid_only=False):
    if frame is None or frame.empty:return []
    data=frame.loc[frame.Original_Label.isin([1,2])].copy() if valid_only else frame.copy()
    if data.empty:return []
    t=data.Time_ms.to_numpy(float)/1000
    valid=data.Original_Label.isin([1,2]).to_numpy()
    boundaries=np.flatnonzero((np.diff(t)>1.5/fs)|(valid[1:]!=valid[:-1]))+1
    return [(i+1,part.copy()) for i,part in enumerate([data.iloc[a:b] for a,b in zip(np.r_[0,boundaries],np.r_[boundaries,len(data)])]) if len(part)]


def gui_select(frame,config):
    scope=config['selection'];mask=np.ones(len(frame),bool)
    if scope['mode']==0:mask &= frame.Original_Label.isin([1,2]).to_numpy()
    if scope['mode']==1:
        if scope['end']<=scope['start']:raise ValueError('End phải lớn hơn Start.')
        t=frame.Time_ms.to_numpy()/1000
        mask &= (t>=scope['start'])&(t<=scope['end'])
    if scope['exclude_zero']:mask &= frame.Original_Label.isin([1,2]).to_numpy()
    return gui_split(frame.loc[mask],64.)


def gui_process(runs,config):
    engine=LegacyEngine(config);controls=config['controls'];fs=64.
    if controls['resample_checkbox']:fs=float(controls['target_fs_spinbox'])
    output=[]
    for sid,frame in runs:
        data=frame.copy()
        if controls['resample_checkbox']:data=engine.resample_dataframe(data,64.,fs)
        if controls['filter_checkbox']:data=engine.apply_filter_to_dataframe(data,fs)
        if controls['detrend_checkbox']:data=engine.apply_detrend_to_dataframe(data)
        data=gui_labels(data);output.append((sid,data))
    return output,fs


def gui_windows(runs,config,fs):
    engine=LegacyEngine(config);c=config['controls']
    size=max(1,int(round(c['window_length_spinbox']*fs)))
    if size<2:raise ValueError('Window cần ít nhất 2 mẫu để phân tích feature.')
    step=max(1,int(round(size*(1-c['window_overlap_spinbox']/100))))
    axes=[a for a,flag in zip(GUI_AXES,['windowing_axis_x_checkbox','windowing_axis_y_checkbox','windowing_axis_z_checkbox']) if c[flag]]
    if not axes:raise ValueError('Hãy chọn ít nhất một axis.')
    sensor=c['windowing_sensor_combo'];columns=[f'{sensor}_{a}_mg' for a in axes]
    items=[];meta=[];sid=0
    for _,frame in runs:
        # Never join separate runs; outside-experiment samples cannot reach ML.
        for _,part in gui_split(frame,fs,True):
            sid+=1;t=part.Time_ms.to_numpy(float)/1000;y=part.Binary_Label.to_numpy(float)
            for start in range(0,len(part)-size+1,step):
                end=start+size;wid=len(items)+1
                label,ratio=engine.assign_window_label(y[start:end])
                items.append({'Window_ID':wid,'Time':t[start:end].copy(),
                              'Signals':{col:part[col].to_numpy(float)[start:end].copy()/1000 for col in columns}})
                meta.append({'Window_ID':wid,'Segment_ID':sid,'Start_Index':start,'End_Index':end-1,
                             'Start_Time':float(t[start]),'End_Time':float(t[end-1]),'Samples':size,
                             'Binary_Label':int(label),'FOG_Ratio':ratio})
    if not items:raise ValueError('Không có đoạn hợp lệ đủ dài để tạo window.')
    return items,meta,size,step


def gui_feature_rows(items,meta,fs,config):
    # Reuse original feature/export logic; schema stays one window/axis per row.
    engine=LegacyEngine(config);engine.windowing_applied=True;engine.window_metadata=meta
    engine.window_signal_data=items;engine.windowing_sampling_rate=fs
    return engine._all_feature_rows()


def gui_export(table,config,prefix,format_index,report=None):
    prefix=Path(prefix)
    if prefix.suffix.lower() in ('.csv','.npz','.json'):prefix=prefix.with_suffix('')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    filenames=[]
    features=[name.replace(' ','_') for name in GUI_FEATURES]
    # Stage all outputs before replacing destination files.
    with tempfile.TemporaryDirectory(dir=prefix.parent) as tmp:
        staged=[]
        if format_index in (0,2):
            name=prefix.name+'.csv';table.to_csv(Path(tmp)/name,index=False);staged.append(name)
        if format_index in (0,1):
            name=prefix.name+'.npz'
            payload={'X':table[features].to_numpy(np.float32),'y':table.Binary_Label.to_numpy(np.int8),
                     'feature_names':np.asarray(features,dtype=str),'config_json':np.asarray(json.dumps(config,ensure_ascii=False))}
            for col in table.columns:
                if col in features or col=='Binary_Label':continue
                values=table[col]
                payload[col]=values.to_numpy(dtype=str) if values.dtype==object else values.to_numpy()
            np.savez_compressed(Path(tmp)/name,**payload);staged.append(name)
        name=prefix.name+'_config.json';(Path(tmp)/name).write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8');staged.append(name)
        if report is not None:
            name=prefix.name+'_report.csv';report.to_csv(Path(tmp)/name,index=False);staged.append(name)
        for name in staged:
            os.replace(Path(tmp)/name,prefix.parent/name);filenames.append(str(prefix.parent/name))
    return filenames

class GUIWorker(QObject):
    done=Signal(object);failed=Signal(str);progress=Signal(int,int,str)
    def __init__(self,task):super().__init__();self.task=task
    @Slot()
    def run(self):
        try:self.done.emit(self.task(self.progress.emit))
        except Exception as exc:self.failed.emit(str(exc))

class PassiveLabelView(pg.ViewBox):
    """Label overlay never captures drag, click, hover or wheel interaction."""

    def hoverEvent(self, event):
        pass

    def mouseDragEvent(self, event, axis=None):
        event.ignore()

    def mouseClickEvent(self, event):
        event.ignore()

    def wheelEvent(self, event, axis=None):
        event.ignore()


class SignalViewer(QWidget):
    """One chart, independent acceleration/label Y axes, shared absolute time."""
    regionSelected = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runs = []; self.curves = []; self.region = None; self._drawing = False
        self.times = np.array([]); self.values = []; self.source_name = 'No data'
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        controls = QGridLayout()
        controls.setHorizontalSpacing(8); controls.setVerticalSpacing(8)
        self.sensor = QComboBox(); self.sensor.addItems(GUI_SENSORS)
        self.labels = QComboBox(); self.labels.addItems(['Original 0/1/2','Binary 0/1','Hide labels'])
        self.scope = QComboBox(); self.scope.addItems(['All source samples','Experiment only'])
        self.checks = {}
        axes = QWidget(); axis_row = QHBoxLayout(axes); axis_row.setContentsMargins(0,0,0,0)
        for axis in GUI_AXES:
            box=QCheckBox(axis); box.setChecked(True); self.checks[axis]=box; axis_row.addWidget(box)
            box.toggled.connect(self.redraw)
        controls.addWidget(QLabel('Sensor'),0,0); controls.addWidget(self.sensor,0,1)
        controls.addWidget(QLabel('Label'),0,2); controls.addWidget(self.labels,0,3)
        controls.addWidget(QLabel('View'),0,4); controls.addWidget(self.scope,0,5)
        controls.addWidget(axes,0,6,1,2)
        self.start=QDoubleSpinBox(); self.end=QDoubleSpinBox()
        for spin in (self.start,self.end):
            spin.setDecimals(3); spin.setRange(0,1e9); spin.setSuffix(' s')
        controls.addWidget(QLabel('Start'),1,0); controls.addWidget(self.start,1,1)
        controls.addWidget(QLabel('End'),1,2); controls.addWidget(self.end,1,3)
        for col,text,callback in [(4,'VIEW RANGE',self.view_range),(5,'RESET VIEW',self.reset_view),
                                  (6,'SELECT REGION',self.select_region),(7,'USE REGION',self.use_region)]:
            button=QPushButton(text); button.clicked.connect(callback); controls.addWidget(button,1,col)
        for col in (1,3,5,6,7): controls.setColumnStretch(col,1)
        layout.addLayout(controls)
        self.chart=pg.PlotWidget(); self.chart.setBackground('w'); layout.addWidget(self.chart,1)
        self.chart.viewport().setCursor(Qt.OpenHandCursor)
        item=self.chart.getPlotItem(); item.showGrid(x=True,y=True,alpha=.15)
        item.setLabel('bottom','Absolute time',units='s'); item.setLabel('left','Acceleration',units='g')
        item.getAxis('bottom').enableAutoSIPrefix(False);item.getAxis('left').enableAutoSIPrefix(False)
        item.addLegend(offset=(12,12)); item.showAxis('right')
        self.label_view=PassiveLabelView(enableMenu=False); item.scene().addItem(self.label_view)
        item.getAxis('right').linkToView(self.label_view)
        self.label_view.setMouseEnabled(x=False,y=False)
        self.label_view.setAcceptedMouseButtons(Qt.NoButton)
        self.label_view.setZValue(5)

        self.label_curve=pg.PlotDataItem(pen=pg.mkPen('#c52d59',width=1.7),connect='finite')
        self.label_view.addItem(self.label_curve)
        item.vb.setMouseEnabled(x=True,y=False); 
        item.vb.enableAutoRange(x=False,y=False)
        item.vb.setMouseMode(pg.ViewBox.PanMode)
        item.vb.setMenuEnabled(False)
        item.vb.sigResized.connect(self.sync_geometry); 
        item.vb.sigXRangeChanged.connect(self.sync_geometry); 
        item.vb.sigXRangeChanged.connect(self.scale_y)
        self.info=QLabel('Open a recording to display data.'); self.info.setWordWrap(True);layout.addWidget(self.info)
        for control in (self.sensor,self.labels,self.scope):control.currentIndexChanged.connect(self.redraw)
        self.sync_geometry()

    def sync_geometry(self,*args):
        vb=self.chart.getViewBox(); self.label_view.setGeometry(vb.sceneBoundingRect())
        self.label_view.setXRange(*vb.viewRange()[0],padding=0)
        if not self._drawing and len(self.times):
            for control,value in zip((self.start,self.end),vb.viewRange()[0]):
                control.blockSignals(True);control.setValue(value);control.blockSignals(False)

    def set_runs(self,runs,name):
        self.runs=runs; self.source_name=name; self.redraw(reset=True)

    def clear_source(self,message):
        self.runs=[];self.source_name=message;self.redraw(reset=True)

    @staticmethod
    def stair_data(t,y):
        # Retain exact label transitions, without interpolating across missing labels.
        changes=np.r_[True, y[1:]!=y[:-1]]
        points=np.flatnonzero(changes)
        x=t[points]; v=y[points]
        if not len(x):return np.array([]),np.array([])
        end=t[-1]
        return np.repeat(np.r_[x,end],2)[1:-1],np.repeat(v,2)

    def redraw(self,*args,reset=False):
        self._drawing=True
        old_range=self.chart.viewRange()[0]
        self.times=np.array([]); self.values=[]
        for curve in self.curves:self.chart.removeItem(curve)
        self.curves=[]; self.chart.getPlotItem().legend.clear()

        if self.region is not None:self.chart.removeItem(self.region);self.region=None
        self.chart.viewport().setCursor(Qt.OpenHandCursor)
        self.label_curve.setData([],[])
        time_parts=[];signal_parts={a:[] for a in GUI_AXES};lx=[];ly=[]
        sensor=self.sensor.currentText(); binary=self.labels.currentIndex()==1
        for _,df in self.runs:
            if self.scope.currentIndex()==1:df=df.loc[df.Valid_Experiment]
            if df.empty:continue
            t=df.Time_ms.to_numpy(float)/1000
            time_parts.extend([t,np.array([np.nan])])
            for axis in GUI_AXES:signal_parts[axis].extend([df[f'{sensor}_{axis}_mg'].to_numpy(float)/1000,np.array([np.nan])])
            y=df.Binary_Label.to_numpy(float) if binary else df.Original_Label.to_numpy(float)
            sx,sy=self.stair_data(t,y);lx.extend([sx,np.array([np.nan])]);ly.extend([sy,np.array([np.nan])])
        if time_parts:
            self.times=np.concatenate(time_parts)
            for axis,color in zip(GUI_AXES,['#296acc','#d37521','#17896b']):
                if not self.checks[axis].isChecked():continue
                values=np.concatenate(signal_parts[axis]);self.values.append(values)
                curve=self.chart.plot(self.times, values, pen=pg.mkPen(color,width=1.1), name=axis, connect='finite')
                # Avoid peak decimation joining disjoint runs across NaN separators.
                curve.setDownsampling(auto=True, method='peak')
                curve.setClipToView(True);self.curves.append(curve)
            self.label_curve.setData(np.concatenate(lx),np.concatenate(ly),connect='finite')
        hidden=self.labels.currentIndex()==2;self.label_curve.setVisible(not hidden)
        self.chart.getPlotItem().getAxis('right').setVisible(not hidden)
        ticks=[(0,'0 · Non-FOG'),(1,'1 · FOG')] if binary else [(0,'0 · Outside'),(1,'1 · Non-FOG'),(2,'2 · FOG')]
        self.chart.getPlotItem().getAxis('right').setTicks([ticks])
        self.chart.getPlotItem().getAxis('right').setStyle(textFillLimits=[(0,1.0)])
        self.chart.getPlotItem().setLabel('right','Binary label' if binary else 'Original label',color='#c52d59')
        self.label_view.setYRange(-.12,1.12 if binary else 2.12,padding=0)
        self.chart.setTitle(f'{self.source_name} | {sensor}')
        finite=self.times[np.isfinite(self.times)]
        self._drawing=False
        if len(finite):
            low,high=float(finite[0]),float(finite[-1])
            for spin in (self.start,self.end):spin.setRange(low,high)
            self.start.setValue(low);self.end.setValue(high)
            if reset or old_range[1]<low or old_range[0]>high:self.reset_view()
            else:self.chart.setXRange(max(low,old_range[0]),min(high,old_range[1]),padding=0);self.scale_y()
            self.info.setText(f'{len(finite):,} samples | Left: g · Right: label (pink) | Gaps remain gaps. Zoom/pan does not modify data.')
        else:
            self.chart.setXRange(0,1,padding=0);self.chart.setYRange(-1,1,padding=0)
            self.info.setText('No data in this source. Apply Data Selection / Preprocessing first.')
        self.sync_geometry()

    def scale_y(self,*args):
        if self._drawing or not len(self.times):return
        lo,hi=self.chart.viewRange()[0];mask=(self.times>=lo)&(self.times<=hi)
        self.render_visible(lo,hi)
        arrays=[v[mask & np.isfinite(v)] for v in self.values];arrays=[v for v in arrays if len(v)]
        if arrays:
            a=min(v.min() for v in arrays);b=max(v.max() for v in arrays);pad=max(float(b-a),.02)*.08
            self.chart.setYRange(float(a-pad),float(b+pad),padding=0)

    def render_visible(self,low,high):
        # Peak-preserving screen rendering only. Full data remain available for analysis.
        visible=np.flatnonzero(np.isfinite(self.times)&(self.times>=low)&(self.times<=high))
        if not len(visible):
            for curve in self.curves:curve.setData([],[])
            return
        start=max(0,int(visible[0])-1);stop=min(len(self.times),int(visible[-1])+2)
        t=self.times[start:stop];budget=max(1000,int(self.chart.width())*3)
        stride=max(1,int(np.ceil(len(t)/budget)))
        for curve,full_values in zip(self.curves,self.values):
            y=full_values[start:stop];valid=np.flatnonzero(np.isfinite(t)&np.isfinite(y))
            bounds=np.r_[0,np.flatnonzero(np.diff(valid)>1)+1,len(valid)]
            tx=[];vy=[]
            for a,b in zip(bounds[:-1],bounds[1:]):
                idx=valid[a:b]
                if not len(idx):continue
                if stride>1 and len(idx)>stride:
                    count=len(idx)//stride
                    blocks=idx[:count*stride].reshape(count,stride)
                    val=y[blocks];rows=np.arange(count)
                    chosen=np.unique(np.r_[idx[0],blocks[rows,np.argmin(val,axis=1)],blocks[rows,np.argmax(val,axis=1)],idx[count*stride:],idx[-1]])
                else:chosen=idx
                tx.extend([t[chosen],np.array([np.nan])]);vy.extend([y[chosen],np.array([np.nan])])
            curve.setData(np.concatenate(tx) if tx else [],np.concatenate(vy) if vy else [],connect='finite')

    def reset_view(self):
        finite=self.times[np.isfinite(self.times)]
        if not len(finite):return
        a,b=float(finite[0]),float(finite[-1]);self.chart.setXRange(a,b if b>a else a+.02,padding=0)
        self.start.setValue(a);self.end.setValue(b);self.scale_y();self.sync_geometry()

    def view_range(self):
        a,b=self.start.value(),self.end.value()
        if b<=a:self.info.setText('End time must be greater than start time.');return
        self.chart.setXRange(a,b,padding=0);self.scale_y()

    def select_region(self):
        if not len(self.times):return
        if self.region is not None:self.chart.removeItem(self.region)
        a,b=self.chart.viewRange()[0];self.region=pg.LinearRegionItem([a+(b-a)*.25,a+(b-a)*.75])
        self.region.setZValue(10);self.chart.addItem(self.region)
        self.chart.viewport().setCursor(Qt.ArrowCursor)

    def use_region(self):
        if self.region is None:self.info.setText('Click SELECT REGION, then drag its boundaries.');return
        a,b=self.region.getRegion();self.regionSelected.emit(float(a),float(b))


class FOGSignalAnalyzer(CoreFOGAnalyzer):
    """GUI-only extension around the retained original application."""
    CONTROL_NAMES = (
        'resample_checkbox','target_fs_spinbox','filter_checkbox','filter_type_combo',
        'filter_order_spinbox','cutoff_1_spinbox','cutoff_2_spinbox','detrend_checkbox','detrend_type_combo',
        'windowing_sensor_combo','windowing_axis_x_checkbox','windowing_axis_y_checkbox',
        'windowing_axis_z_checkbox','window_length_spinbox','window_overlap_spinbox',
        'window_label_mode_combo','window_fog_threshold_spinbox')

    def __init__(self):
        if QApplication.instance() is not None:QApplication.instance().setStyle('Fusion')
        self.selected_runs=[];self.processed_runs=[];self.selection_applied=False
        self.applied_config=None;self.applied_selection_config=None;self.window_config=None
        self.batch_files=[];self._jobs=[];self._busy=False;self._building=True;self._donors=[]
        super().__init__()
        self._building=False;self._connect_changes();self._align_ui()
        self.toggle_resample_controls();self.toggle_filter_controls();self.toggle_detrend_controls()
        self.setWindowTitle('FOG Signal Analyzer — Original Core + GUI Update')
        self.resize(1365,720);self.setMinimumSize(1200,680)

    def _heading_page(self,title):
        page=QWidget();layout=QVBoxLayout(page);layout.setContentsMargins(6,4,4,4);layout.setSpacing(8)
        label=QLabel(title);label.setStyleSheet('font-size:21px;font-weight:700;');layout.addWidget(label)
        return page,layout

    def _button(self,text,slot):
        button=QPushButton(text);button.clicked.connect(slot);return button

    def _note(self,text):
        label=QLabel(text);label.setWordWrap(True);return label

    def _table(self):
        table=QTableView();model=DataFrameTableModel(parent=table);table.setModel(model)
        table.setAlternatingRowColors(True);table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows);table.verticalHeader().setDefaultSectionSize(25)
        table.horizontalHeader().setDefaultSectionSize(145);return table,model

    def create_ui(self):
        self._apply_app_style();central=QWidget();self.setCentralWidget(central)
        root=QVBoxLayout(central);root.setContentsMargins(10,8,10,8);root.setSpacing(8)
        top=QHBoxLayout();title=QLabel('FOG SIGNAL ANALYZER');title.setStyleSheet('font-size:22px;font-weight:700')
        top.addWidget(title);top.addStretch();self.open_button=self._button('OPEN DATASET',self.open_dataset)
        top.addWidget(self.open_button);self.path_label=QLabel('No dataset selected')
        self.path_label.setMaximumWidth(310);self.path_label.setMinimumWidth(180)
        self.path_label.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred);top.addWidget(self.path_label)
        root.addLayout(top);body=QHBoxLayout();self.navigation=QListWidget();self.navigation.setFixedWidth(205)
        self.pages=QStackedWidget();body.addWidget(self.navigation);body.addWidget(self.pages,1);root.addLayout(body,1)
        self.dataset_page=self.create_dataset_page_redesigned()
        self.batch_page=self._create_batch_page()
        self.time_domain_page=self._create_raw_page()
        self.data_selection_page=self._create_selection_page()
        self.selected_time_page=self._create_selected_page()
        self.preprocessing_page=self._create_processing_page()
        self.windowing_page=self._create_window_page()
        self.time_analysis_page=self.create_time_analysis_page()
        self.frequency_analysis_page=self.create_frequency_analysis_page()
        self.publish_page=self._create_export_page();self.analysis_page=self.time_analysis_page
        self._page_list=[self.batch_page,self.dataset_page,self.time_domain_page,self.data_selection_page,
                         self.selected_time_page,self.preprocessing_page,self.windowing_page,
                         self.time_analysis_page,self.frequency_analysis_page,self.publish_page]
        self.navigation.addItems(['Batch Preparation','1. Dataset','2. Raw Time Domain','3. Data Selection',
                                  '4. Selected Time Domain','5. Preprocessing','6. Windowing',
                                  '7. Time Analysis','8. Frequency Analysis','9. Export'])
        for page in self._page_list:self.pages.addWidget(page)
        self.navigation.currentRowChanged.connect(self._navigate)
        self.job_progress=QProgressBar();self.job_progress.hide();root.addWidget(self.job_progress)
        self._go(self.dataset_page)
        self.setStyleSheet(self.styleSheet()+'''
            QListWidget{border:1px solid #c5cdd9;font-size:13px;background:#f8faff;}
            QListWidget::item{padding:10px 7px;}
            QListWidget::item:selected{background:#416fba;color:white;}
            QPushButton{min-height:24px;max-height:24px;padding:4px 10px;}
            QGroupBox{border:1px solid #c5cdd9;margin-top:10px;padding:10px;}
            QComboBox,QDoubleSpinBox,QSpinBox{min-height:26px;max-height:26px;padding:2px 6px;}
        ''')
        self.statusBar().showMessage('Raw giữ nguyên · Các hàm xử lý và công thức feature gốc được giữ lại.')

    def _align_ui(self):
        for kind in (QPushButton,QComboBox,QSpinBox,QDoubleSpinBox):
            for control in self.findChildren(kind):
                control.setMinimumWidth(0);control.setFixedHeight(34)
                control.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.open_button.setFixedWidth(155)
        for group in self.findChildren(QGroupBox):
            grid=group.layout()
            if isinstance(grid,QGridLayout):
                grid.setHorizontalSpacing(10);grid.setVerticalSpacing(8)
                grid.setColumnMinimumWidth(0,100);grid.setColumnStretch(1,1)
                grid.setRowStretch(grid.rowCount(),1)
        for page in (self.time_analysis_page,self.frequency_analysis_page):
            page.layout().setContentsMargins(6,4,4,4)

    def _go(self,page):
        self.pages.setCurrentWidget(page)
        index=self._page_list.index(page)
        self.navigation.blockSignals(True);self.navigation.setCurrentRow(index);self.navigation.blockSignals(False)

    def _navigate(self,index):
        actions=[self.show_batch,self.show_dataset_page,self.show_time_domain,self.show_data_selection,
                 self.show_selected_time,self.show_preprocessing,self.show_windowing,self.show_analysis,
                 self.show_frequency_analysis,self.show_publish_page]
        if 0<=index<len(actions):actions[index]()

    def _set_nav_active(self,index):
        # Original analysis methods still call this API with their original indices.
        legacy=[self.dataset_page,self.time_domain_page,self.preprocessing_page,self.windowing_page,
                self.time_analysis_page,self.frequency_analysis_page,self.publish_page]
        if 0<=index<len(legacy):self._go(legacy[index])

    def set_label_mapping_enabled(self,checked=True):
        # Mapping is available from load; Original/Binary is only a display option.
        if self.df_raw is not None:
            self.df=gui_labels(self.df_raw);self.show_preview()

    def show_preview(self):
        if self.df is not None:
            base_cols = [c for c in self.df.columns if c not in ['Label', 'Original_Label', 'Binary_Label', 'Valid_Experiment']]          
            display_order= base_cols + ['Original_Label', 'Binary_Label', 'Valid_Experiment']           
            self.preview_model.set_dataframe(self.df[display_order])
            self.preview_table.horizontalHeader().setDefaultSectionSize(135)
            self.preview_status.setText(f'{len(self.df):,} rows · Original + Binary · — = outside experiment')

    def _create_raw_page(self):
        page,layout=self._heading_page('RAW TIME DOMAIN')
        layout.addWidget(self._note('Tín hiệu g ở trục trái · Nhãn ở trục phải · Cùng timestamp và cùng vùng zoom/pan.'))
        self.raw_viewer=SignalViewer();self.raw_viewer.regionSelected.connect(self._accept_region)
        layout.addWidget(self.raw_viewer,1)
        layout.addWidget(self._button('NEXT → DATA SELECTION',self.show_data_selection));return page

    def _create_selection_page(self):
        page,layout=self._heading_page('DATA SELECTION');row=QHBoxLayout()
        scope=QGroupBox('SCOPE');grid=QGridLayout(scope)
        self.selection_mode=QComboBox();self.selection_mode.addItems(['Chỉ đoạn trong thí nghiệm','Khoảng thời gian chọn','Toàn bộ bản ghi'])
        self.selection_start=QDoubleSpinBox();self.selection_end=QDoubleSpinBox()
        for spin in (self.selection_start,self.selection_end):
            spin.setRange(0,1e9);spin.setDecimals(3);spin.setSuffix(' s')
        self.selection_end.setValue(1)
        for r,title,control in [(0,'Phạm vi',self.selection_mode),(1,'Start',self.selection_start),(2,'End',self.selection_end)]:
            grid.addWidget(QLabel(title),r,0);grid.addWidget(control,r,1)
        row.addWidget(scope,1);options=QGroupBox('OPTIONS');ol=QVBoxLayout(options)
        self.selection_exclude=QCheckBox('Loại Original Label 0 trước preprocessing');self.selection_exclude.setChecked(True)
        ol.addWidget(self.selection_exclude);ol.addWidget(self._note('Giữ timestamp gốc. Mỗi đoạn liên tục được xử lý riêng. Window ML luôn loại Original Label 0.'))
        buttons=QHBoxLayout();buttons.addWidget(self._button('APPLY SELECTION',self.apply_data_selection));buttons.addWidget(self._button('RESET',self.reset_data_selection))
        ol.addLayout(buttons);row.addWidget(options,1);layout.addLayout(row)
        self.selection_message=self._note('Chưa mở dữ liệu.');layout.addWidget(self.selection_message)
        self.selection_table,self.selection_model=self._table();layout.addWidget(self.selection_table,1)
        layout.addWidget(self._button('VIEW SELECTION → TIME DOMAIN',self.show_selected_time))
        return page

    def _create_selected_page(self):
        page,layout=self._heading_page('TIME DOMAIN AFTER DATA SELECTION');row=QHBoxLayout()
        row.addWidget(QLabel('Source'));self.selected_source=QComboBox()
        self.selected_source.addItems(['Selected — trước preprocessing','Processed — sau preprocessing'])
        self.selected_source.currentIndexChanged.connect(self._refresh_selected)
        row.addWidget(self.selected_source,1);row.addWidget(self._button('EDIT SELECTION',self.show_data_selection));layout.addLayout(row)
        self.selected_viewer=SignalViewer();self.selected_viewer.regionSelected.connect(self._accept_region)
        layout.addWidget(self.selected_viewer,1);layout.addWidget(self._button('NEXT → PREPROCESSING',self.show_preprocessing))
        return page

    def _donor(self,page):
        page.setParent(self);page.hide();self._donors.append(page)
        return {g.title():g for g in page.findChildren(QGroupBox)}

    def _create_processing_page(self):
        donor=CoreFOGAnalyzer.create_preprocessing_page(self);groups=self._donor(donor)
        page,layout=self._heading_page('PREPROCESSING');self.processing_input=self._note('Input: APPLY DATA SELECTION trước.')
        layout.addWidget(self.processing_input);row=QHBoxLayout();row.setSpacing(12)
        for name in ('RESAMPLE','FILTER','DETRENDING'):row.addWidget(groups[name],1)
        layout.addLayout(row)
        layout.addWidget(self._note('Resample → Filter → Detrend trên từng đoạn. Bộ lọc dùng đúng hàm gốc: filtfilt; đoạn ngắn dùng lfilter như bản gốc.'))
        actions=QHBoxLayout()
        for text,slot in [('RESET',self.reset_preprocessing),('APPLY PREPROCESSING',self.apply_preprocessing),
                          ('VIEW PROCESSED',self.view_processed_time_domain),('NEXT → WINDOWING',self.show_windowing)]:
            actions.addWidget(self._button(text,slot))
        layout.addLayout(actions);self.processing_message=self._note('Chưa xử lý.');layout.addWidget(self.processing_message)
        self.processing_table,self.processing_model=self._table();layout.addWidget(self.processing_table,1)
        return page

    def _create_window_page(self):
        donor=CoreFOGAnalyzer.create_windowing_page(self);groups=self._donor(donor)
        page,layout=self._heading_page('WINDOWING');self.window_input_note=self._note('Input: Selected hoặc Processed · Original Label 0 luôn bị loại.')
        layout.addWidget(self.window_input_note);row=QHBoxLayout();row.setSpacing(12)
        for name in ('SIGNAL','WINDOW PARAMETERS','LABEL ASSIGNMENT'):row.addWidget(groups[name],1)
        layout.addLayout(row);actions=QHBoxLayout()
        for text,slot in [('RESET WINDOWS',self.reset_windowing),('APPLY WINDOWING',self.apply_windowing),
                          ('TIME ANALYSIS',self.show_analysis),('FREQUENCY ANALYSIS',self.show_frequency_analysis)]:
            actions.addWidget(self._button(text,slot))
        layout.addLayout(actions);layout.addWidget(self.windowing_status_message)
        self.window_table,self.window_model=self._table();layout.addWidget(self.window_table,1);return page

    def _create_export_page(self):
        page,layout=self._heading_page('EXPORT FEATURES');row=QHBoxLayout()
        self.export_format=QComboBox();self.export_format.addItems(['CSV + NPZ','NPZ','CSV'])
        row.addWidget(QLabel('Format'));row.addWidget(self.export_format,1)
        row.addWidget(self._button('EXPORT CURRENT WINDOWS',self.publish_features_csv))
        row.addWidget(self._button('SAVE CONFIG JSON',self.save_config));layout.addLayout(row)
        self.publish_info=self._note('Áp dụng Windowing trước khi xuất feature.');layout.addWidget(self.publish_info)
        self.export_table,self.export_model=self._table();layout.addWidget(self.export_table,1)
        layout.addWidget(self._note('Giữ schema gốc: mỗi hàng = một window / sensor / axis. JSON lưu thông số; không chứa dữ liệu tín hiệu.'))
        layout.addWidget(self._button('OPEN BATCH PREPARATION',self.show_batch));return page

    def _create_batch_page(self):
        page,layout=self._heading_page('BATCH PREPARATION')
        layout.addWidget(self._note('Thêm các file raw → kiểm tra thông số chung → xuất feature CSV/NPZ.'))
        row=QHBoxLayout()
        for text,slot in [('ADD FILES',self._add_files),('ADD FOLDER',self._add_folder),('REMOVE SELECTED',self._remove_files),('CLEAR LIST',self._clear_files)]:
            row.addWidget(self._button(text,slot))
        layout.addLayout(row);self.batch_list=QListWidget();self.batch_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.batch_list,1);self.batch_summary=self._note('Thông số mặc định.');layout.addWidget(self.batch_summary)
        row=QHBoxLayout()
        for text,slot in [('DATA SELECTION',self.show_data_selection),('PREPROCESSING',self.show_preprocessing),('WINDOWING',self.show_windowing),('LOAD CONFIG JSON',self.load_config)]:
            row.addWidget(self._button(text,slot))
        layout.addLayout(row);row=QHBoxLayout();row.addWidget(QLabel('Format'))
        self.batch_format=QComboBox();self.batch_format.addItems(['CSV + NPZ','NPZ','CSV']);row.addWidget(self.batch_format,1)
        row.addWidget(self._button('PREPARE BATCH',self.prepare_batch));layout.addLayout(row)
        self.batch_message=self._note('Chưa chọn file.');layout.addWidget(self.batch_message)
        self.batch_report,self.batch_report_model=self._table();layout.addWidget(self.batch_report,1)
        return page

    def capture_config(self):
        controls={}
        for name in self.CONTROL_NAMES:
            widget=getattr(self,name)
            controls[name]=widget.isChecked() if isinstance(widget,QCheckBox) else widget.currentText() if isinstance(widget,QComboBox) else widget.value()
        config={'schema':'original-core-gui-v1','source_fs':64.,'feature_schema':'one-window-sensor-axis-per-row',
                'selection':{'mode':self.selection_mode.currentIndex(),'start':self.selection_start.value(),
                             'end':self.selection_end.value(),'exclude_zero':self.selection_exclude.isChecked()},
                'controls':controls,'filter_implementation':'original filtfilt; lfilter fallback on short segments',
                'label_mapping':{'0':None,'1':0,'2':1},'processing_order':['selection','resample','filter','detrend','windowing']}
        config['config_id']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
        return config

    def _connect_changes(self):
        self.selection_mode.currentIndexChanged.connect(self._selection_mode_changed)
        for control in (self.selection_start,self.selection_end):control.valueChanged.connect(self._selection_changed)
        self.selection_exclude.toggled.connect(self._selection_changed)
        for name in self.CONTROL_NAMES:
            control=getattr(self,name)
            signal_=control.toggled if isinstance(control,QCheckBox) else control.currentIndexChanged if isinstance(control,QComboBox) else control.valueChanged
            signal_.connect(self._processing_changed if name in self.CONTROL_NAMES[:9] else self._window_changed)
        self._selection_mode_changed()

    def _selection_mode_changed(self,*args):
        idx=self.selection_mode.currentIndex()
        self.selection_start.setEnabled(idx==1);self.selection_end.setEnabled(idx==1)
        self.selection_exclude.setEnabled(idx!=0)
        self.selection_exclude.blockSignals(True);self.selection_exclude.setChecked(idx!=2);self.selection_exclude.blockSignals(False)
        self._selection_changed()

    def _selection_changed(self,*args):
        if self._building:return
        self.selected_runs=[];self.selection_applied=False;self.applied_selection_config=None
        self._invalidate_processing();self._refresh_selected()
        self.processing_input.setText('Input: lựa chọn đã đổi — APPLY DATA SELECTION.')
        if self.df_raw is not None:self._selection_preview()

    def _invalidate_windows(self):
        CoreFOGAnalyzer.clear_windowing_result(self);self.window_config=None
        self.window_model.set_dataframe(pd.DataFrame());self.export_model.set_dataframe(pd.DataFrame())
        for combo in (self.analysis_window_combo,self.frequency_window_combo):
            combo.blockSignals(True);combo.clear();combo.blockSignals(False)
        self.reset_analysis();self.analysis_window_id=None
        self.publish_info.setText('Áp dụng Windowing trước khi xuất feature.')

    def _invalidate_processing(self):
        self.processed_runs=[];self.applied_config=None;self.preprocessing_applied=False
        self.df_processed=None;self.df_selected=None;self.processed_sampling_rate=None
        self._invalidate_windows();self.processing_model.set_dataframe(pd.DataFrame())
        self.processing_message.setText('Chưa xử lý / thông số đã đổi.')
        self._refresh_selected()

    def _processing_changed(self,*args):
        if not self._building:self._invalidate_processing()

    def _window_changed(self,*args):
        if not self._building:self._invalidate_windows()

    def _selection_preview(self):
        try:
            runs=gui_select(gui_labels(self.df_raw),self.capture_config())
            rows=[{'Segment':sid,'Start_s':d.Time_ms.iloc[0]/1000,'End_s':d.Time_ms.iloc[-1]/1000,
                   'Samples':len(d),'Original_0':int((d.Original_Label==0).sum())} for sid,d in runs]
            self.selection_model.set_dataframe(pd.DataFrame(rows));kept=sum(len(d) for _,d in runs)
            state='APPLIED' if self.selection_applied else 'PREVIEW — chưa áp dụng'
            self.selection_message.setText(f'{state} | {kept:,} mẫu | {len(runs)} đoạn | loại {len(self.df_raw)-kept:,} mẫu')
        except Exception as exc:self.selection_message.setText(str(exc));self.selection_model.set_dataframe(pd.DataFrame())

    def apply_data_selection(self):
        if self.df_raw is None:self._error('Hãy mở dataset trước.');return
        try:
            config=self.capture_config();runs=gui_select(gui_labels(self.df_raw),config)
            if not runs:raise ValueError('Lựa chọn không có dữ liệu.')
            self._invalidate_processing();self.selected_runs=runs;self.selection_applied=True
            self.applied_selection_config=config;self._selection_preview();self._refresh_selected()
            self.processing_input.setText(f'Input: {sum(len(d) for _,d in runs):,} mẫu · {len(runs)} đoạn · 64 Hz')
        except Exception as exc:self._error(exc)

    def reset_data_selection(self):
        self.selection_mode.setCurrentIndex(0);self.selection_exclude.setChecked(True)
        if self.df_raw is not None:
            self.selection_start.setValue(self.df_raw.Time_ms.iloc[0]/1000);self.selection_end.setValue(self.df_raw.Time_ms.iloc[-1]/1000)
        self._selection_changed()

    def _accept_region(self,start,end):
        self.selection_mode.setCurrentIndex(1);self.selection_start.setValue(start);self.selection_end.setValue(end)
        self.show_data_selection()

    def _refresh_selected(self,*args):
        if not hasattr(self,'selected_viewer'):return
        processed=self.selected_source.currentIndex()==1
        self.selected_viewer.set_runs(self.processed_runs if processed else self.selected_runs,
                                      ('PROCESSED' if processed else 'SELECTED')+' · '+os.path.basename(self.file_path or 'No file'))

    def _error(self,error):QMessageBox.critical(self,'FOG Signal Analyzer',str(error))

    def _start_job(self,task,callback):
        if self._busy:return
        self._busy=True;self._job_callback=callback
        thread=QThread(self);worker=GUIWorker(task);worker.moveToThread(thread)
        thread.started.connect(worker.run);worker.done.connect(self._job_done);worker.failed.connect(self._job_failed)
        worker.progress.connect(self._job_progress);worker.done.connect(thread.quit);worker.failed.connect(thread.quit)
        worker.done.connect(worker.deleteLater);worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._job_finished);thread.finished.connect(thread.deleteLater)
        self._jobs=[(thread,worker)];self.pages.setEnabled(False);self.navigation.setEnabled(False);self.open_button.setEnabled(False)
        self.job_progress.setRange(0,0);self.job_progress.show();thread.start()

    @Slot(object)
    def _job_done(self,result):
        try:self._job_callback(result)
        except Exception as exc:self._error(exc)

    @Slot(str)
    def _job_failed(self,error):
        self.batch_message.setText('Thao tác thất bại: '+error);self._error(error)

    @Slot(int,int,str)
    def _job_progress(self,i,n,name):
        self.job_progress.setRange(0,n);self.job_progress.setValue(i);self.statusBar().showMessage(name)

    @Slot()
    def _job_finished(self):
        self._jobs=[];self._busy=False;self.job_progress.hide()
        self.pages.setEnabled(True);self.navigation.setEnabled(True);self.open_button.setEnabled(True)

    def closeEvent(self,event):
        if self._busy:self.statusBar().showMessage('Đợi tác vụ hoàn tất trước khi đóng ứng dụng.');event.ignore()
        else:super().closeEvent(event)

    def open_dataset(self):
        path,_=QFileDialog.getOpenFileName(self,'Open DAPHNET raw','','Raw dataset (*.txt *.csv)')
        if path:self._start_job(lambda progress:gui_read_raw(path),lambda frame:self._install_raw(frame,path))

    def _install_raw(self,frame,path):
        self.df_raw=frame.copy();self.df=gui_labels(frame);self.file_path=str(path);self.sampling_rate=64.
        self.display_df=self.df.copy();self.display_sampling_rate=64.;self.display_processed_label=False
        self.selected_runs=[];self.selection_applied=False;self._invalidate_processing()
        self.path_label.setText(Path(path).name);self.path_label.setToolTip(str(path))
        self.file_value.setText(Path(path).name);self.file_value.setToolTip(str(path))
        self.columns_value.setText('11 raw + label metadata');self.samples_value.setText(f'{len(frame):,}')
        self.fs_value.setText('64 Hz');self.duration=(frame.Time_ms.iloc[-1]-frame.Time_ms.iloc[0])/1000
        self.duration_value.setText(f'{self.duration:.3f} sec');self.show_preview()
        self.label_table.setRowCount(3)
        for r,row in enumerate([(0,'Outside experiment','#N/A'),(1,'Non-FOG',0),(2,'FOG',1)]):
            for col,value in enumerate(row):self.label_table.setItem(r,col,QTableWidgetItem(str(value)))
        for spin,value in [(self.selection_start,frame.Time_ms.iloc[0]/1000),(self.selection_end,frame.Time_ms.iloc[-1]/1000)]:
            spin.blockSignals(True);spin.setRange(frame.Time_ms.iloc[0]/1000,frame.Time_ms.iloc[-1]/1000);spin.setValue(value);spin.blockSignals(False)
        self.selection_mode.setCurrentIndex(0);self.selection_exclude.setChecked(True)
        self.apply_data_selection();self.raw_viewer.set_runs(gui_split(self.df),f'RAW · {Path(path).stem}')
        self.preprocessing_original_fs_label.setText('64 Hz');self._go(self.dataset_page)

    def show_dataset_page(self):self._go(self.dataset_page)
    def show_time_domain(self):self._go(self.time_domain_page)
    def show_data_selection(self):
        if self.df_raw is not None:self._selection_preview()
        self._go(self.data_selection_page)
    def show_selected_time(self):self._go(self.selected_time_page)
    def show_preprocessing(self):self._go(self.preprocessing_page)
    def show_time_domain_from_preprocessing(self):self.show_selected_time()
    def show_preprocessing_from_windowing(self):self.show_preprocessing()
    def show_windowing_from_analysis(self):self.show_windowing()

    def apply_preprocessing(self):
        if not self.selection_applied:self._error('Hãy APPLY DATA SELECTION trước.');return
        config=self.capture_config();runs=[(sid,frame.copy()) for sid,frame in self.selected_runs]
        self._start_job(lambda progress:gui_process(runs,config),lambda result:self._install_processed(result,config))

    def _install_processed(self,result,config):
        runs,fs=result;self._invalidate_windows();self.processed_runs=runs;self.applied_config=config
        self.df_processed=pd.concat([d for _,d in runs],ignore_index=True);self.df_selected=self.df_processed.copy()
        self.processed_sampling_rate=fs;self.preprocessing_applied=True
        self.processed_start_time=self.df_processed.Time_ms.iloc[0]/1000;self.processed_end_time=self.df_processed.Time_ms.iloc[-1]/1000
        self.processing_model.set_dataframe(self.df_processed)
        self.processing_message.setText(f'APPLIED · {len(self.df_processed):,} mẫu · {len(runs)} đoạn · {fs:g} Hz · hàm xử lý gốc')
        self._refresh_selected()

    def reset_preprocessing(self):
        self.resample_checkbox.setChecked(False);self.filter_checkbox.setChecked(False);self.detrend_checkbox.setChecked(False)
        self.target_fs_spinbox.setValue(64);self._invalidate_processing()

    def view_processed_time_domain(self):
        if not self.preprocessing_applied:self._error('Hãy APPLY PREPROCESSING trước.');return
        self.selected_source.setCurrentIndex(1);self._refresh_selected();self.show_selected_time()

    def _window_input(self):
        return (self.processed_runs,self.processed_sampling_rate) if self.preprocessing_applied else (self.selected_runs,64.)

    def show_windowing(self):
        runs,fs=self._window_input();self.windowing_sampling_rate=fs
        self.windowing_input_df=pd.concat([d for _,d in runs],ignore_index=True) if runs else None
        self.update_window_parameter_calculation()
        self.window_input_note.setText(f'Input: {"Processed" if self.preprocessing_applied else "Selected"} · {sum(len(d) for _,d in runs):,} mẫu · {fs:g} Hz · window luôn loại nhãn 0')
        self._go(self.windowing_page)

    def update_window_parameter_calculation(self):
        CoreFOGAnalyzer.update_window_parameter_calculation(self)
        if hasattr(self,'selected_runs') and self.windowing_sampling_rate:
            runs,fs=self._window_input();n=0
            for _,frame in runs:
                for _,part in gui_split(frame,fs,True):
                    n+=max(0,(len(part)-self.window_samples)//self.window_step_samples+1)
            self.window_count_calculated_value.setText(f'{n:,}')

    def apply_windowing(self):
        runs,fs=self._window_input()
        if not runs:self._error('Hãy APPLY DATA SELECTION trước.');return
        config=self.capture_config()
        if self.applied_config:
            for name in self.CONTROL_NAMES[:9]:config["controls"][name]=self.applied_config["controls"][name]
        self._start_job(lambda progress:gui_windows(runs,config,fs),lambda result:self._install_windows(result,config,fs))

    def _install_windows(self,result,config,fs):
        config=dict(config);config.pop("config_id",None)
        config["config_id"]=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
        items,meta,size,step=result;self.window_signal_data=items;self.window_metadata=meta
        self.windowing_sampling_rate=fs;self.windowing_applied=True;self.window_config=config
        self.window_model.set_dataframe(pd.DataFrame(meta));self.update_analysis_window_list()
        self.windowing_status_message.setText(f'APPLIED · {len(meta):,} windows · {size} samples/window · step {step} · không vượt đoạn hoặc nhãn 0')
        self.statusBar().showMessage(self.windowing_status_message.text())

    def reset_windowing(self):self._invalidate_windows()

    def show_analysis(self):
        if not self.windowing_applied:self._error('Hãy APPLY WINDOWING trước.');return
        self._go(self.time_analysis_page)
    def show_frequency_analysis(self):
        if not self.windowing_applied:self._error('Hãy APPLY WINDOWING trước.');return
        self._go(self.frequency_analysis_page)
    def show_publish_page(self):
        self._go(self.publish_page);self.update_publish_preview()

    def update_publish_preview(self):
        if self.pages.currentWidget() is not self.publish_page or not self.windowing_applied:return
        config=self.window_config;items=self.window_signal_data;meta=self.window_metadata;fs=self.windowing_sampling_rate
        self._start_job(lambda progress:gui_feature_rows(items,meta,fs,config),self._install_export_preview)

    def _install_export_preview(self,table):
        self.export_model.set_dataframe(table)
        self.publish_info.setText(f'{len(table):,} feature rows · schema gốc: window / sensor / axis · 14 features mỗi hàng')

    def publish_features_csv(self):
        if not self.windowing_applied:self._error('Hãy APPLY WINDOWING trước.');return
        prefix,_=QFileDialog.getSaveFileName(self,'Export feature dataset',Path(self.file_path or 'dataset').stem+'_features','Dataset (*.csv *.npz)')
        if not prefix:return
        config=self.window_config;items=self.window_signal_data;meta=self.window_metadata;fs=self.windowing_sampling_rate;fmt=self.export_format.currentIndex()
        def task(progress):
            table=gui_feature_rows(items,meta,fs,config);return gui_export(table,config,prefix,fmt)
        self._start_job(task,lambda files:self.publish_info.setText('Đã xuất: '+' | '.join(files)))

    def save_config(self):
        path,_=QFileDialog.getSaveFileName(self,'Save current configuration','fog_config.json','JSON (*.json)')
        if not path:return
        try:
            path=str(Path(path).with_suffix('.json'));Path(path).write_text(json.dumps(self.capture_config(),ensure_ascii=False,indent=2),encoding='utf-8')
            self.statusBar().showMessage('Đã lưu cấu hình: '+path)
        except Exception as exc:self._error(exc)

    def load_config(self):
        path,_=QFileDialog.getOpenFileName(self,'Load configuration','','JSON (*.json)')
        if not path:return
        try:
            config=json.loads(Path(path).read_text(encoding='utf-8'))
            if config.get('schema')!='original-core-gui-v1':raise ValueError('JSON không thuộc phiên bản GUI dùng core gốc này.')
            # Validate all values first; no silent clamping by spinboxes.
            for name in self.CONTROL_NAMES:
                control=getattr(self,name);value=config['controls'][name]
                if isinstance(control,QCheckBox):
                    if not isinstance(value,bool):raise ValueError(name+' cần true/false.')
                elif isinstance(control,QComboBox):
                    if control.findText(value)<0:raise ValueError('Giá trị không hợp lệ: '+name)
                elif not control.minimum()<=float(value)<=control.maximum():raise ValueError('Giá trị ngoài phạm vi: '+name)
            scope=config['selection'];idx=int(scope['mode'])
            if idx not in (0,1,2) or scope['end']<=scope['start']:raise ValueError('Data Selection không hợp lệ.')
            if idx==1 and self.df_raw is not None:
                if scope['start']<self.selection_start.minimum() or scope['end']>self.selection_end.maximum():raise ValueError('Khoảng thời gian JSON nằm ngoài bản ghi đang mở.')
            controls=[getattr(self,n) for n in self.CONTROL_NAMES]+[self.selection_mode,self.selection_start,self.selection_end,self.selection_exclude]
            for control in controls:control.blockSignals(True)
            try:
                for name in self.CONTROL_NAMES:
                    control=getattr(self,name);value=config['controls'][name]
                    if isinstance(control,QCheckBox):control.setChecked(value)
                    elif isinstance(control,QComboBox):control.setCurrentText(value)
                    else:control.setValue(value)
                self.selection_mode.setCurrentIndex(idx);self.selection_start.setValue(scope['start']);self.selection_end.setValue(scope['end'])
                self.selection_exclude.setChecked(scope['exclude_zero'] if idx else True)
            finally:
                for control in controls:control.blockSignals(False)
            self.selection_start.setEnabled(idx==1);self.selection_end.setEnabled(idx==1);self.selection_exclude.setEnabled(idx!=0)
            self.toggle_resample_controls();self.toggle_filter_controls();self.toggle_detrend_controls();self.update_window_label_controls()
            self._selection_changed();self.show_batch()
        except Exception as exc:self._error(exc)

    def show_batch(self):
        c=self.capture_config();v=c['controls']
        self.batch_summary.setText(f'CONFIG · Scope {self.selection_mode.currentText()} · Filter {v["filter_type_combo"] if v["filter_checkbox"] else "Off"} · Fs {v["target_fs_spinbox"] if v["resample_checkbox"] else 64:g} Hz · {v["windowing_sensor_combo"]} · Window {v["window_length_spinbox"]:g} s · Overlap {v["window_overlap_spinbox"]:g}%')
        self._go(self.batch_page)

    def _sync_batch_list(self):
        self.batch_list.clear();self.batch_list.addItems(self.batch_files);self.batch_message.setText(f'{len(self.batch_files)} files selected.')
    def _add_files(self):
        paths,_=QFileDialog.getOpenFileNames(self,'Add raw recordings','','Raw (*.txt *.csv)')
        self.batch_files=list(dict.fromkeys(self.batch_files+[str(Path(p).resolve()) for p in paths]));self._sync_batch_list()
    def _add_folder(self):
        path=QFileDialog.getExistingDirectory(self,'Add raw folder')
        if path:
            files=[str(p.resolve()) for p in sorted(Path(path).iterdir()) if p.suffix.lower() in ('.txt','.csv')]
            self.batch_files=list(dict.fromkeys(self.batch_files+files));self._sync_batch_list()
    def _remove_files(self):
        remove={item.text() for item in self.batch_list.selectedItems()};self.batch_files=[p for p in self.batch_files if p not in remove];self._sync_batch_list()
    def _clear_files(self):self.batch_files=[];self._sync_batch_list()

    def prepare_batch(self):
        if not self.batch_files:self._error('Hãy thêm file raw cho batch.');return
        prefix,_=QFileDialog.getSaveFileName(self,'Export batch dataset','fog_batch','Dataset (*.csv *.npz)')
        if not prefix:return
        paths=self.batch_files.copy();config=self.capture_config();fmt=self.batch_format.currentIndex()
        def task(progress):
            tables=[];reports=[];seen=set()
            for i,path in enumerate(paths,1):
                name=Path(path).stem
                try:
                    if name in seen:raise ValueError('Recording name trùng trong batch.')
                    seen.add(name);raw=gui_labels(gui_read_raw(path));runs=gui_select(raw,config)
                    processed,fs=gui_process(runs,config);items,meta,_,_=gui_windows(processed,config,fs)
                    table=gui_feature_rows(items,meta,fs,config)
                    subject=re.match(r'(S\d+)',name,re.I)
                    table.insert(0,'Recording',name);table.insert(0,'Subject',subject.group(1).upper() if subject else 'unknown')
                    tables.append(table);reports.append({'File':str(path),'Status':'ok','Windows':len(meta),'Feature_Rows':len(table),'Error':''})
                except Exception as exc:reports.append({'File':str(path),'Status':'error','Windows':0,'Feature_Rows':0,'Error':str(exc)})
                progress(i,len(paths),name)
            report=pd.DataFrame(reports)
            if not tables:
                path=Path(prefix).with_suffix('');report.to_csv(str(path)+'_report.csv',index=False)
                return report,[],0
            table=pd.concat(tables,ignore_index=True);files=gui_export(table,config,prefix,fmt,report)
            return report,files,len(table)
        self.batch_message.setText('Đang xử lý batch…');self._start_job(task,self._install_batch)

    def _install_batch(self,result):
        report,files,count=result;self.batch_report_model.set_dataframe(report)
        failed=int((report.Status=='error').sum())
        self.batch_message.setText(f'Hoàn tất · {count:,} feature rows · {failed} file lỗi · '+(' | '.join(files) if files else 'Không có dataset hợp lệ; đã lưu report.'))

def main():

    app = QApplication(
        sys.argv
    )

    window = FOGSignalAnalyzer()

    window.show()

    sys.exit(
        app.exec()
    )


# =============================================================
# RUN
# =============================================================

if __name__ == "__main__":

    main()