import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QMessageBox,
    QTabWidget,
    QGridLayout,
    QSpinBox,
    QDoubleSpinBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from Visualization import Visualization
from HeisenbergFCC import HeisenbergFCC
from updaters import create_updater
from typing import Optional, List, Set, Dict
from simulation import AnnealingSimulation


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)


class InteractiveApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heisenberg Model Visualization")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化模型参数
        self.L = 4
        self.T = 1.0
        self.algorithm = "wolff"
        self.measurement_interval = 10
        self.current_step = 0

        # 创建模型和可视化实例
        self.model = HeisenbergFCC(L=self.L, T=self.T)
        self.updater = create_updater(self.model, self.algorithm)
        self.vis = Visualization()

        # 存储物理量历史数据
        self.energy_history = []
        self.magnetization_history = []
        self.steps = []

        # 创建主窗口部件
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # 创建标签页
        self.create_tabs()

        # 创建控制面板
        self.create_control_panel()

        # 定时器用于动画
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.animating = False

    def create_control_panel(self):
        control_panel = QWidget()
        control_layout = QGridLayout()

        # 系统参数控制
        control_layout.addWidget(QLabel("System Size (L):"), 0, 0)
        self.L_spin = QSpinBox()
        self.L_spin.setRange(2, 32)
        self.L_spin.setValue(self.L)
        control_layout.addWidget(self.L_spin, 0, 1)

        control_layout.addWidget(QLabel("Temperature (T):"), 0, 2)
        self.T_spin = QDoubleSpinBox()
        self.T_spin.setRange(0.1, 10.0)
        self.T_spin.setValue(self.T)
        self.T_spin.setSingleStep(0.1)
        control_layout.addWidget(self.T_spin, 0, 3)

        control_layout.addWidget(QLabel("Algorithm:"), 0, 4)
        self.alg_combo = QComboBox()
        self.alg_combo.addItems(["wolff", "sw", "single"])
        self.alg_combo.setCurrentText(self.algorithm)
        control_layout.addWidget(self.alg_combo, 0, 5)

        # 控制按钮
        self.update_button = QPushButton("Update Model")
        self.update_button.clicked.connect(self.update_model)
        control_layout.addWidget(self.update_button, 1, 0)

        self.animate_button = QPushButton("Start Animation")
        self.animate_button.clicked.connect(self.toggle_animation)
        control_layout.addWidget(self.animate_button, 1, 1)

        self.reset_button = QPushButton("Reset Data")
        self.reset_button.clicked.connect(self.reset_data)
        control_layout.addWidget(self.reset_button, 1, 2)

        control_panel.setLayout(control_layout)
        self.layout.addWidget(control_panel)

    def create_tabs(self):
        self.tabs = QTabWidget()

        # 自旋构型标签页
        self.spins_tab = QWidget()
        spins_layout = QVBoxLayout()
        self.spins_canvas = MplCanvas(self, width=8, height=6)
        self.spins_toolbar = NavigationToolbar(self.spins_canvas, self)
        spins_layout.addWidget(self.spins_toolbar)
        spins_layout.addWidget(self.spins_canvas)
        self.spins_tab.setLayout(spins_layout)

        # 物理量演化标签页
        self.evolution_tab = QWidget()
        evolution_layout = QVBoxLayout()
        self.evolution_canvas = MplCanvas(self, width=8, height=6)
        self.evolution_toolbar = NavigationToolbar(self.evolution_canvas, self)
        evolution_layout.addWidget(self.evolution_toolbar)
        evolution_layout.addWidget(self.evolution_canvas)
        self.evolution_tab.setLayout(evolution_layout)

        # 关联函数标签页
        self.correlation_tab = QWidget()
        correlation_layout = QVBoxLayout()
        self.correlation_canvas = MplCanvas(self, width=8, height=6)
        self.correlation_toolbar = NavigationToolbar(self.correlation_canvas, self)
        correlation_layout.addWidget(self.correlation_toolbar)
        correlation_layout.addWidget(self.correlation_canvas)
        self.correlation_tab.setLayout(correlation_layout)

        # 结构因子标签页
        self.structure_tab = QWidget()
        structure_layout = QVBoxLayout()
        self.structure_canvas = MplCanvas(self, width=8, height=6)
        self.structure_toolbar = NavigationToolbar(self.structure_canvas, self)
        structure_layout.addWidget(self.structure_toolbar)
        structure_layout.addWidget(self.structure_canvas)
        self.structure_tab.setLayout(structure_layout)

        # 添加相变分析标签页
        self.phase_transition_tab = QWidget()
        phase_layout = QVBoxLayout()

        # 控制面板
        control_panel = QWidget()
        control_grid = QGridLayout()

        # 温度范围控制
        control_grid.addWidget(QLabel("T start:"), 0, 0)
        self.T_start_spin = QDoubleSpinBox()
        self.T_start_spin.setRange(0.1, 10.0)
        self.T_start_spin.setValue(1.0)
        self.T_start_spin.setSingleStep(0.1)
        control_grid.addWidget(self.T_start_spin, 0, 1)

        control_grid.addWidget(QLabel("T end:"), 0, 2)
        self.T_end_spin = QDoubleSpinBox()
        self.T_end_spin.setRange(0.1, 10.0)
        self.T_end_spin.setValue(10.0)
        self.T_end_spin.setSingleStep(0.1)
        control_grid.addWidget(self.T_end_spin, 0, 3)

        control_grid.addWidget(QLabel("Cooling steps:"), 0, 4)
        self.cooling_steps_spin = QSpinBox()
        self.cooling_steps_spin.setRange(10, 1000)
        self.cooling_steps_spin.setValue(200)
        control_grid.addWidget(self.cooling_steps_spin, 0, 5)

        # 运行按钮
        self.run_annealing_button = QPushButton("Run Annealing")
        self.run_annealing_button.clicked.connect(self.run_phase_transition_analysis)
        control_grid.addWidget(self.run_annealing_button, 1, 0, 1, 2)

        control_panel.setLayout(control_grid)
        phase_layout.addWidget(control_panel)

        # 结果显示区域
        self.phase_canvas = MplCanvas(self, width=8, height=6)
        self.phase_toolbar = NavigationToolbar(self.phase_canvas, self)
        phase_layout.addWidget(self.phase_toolbar)
        phase_layout.addWidget(self.phase_canvas)

        self.phase_transition_tab.setLayout(phase_layout)

        # 添加标签页
        self.tabs.addTab(self.spins_tab, "Spin Configuration")
        self.tabs.addTab(self.evolution_tab, "Physical Quantities")
        self.tabs.addTab(self.correlation_tab, "Correlation Function")
        self.tabs.addTab(self.structure_tab, "Structure Factor")
        self.tabs.addTab(self.phase_transition_tab, "Phase Transition")

        self.layout.addWidget(self.tabs)

        # 初始化绘图
        self.plot_spins()
        self.plot_evolution()
        self.plot_correlation()
        self.plot_structure_factor()

    def update_model(self):
        try:
            new_L = self.L_spin.value()
            new_T = self.T_spin.value()
            new_algorithm = self.alg_combo.currentText()

            # 检查是否需要重新创建模型
            if new_L != self.L or new_T != self.T or new_algorithm != self.algorithm:
                self.L = new_L
                self.T = new_T
                self.algorithm = new_algorithm

                # 重新创建模型和更新器
                self.model = HeisenbergFCC(L=self.L, T=self.T)
                self.updater = create_updater(self.model, self.algorithm)

                # 重置数据
                self.reset_data()

                QMessageBox.information(self, "Success", "Model updated successfully!")

                # 更新所有视图
                self.plot_spins()
                self.plot_evolution()
                self.plot_correlation()
                self.plot_structure_factor()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update model: {str(e)}")

    def reset_data(self):
        self.energy_history = []
        self.magnetization_history = []
        self.steps = []
        self.current_step = 0
        self.plot_evolution()

    def plot_spins(self, clusters: Optional[List[Set[int]]] = None):
        self.spins_canvas.fig.clear()
        ax = self.spins_canvas.fig.add_subplot(111, projection="3d")
        self.vis.plot_spins(self.model, clusters=clusters, ax=ax)
        self.spins_canvas.draw()

    def plot_evolution(self):
        self.evolution_canvas.fig.clear()
        fig = self.evolution_canvas.fig

        # 创建两个子图
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # 绘制能量演化
        if self.steps:
            ax1.plot(self.steps, self.energy_history, "b-", label="Energy/N")
            ax2.plot(self.steps, self.magnetization_history, "r-", label="|M|")

            ax1.legend()
            ax2.legend()

        ax1.set_xlabel("MC Steps")
        ax1.set_ylabel("Energy per Spin")
        ax1.grid(True)

        ax2.set_xlabel("MC Steps")
        ax2.set_ylabel("Magnetization")
        ax2.grid(True)

        fig.tight_layout()
        self.evolution_canvas.draw()

    def plot_correlation(self):
        self.correlation_canvas.fig.clear()
        correlation = self.model.measure_correlation()
        distance = np.arange(len(correlation))

        ax = self.correlation_canvas.fig.add_subplot(111)
        ax.plot(distance, correlation, "o-")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Correlation")
        ax.set_yscale("log")
        ax.grid(True)
        ax.set_title("Spin-Spin Correlation Function")

        self.correlation_canvas.draw()

    def plot_structure_factor(self):
        self.structure_canvas.fig.clear()
        sf = self.model.measure_structure_factor()
        L = sf.shape[0]

        ax = self.structure_canvas.fig.add_subplot(111)
        im = ax.imshow(
            sf[:, :, L // 2],
            extent=[-np.pi, np.pi, -np.pi, np.pi],
            cmap="hot",
            aspect="auto",
        )
        self.structure_canvas.fig.colorbar(im, ax=ax, label="S(q)")
        ax.set_xlabel("qx")
        ax.set_ylabel("qy")
        ax.set_title("Structure Factor S(q) [qz=0 plane]")

        self.structure_canvas.draw()

    def toggle_animation(self):
        if not self.animating:
            self.animating = True
            self.animate_button.setText("Stop Animation")
            self.timer.start(100)  # 每100毫秒更新一次
        else:
            self.animating = False
            self.animate_button.setText("Start Animation")
            self.timer.stop()

    def update_animation(self):
        # 执行更新
        self.updater.update()
        self.current_step += 1

        # 获取簇信息
        clusters = None
        if hasattr(self.updater, "cluster"):
            clusters = [self.updater.cluster]
        elif hasattr(self.updater, "clusters"):
            clusters = self.updater.clusters

        # 每隔一定步数记录物理量
        if self.current_step % self.measurement_interval == 0:
            energy = self.model.energy / self.model.N
            magnetization = np.linalg.norm(self.model.calculate_magnetization())

            self.energy_history.append(energy)
            self.magnetization_history.append(magnetization)
            self.steps.append(self.current_step)

            # 更新图表
            self.plot_evolution()

            # 定期更新关联函数和结构因子
            if self.current_step % (self.measurement_interval * 10) == 0:
                self.plot_correlation()
                self.plot_structure_factor()

        # 更新自旋构型显示
        self.plot_spins(clusters=clusters)

    def run_phase_transition_analysis(self):
        """执行相变分析"""
        try:
            # 获取参数
            T_start = self.T_start_spin.value()
            T_end = self.T_end_spin.value()
            cooling_steps = self.cooling_steps_spin.value()

            # 创建模拟实例
            sim = AnnealingSimulation(
                L=self.L,
                T_start=T_start,
                T_end=T_end,
                cooling_steps=cooling_steps,
                mc_steps_per_T=1000,  # 可以添加控制
                thermalization_steps=100,  # 可以添加控制
            )

            # 运行模拟
            results = sim.run()

            # 显示结果
            self.plot_phase_transition_results(results)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to run analysis: {str(e)}")

    def plot_phase_transition_results(self, results: Dict):
        """绘制相变分析结果"""
        self.phase_canvas.fig.clear()

        # 创建2x2的子图
        ((ax1, ax2), (ax3, ax4)) = self.phase_canvas.fig.subplots(2, 2)

        T = results["temperature"]

        # 能量
        ax1.plot(T, results["energy"], "o-")
        ax1.set_xlabel("Temperature")
        ax1.set_ylabel("Energy per spin")
        ax1.grid(True)

        # 磁化强度
        ax2.plot(T, results["magnetization"], "o-")
        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("Magnetization")
        ax2.grid(True)

        # 比热容
        ax3.plot(T, results["specific_heat"], "o-")
        ax3.set_xlabel("Temperature")
        ax3.set_ylabel("Specific Heat")
        ax3.grid(True)

        # 磁化率
        ax4.plot(T, results["susceptibility"], "o-")
        ax4.set_xlabel("Temperature")
        ax4.set_ylabel("Susceptibility")
        ax4.grid(True)

        self.phase_canvas.fig.tight_layout()
        self.phase_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InteractiveApp()
    window.show()
    sys.exit(app.exec_())
