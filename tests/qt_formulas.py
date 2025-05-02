import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib

# enable full LaTeX rendering (requires a TeX installation)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
# ensure aligned, align etc. are available
matplotlib.rcParams['text.latex.preamble'] = ['\\usepackage{amsmath}']

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        fig = plt.figure(figsize=(6, 4))
        fig.patch.set_facecolor('white')
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(111)
        ax.axis('off')

        # use a display‐math block with aligned
        formula = r"""\[
\begin{aligned}
m_i \ddot q_i + c_i \dot q_i \\
\sum_{j=1}^N k_{ij} q_j \\
\sum_{j,k=1}^N \alpha_{ijk} q_j q_k \\
\sum_{j,k,l=1}^N \gamma_{ijkl} q_j q_k q_l \\
= f_i \cos(\omega_d t),\quad i=1,\dots,N
\end{aligned}
\]"""
        ax.text(0.5, 0.5, formula,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=16)

        # ensure the text isn't clipped and gets drawn
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        canvas.draw()

        self.setCentralWidget(canvas)
        self.setWindowTitle('Qt LaTeX Formula Display')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
