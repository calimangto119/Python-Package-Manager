import os
import sys
import subprocess
import shutil
import zipfile
import stat
import tempfile
import logging
import multiprocessing

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QFileDialog, QLineEdit, QCheckBox, QTabWidget,
    QMessageBox, QProgressBar, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import QtGui

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='package_manager.log',
    filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# --- Use no-window flag for subprocesses (only on Windows) ---
if os.name == 'nt':
    POPEN_FLAGS = {'creationflags': subprocess.CREATE_NO_WINDOW}
else:
    POPEN_FLAGS = {}

# --- Function to Locate the Windows Python Executable ---
def get_python_executable():
    if getattr(sys, 'frozen', False):
        logging.info("App is frozen. Skipping sys.executable.")
    else:
        if sys.executable and os.path.exists(sys.executable):
            logging.info(f"Using sys.executable: {sys.executable}")
            return sys.executable

    base_dir = os.path.dirname(sys.executable)
    embedded_python = os.path.join(base_dir, "python", "python.exe")
    if os.path.exists(embedded_python):
        logging.info(f"Found embedded python.exe at: {embedded_python}")
        return embedded_python

    python_exe = shutil.which("python.exe")
    if python_exe and os.path.exists(python_exe):
        logging.info(f"Found python.exe in PATH: {python_exe}")
        return python_exe

    common_paths = [
        r"C:\Python39\python.exe",
        r"C:\Python38\python.exe",
        r"C:\Python37\python.exe",
        r"C:\Program Files\Python39\python.exe",
        r"C:\Program Files (x86)\Python39\python.exe"
    ]
    for path in common_paths:
        if os.path.exists(path):
            logging.info(f"Found python.exe at standard location: {path}")
            return path

    logging.error("Could not locate a valid python.exe")
    return None

PYTHON_EXE = get_python_executable()
if PYTHON_EXE is None:
    QMessageBox.critical(None, "Error", "Unable to locate a valid python.exe. Please ensure Python is installed.")
    sys.exit(1)

# --- Extensive List of Popular Packages ---
POPULAR_PACKAGES = sorted(set([
    "numpy", "scipy", "pandas", "matplotlib", "seaborn", "scikit-learn",
    "tensorflow", "keras", "torch", "nltk", "spacy", "flask", "django",
    "requests", "beautifulsoup4", "lxml", "pytest", "sphinx", "jupyter",
    "notebook", "ipython", "sympy", "networkx", "statsmodels", "sqlalchemy",
    "pillow", "pyqt5", "pygame", "opencv-python", "scikit-image", "plotly",
    "dash", "streamlit", "bokeh", "cryptography", "pyyaml", "tornado",
    "paramiko", "fabric", "pymongo", "mysqlclient", "psycopg2", "boto3",
    "awscli", "google-api-python-client", "selenium", "gevent", "aiohttp",
    "uvloop", "asyncio", "twisted", "click", "docopt", "cffi", "pyinstaller",
    "cx_Freeze", "pydub", "youtube-dl", "moviepy", "mypy", "black", "flake8",
    "pylint", "rope", "pycodestyle", "autopep8", "isort", "eli5", "lime",
    "optuna", "xgboost", "lightgbm", "catboost", "imbalanced-learn", "gensim",
    "transformers", "huggingface-hub", "datasets", "nuitka", "pyside2", "kivy",
    "pyopengl", "pandas-profiling", "reportlab", "fpdf", "openpyxl", "xlrd",
    "xlwt", "pyodbc", "jinja2", "bottle", "cherrypy", "tqdm", "rich",
    "pygments", "markdown", "docutils", "pandasql", "pymc3", "arch",
    "bamboolib", "altair", "folium", "voila", "panel", "holoviews",
    "pyfiglet", "emoji", "colorama", "watchdog", "pandas-datareader", "mne",
    "netCDF4", "obspy", "acres", "aiosmtpd", "altgraph", "annotated-types",
    "anyio", "atpublic", "attrs", "blinker", "certifi", "chardet",
    "charset-normalizer", "ci-info", "common", "config", "configobj",
    "configparser", "contourpy", "cycler", "dacite", "decorator",
    "Deprecated", "dnslib", "et_xmlfile", "etelemetry", "fastapi", "filelock",
    "fitz", "flask-cors", "fonttools", "fuzzywuzzy", "h11", "httplib2",
    "idna", "imageio", "imageio-ffmpeg", "isodate", "itsdangerous", "jaconv",
    "Jinja2", "kiwisolver", "looseversion", "markdown-it-py", "MarkupSafe",
    "mdurl", "mutagen", "mysql", "nibabel", "nipype", "pdf2image", "pefile",
    "pip", "platformdirs", "plyer", "proglog", "prov", "psutil", "puremagic",
    "pycparser", "pydantic", "pydantic_core", "pydot", "Pygments",
    "pyinstaller-hooks-contrib", "pykakasi", "PyMuPDF", "pyodbc", "pyparsing",
    "PyQt5-Qt5", "PyQt5_sip", "PyQtWebEngine", "PyQtWebEngine-Qt5", "PySocks",
    "python-dateutil", "python-dotenv", "python-slugify", "pytube", "pytz",
    "pywin32", "pywin32-ctypes", "pyxnat", "QDarkStyle", "qt-material", "QtPy",
    "RapidFuzz", "rdflib", "redis", "setuptools", "simplejson", "six", "sniffio",
    "sortedcontainers", "soundcloud-v2", "soupsieve", "spotdl", "spotipy",
    "starlette", "syncedlyrics", "tenacity", "text-unidecode", "traits", "trio",
    "trio-websocket", "typing_extensions", "typing-inspection", "tzdata",
    "undetected-chromedriver", "Unidecode", "urllib3", "uvicorn",
    "websocket-client", "websockets", "Werkzeug", "wrapt", "wsproto",
    "XlsxWriter", "yt-dlp", "ytmusicapi",
    # Additional packages
    "dask", "joblib", "pyarrow", "fsspec", "s3fs", "simpy", "hypothesis",
    "nose2", "fastai", "tensorboard", "mlflow", "wandb", "stable-baselines3",
    "gym", "pettingzoo", "ray", "prefect", "geopandas", "shapely", "pyproj",
    "rasterio", "fastparquet", "tables", "scrapy", "fire", "numba", "cupy",
    "apache-airflow", "comet_ml", "pytorch-lightning", "mkdocs", "mkdocs-material",
    "pdoc", "pipenv", "poetry", "twine", "wheel", "hyperopt", "prophet",
    "pymoo", "zarr", "ipywidgets", "sentence-transformers", "fairseq", "librosa",
    "sounddevice", "sentry-sdk", "pympler", "pyinstrument", "visdom", "pycaret",
    "tpot", "auto-sklearn", "yfinance", "tabulate"
]))

class DownloadWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str, int)

    def __init__(self, packages, zip_output_path, parent=None):
        super().__init__(parent)
        self.packages = packages
        self.zip_output_path = zip_output_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        self.log.emit("Download cancelled by user.")

    def run(self):
        downloaded_count = 0
        temp_dir = tempfile.mkdtemp(prefix="pkg_download_")
        self.log.emit(f"Created temporary directory: {temp_dir}")
        total = len(self.packages)
        for idx, package in enumerate(self.packages):
            if self._cancelled:
                self.log.emit("Download cancelled.")
                break
            self.log.emit(f"Downloading package: {package}")
            try:
                proc = subprocess.Popen(
                    [PYTHON_EXE, "-m", "pip", "download", "-d", temp_dir, package],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    **POPEN_FLAGS
                )
                for line in proc.stdout:
                    self.log.emit(line.strip())
                proc.wait()
                if proc.returncode == 0:
                    downloaded_count += 1
                else:
                    self.log.emit(f"Error downloading {package}.")
            except Exception as e:
                self.log.emit(f"Exception: {e}")
            self.progress.emit(idx + 1)
        self.log.emit("Creating ZIP file...")
        try:
            with zipfile.ZipFile(self.zip_output_path, 'w') as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, temp_dir)
                        zipf.write(full_path, arcname=arcname)
            self.log.emit(f"ZIP file created: {self.zip_output_path}")
            success = True
        except Exception as e:
            self.log.emit(f"Error creating ZIP file: {e}")
            success = False
        try:
            shutil.rmtree(temp_dir)
            self.log.emit("Temporary directory removed.")
        except Exception as e:
            self.log.emit(f"Error removing temp directory: {e}")
        self.finished.emit(success, self.zip_output_path, downloaded_count)

class InstallWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, int)

    def __init__(self, pkg_files, base_dir, parent=None):
        super().__init__(parent)
        self.pkg_files = pkg_files
        self.base_dir = base_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        self.log.emit("Installation cancelled by user.")

    def run(self):
        installed_count = 0
        total = len(self.pkg_files)
        for idx, pkg_file in enumerate(self.pkg_files):
            if self._cancelled:
                self.log.emit("Installation cancelled.")
                break
            self.log.emit(f"Installing: {pkg_file}")
            full_path = os.path.join(self.base_dir, pkg_file)
            try:
                proc = subprocess.Popen(
                    [PYTHON_EXE, "-m", "pip", "install", "--no-index", "--find-links", self.base_dir, full_path],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    **POPEN_FLAGS
                )
                for line in proc.stdout:
                    self.log.emit(line.strip())
                proc.wait()
                if proc.returncode == 0:
                    installed_count += 1
                else:
                    self.log.emit(f"Error installing {pkg_file}.")
            except Exception as e:
                self.log.emit(f"Exception installing {pkg_file}: {e}")
            self.progress.emit(idx + 1)
        success = (installed_count == total)
        self.finished.emit(success, installed_count)

class PackageDownloader(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Package Manager by: https://github.com/calimangto119 ")
        self.resize(800, 600)
        self.tabs = QTabWidget()
        self.download_tab = QWidget()
        self.install_tab = QWidget()
        self.packages_tab = QWidget()

        self.tabs.addTab(self.download_tab, "Download Packages")
        self.tabs.addTab(self.install_tab, "Install from ZIP")
        self.tabs.addTab(self.packages_tab, "Installed Packages")

        self.init_download_tab()
        self.init_install_tab()
        self.init_packages_tab()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        self.current_download_worker = None
        self.current_install_worker = None

    def init_download_tab(self):
        # Function to install packages online directly
        def install_online():
            packages = [item.text() for item in self.package_items if item.checkState() == Qt.Checked]
            if not packages:
                QMessageBox.warning(self, "No Packages", "Please select at least one package for online installation.")
                return
            self.download_output.clear()
            self.download_progress.setValue(0)
            self.download_progress.setMaximum(len(packages))
            try:
                proc = subprocess.Popen(
                    [PYTHON_EXE, "-m", "pip", "install", *packages],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    **POPEN_FLAGS
                )
                for line in proc.stdout:
                    self.download_output.append(line.strip())
                    QApplication.processEvents()
                proc.wait()
                self.download_progress.setValue(len(packages))
                QMessageBox.information(self, "Installation Complete", "Selected packages installed online successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Online installation failed: {e}")

        layout = QVBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search packages...")
        self.search_box.textChanged.connect(self.filter_packages)
        layout.addWidget(self.search_box)

        self.package_list = QListWidget()
        self.package_items = []
        for pkg in POPULAR_PACKAGES:
            item = QListWidgetItem(pkg)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.package_list.addItem(item)
            self.package_items.append(item)
        layout.addWidget(self.package_list)

        self.select_all_download = QCheckBox("Select All")
        self.select_all_download.stateChanged.connect(self.toggle_all_download)
        layout.addWidget(self.select_all_download)

        self.custom_input = QLineEdit()
        self.custom_input.setPlaceholderText("Add custom package (press Enter to add)")
        self.custom_input.returnPressed.connect(self.add_custom_package)
        layout.addWidget(self.custom_input)

        btn_layout = QHBoxLayout()
        self.download_btn = QPushButton("Download Selected Packages to ZIP")
        self.download_btn.clicked.connect(self.start_download)
        btn_layout.addWidget(self.download_btn)

        # New Install Online button
        self.install_online_btn = QPushButton("Install Online")
        self.install_online_btn.clicked.connect(install_online)
        btn_layout.addWidget(self.install_online_btn)

        self.cancel_dl_btn = QPushButton("Cancel Download")
        self.cancel_dl_btn.clicked.connect(self.cancel_download)
        self.cancel_dl_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_dl_btn)
        layout.addLayout(btn_layout)

        self.download_progress = QProgressBar()
        layout.addWidget(self.download_progress)

        self.download_output = QTextEdit()
        self.download_output.setReadOnly(True)
        layout.addWidget(self.download_output)

        self.download_tab.setLayout(layout)

    def filter_packages(self, text):
        for item in self.package_items:
            item.setHidden(text.lower() not in item.text().lower())

    def toggle_all_download(self, state):
        for item in self.package_items:
            item.setCheckState(Qt.Checked if state == Qt.Checked else Qt.Unchecked)

    def add_custom_package(self):
        pkg = self.custom_input.text().strip()
        if pkg:
            item = QListWidgetItem(pkg)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.package_list.addItem(item)
            self.package_items.append(item)
            self.custom_input.clear()

    def start_download(self):
        packages = [item.text() for item in self.package_items if item.checkState() == Qt.Checked]
        if not packages:
            QMessageBox.warning(self, "No Packages", "Please select at least one package.")
            return
        output_zip, _ = QFileDialog.getSaveFileName(self, "Save ZIP", "", "Zip Files (*.zip)")
        if not output_zip:
            return
        self.download_output.clear()
        self.download_progress.setValue(0)
        self.download_progress.setMaximum(len(packages))
        self.current_download_worker = DownloadWorker(packages, output_zip)
        self.current_download_worker.progress.connect(self.download_progress.setValue)
        self.current_download_worker.log.connect(lambda msg: self.download_output.append(msg))
        self.current_download_worker.finished.connect(
            lambda success, path, count: QMessageBox.information(
                self,
                "Download Finished",
                f"Downloaded {count} packages.\nZIP created at:\n{path}" if success else "Download failed."
            )
        )
        self.current_download_worker.start()
        self.cancel_dl_btn.setEnabled(True)

    def cancel_download(self):
        if self.current_download_worker:
            self.current_download_worker.cancel()
            self.cancel_dl_btn.setEnabled(False)

    def init_install_tab(self):
        def toggle_all_zip(state):
            for i in range(self.zip_list.count()):
                self.zip_list.item(i).setCheckState(Qt.Checked if state == Qt.Checked else Qt.Unchecked)
        layout = QVBoxLayout()
        self.load_zip_btn = QPushButton("Load ZIP File")
        self.load_zip_btn.clicked.connect(self.load_zip)
        layout.addWidget(self.load_zip_btn)

        # New Select All checkbox for Install from ZIP tab
        self.select_all_zip = QCheckBox("Select All")
        self.select_all_zip.stateChanged.connect(toggle_all_zip)
        layout.addWidget(self.select_all_zip)

        self.zip_list = QListWidget()
        layout.addWidget(self.zip_list)

        btn_layout = QHBoxLayout()
        self.install_btn = QPushButton("Install Selected Packages")
        self.install_btn.clicked.connect(self.start_install)
        btn_layout.addWidget(self.install_btn)
        self.cancel_install_btn = QPushButton("Cancel Installation")
        self.cancel_install_btn.clicked.connect(self.cancel_install)
        self.cancel_install_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_install_btn)
        layout.addLayout(btn_layout)

        self.install_progress = QProgressBar()
        layout.addWidget(self.install_progress)

        self.install_output = QTextEdit()
        self.install_output.setReadOnly(True)
        layout.addWidget(self.install_output)

        self.install_tab.setLayout(layout)

    def load_zip(self):
        zip_path, _ = QFileDialog.getOpenFileName(self, "Open ZIP File", "", "Zip Files (*.zip)")
        if not zip_path:
            return
        self.temp_install_dir = tempfile.mkdtemp(prefix="pkg_install_")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(self.temp_install_dir)
            self.zip_list.clear()
            for f in os.listdir(self.temp_install_dir):
                if f.endswith(('.whl', '.tar.gz')):
                    item = QListWidgetItem(f)
                    item.setCheckState(Qt.Checked)
                    self.zip_list.addItem(item)
            self.install_output.append("ZIP file loaded and extracted.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to extract ZIP: {e}")

    def start_install(self):
        if not hasattr(self, "temp_install_dir"):
            QMessageBox.warning(self, "No ZIP Loaded", "Please load a ZIP file first.")
            return
        pkg_files = [self.zip_list.item(i).text() for i in range(self.zip_list.count())
                     if self.zip_list.item(i).checkState() == Qt.Checked]
        if not pkg_files:
            QMessageBox.warning(self, "No Packages", "Please select at least one package to install.")
            return
        self.install_progress.setValue(0)
        self.install_progress.setMaximum(len(pkg_files))
        self.install_output.clear()
        self.current_install_worker = InstallWorker(pkg_files, self.temp_install_dir)
        self.current_install_worker.progress.connect(self.install_progress.setValue)
        self.current_install_worker.log.connect(lambda msg: self.install_output.append(msg))
        self.current_install_worker.finished.connect(
            lambda success, count: QMessageBox.information(
                self, "Installation Finished", f"Installed {count} packages." if success else "Some packages failed to install."
            )
        )
        self.current_install_worker.start()
        self.cancel_install_btn.setEnabled(True)

    def init_packages_tab(self):
        layout = QVBoxLayout()
        self.installed_list = QListWidget()
        layout.addWidget(self.installed_list)

        self.select_all_installed = QCheckBox("Select All")
        self.select_all_installed.stateChanged.connect(
            lambda state: [self.installed_list.item(i).setCheckState(Qt.Checked if state == Qt.Checked else Qt.Unchecked)
                           for i in range(self.installed_list.count())]
        )
        layout.addWidget(self.select_all_installed)

        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Package List")
        refresh_btn.clicked.connect(self.refresh_installed_packages)
        btn_layout.addWidget(refresh_btn)

        uninstall_btn = QPushButton("Uninstall Selected")
        uninstall_btn.clicked.connect(self.uninstall_selected_packages)
        btn_layout.addWidget(uninstall_btn)

        export_btn = QPushButton("Download Selected to ZIP")
        export_btn.clicked.connect(self.export_selected_packages_to_zip)
        btn_layout.addWidget(export_btn)

        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(lambda: [self.installed_list.item(i).setCheckState(Qt.Unchecked)
                                             for i in range(self.installed_list.count())])
        btn_layout.addWidget(clear_btn)

        layout.addLayout(btn_layout)

        self.packages_output = QTextEdit()
        self.packages_output.setReadOnly(True)
        layout.addWidget(self.packages_output)

        self.packages_tab.setLayout(layout)
        self.refresh_installed_packages()

    def refresh_installed_packages(self):
        self.installed_list.clear()
        try:
            output = subprocess.check_output([PYTHON_EXE, '-m', 'pip', 'list', '--format=freeze'], text=True)
            for line in output.strip().splitlines():
                pkg = line.split("==")[0]
                item = QListWidgetItem(pkg)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.installed_list.addItem(item)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to retrieve installed packages: {e}")

    def uninstall_selected_packages(self):
        selected = [self.installed_list.item(i).text() for i in range(self.installed_list.count())
                    if self.installed_list.item(i).checkState() == Qt.Checked]
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select at least one package to uninstall.")
            return
        try:
            proc = subprocess.Popen(
                [PYTHON_EXE, '-m', 'pip', 'uninstall', '-y', *selected],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                **POPEN_FLAGS
            )
            self.packages_output.append("Uninstalling: " + ", ".join(selected))
            for line in proc.stdout:
                self.packages_output.append(line.strip())
            proc.wait()
            QMessageBox.information(self, "Done", "Selected packages have been uninstalled.")
            self.refresh_installed_packages()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Uninstallation failed: {e}")

    def export_selected_packages_to_zip(self):
        selected = [self.installed_list.item(i).text() for i in range(self.installed_list.count())
                    if self.installed_list.item(i).checkState() == Qt.Checked]
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select at least one package to export.")
            return
        zip_path, _ = QFileDialog.getSaveFileName(self, "Save Installed Packages ZIP", "", "Zip Files (*.zip)")
        if not zip_path:
            return
        self.packages_output.clear()
        self.export_worker = DownloadWorker(selected, zip_path)
        self.export_worker.progress.connect(lambda v: self.packages_output.append(f"Progress: {v}/{len(selected)}"))
        self.export_worker.log.connect(lambda msg: self.packages_output.append(msg))
        self.export_worker.finished.connect(
            lambda success, path, count: QMessageBox.information(
                self,
                "Export Finished",
                f"Downloaded {count} packages.\nZIP created at:\n{path}" if success else "Export failed."
            )
        )
        self.export_worker.start()

    def cancel_install(self):
        if self.current_install_worker:
            self.current_install_worker.cancel()
            self.cancel_install_btn.setEnabled(False)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(255, 255, 255))
    dark_palette.setColor(QtGui.QPalette.Text, QtGui.QColor(255, 255, 255))
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(255, 255, 255))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
    app.setPalette(dark_palette)

    win = PackageDownloader()
    win.show()
    sys.exit(app.exec_())
