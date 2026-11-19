"""Run processing off the GUI thread. Qt's native finished signal owns cleanup."""

from PyQt6.QtCore import QThread, pyqtSignal


class TaskCancelled(Exception):
    pass


class TaskThread(QThread):
    progress = pyqtSignal(int)

    def __init__(self, work, parent=None):
        super().__init__(parent)
        self.work = work
        self.result = None
        self.error = None
        self.cancelled = False

    def _check_interrupted(self):
        if self.isInterruptionRequested():
            raise TaskCancelled

    def run(self):
        try:
            self._check_interrupted()
            self.result = self.work(self.progress.emit, self._check_interrupted)
        except TaskCancelled:
            self.cancelled = True
        except Exception as error:
            self.error = str(error) or type(error).__name__


def stop_tasks(*tasks):
    running = [task for task in tasks if task is not None and task.isRunning()]
    for task in running:
        task.requestInterruption()
    for task in running:
        task.wait()
