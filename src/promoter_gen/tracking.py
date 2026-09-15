"""Отправка метрик, графиков и файлов в ClearML"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional

if TYPE_CHECKING:
    from clearml import Logger, Task
    from plotly.graph_objects import Figure

class ClearMLTracker:
    """Адаптер ClearML для эксперимента по генерации промоторов"""

    def __init__(self, task: object, logger: object) -> None:
        self.task = task
        self.logger = logger

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        offline: bool,
        project_name: str,
        task_name: str,
    ) -> Optional["ClearMLTracker"]:
        """Создать задачу ClearML или отключить отслеживание"""
        if not enabled:
            return None

        try:
            from clearml import Task
        except ImportError as error:
            raise RuntimeError(
                "ClearML is not installed. Install requirements.txt or use --no-clearml."
            ) from error

        if offline:
            Task.set_offline(offline_mode=True)

        task = Task.init(
            project_name=project_name,
            task_name=task_name,
            task_type=Task.TaskTypes.training,
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False,
        )
        task.add_tags(["dna", "promoter-generation", "plotly-embed"])

        return cls(task=task, logger=task.get_logger())

    def connect_parameters(
        self,
        parameters: Mapping[str, object],
        section: str,
    ) -> None:
        self.task.connect(dict(parameters), name=section)

    def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        self.logger.report_scalar(
            title=title,
            series=series,
            value=float(value),
            iteration=iteration,
        )

    def report_plotly(self, title: str, figure: object) -> None:
        """Сохранить интерактивный график Plotly в задаче."""
        self.logger.report_plotly(
            title=title,
            series="interactive",
            iteration=0,
            figure=figure,
        )

    def upload_artifact(self, name: str, path: Path) -> None:
        """Загрузить созданный файл как артефакт."""
        uploaded = self.task.upload_artifact(
            name=name,
            artifact_object=str(Path(path).resolve()),
            wait_on_upload=True,
        )
        if not uploaded:
            raise RuntimeError(f"ClearML failed to upload artifact: {name}")

    def web_url(self) -> str:
        return str(self.task.get_output_log_web_page())

    def offline_folder(self) -> str:
        return str(self.task.get_offline_mode_folder())

    def close(self) -> None:
        self.task.flush(wait_for_uploads=True)
        self.task.close()