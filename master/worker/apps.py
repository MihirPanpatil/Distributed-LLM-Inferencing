from django.apps import AppConfig


class WorkerConfig(AppConfig):
    name = 'worker'
    verbose_name = 'Worker Node Management'

    def ready(self):
        """
        Initialize any app-specific settings or signals when the app is ready.
        """
        pass
