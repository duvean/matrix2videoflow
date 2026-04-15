class BatchData:
    """Контейнер для батча изображений."""
    def __init__(self, file_path, grid_size=4):
        self.file_path = file_path
        self.grid_size = grid_size # 2 или 4
        self.frames = [] # Список numpy массивов
        self.name = file_path.split('/')[-1]

class ProjectModel:
    """Глобальное состояние проекта."""
    def __init__(self):
        self.batches = [] # Список объектов BatchData
        self.fps = 30
        self.interpolation_enabled = False
