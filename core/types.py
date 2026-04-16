class BatchData:
    """Контейнер для батча изображений."""
    def __init__(self, file_path, grid_size=4):
        self.file_path = file_path
        self.grid_size = grid_size # 2 или 4
        self.frames = [] # Список numpy массивов
        self.name = file_path.split('/')[-1]

class Batch:
    def __init__(self, name, frames):
        self.name = name
        self.frames = frames  # Список numpy-массивов (кадров)

class ProjectManager:
    def __init__(self):
        self.batches = []  # Список объектов Batch

    def get_all_frames_flat(self):
        """Возвращает плоский список всех кадров для экспорта."""
        flat_list = []
        for b in self.batches:
            flat_list.extend(b.frames)
        return flat_list

class ProjectModel:
    """Глобальное состояние проекта."""
    def __init__(self):
        self.batches = [] # Список объектов BatchData
        self.fps = 30
        self.interpolation_enabled = False
