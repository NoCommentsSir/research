import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Если Model и generate_point_cloud уже определены выше в ноутбуке,
# эти импорты не нужны.
# Иначе раскомментируй и поправь пути под свою структуру проекта:
#
from scripts.model.model import Model
from scripts.utils.clouds import generate_point_cloud


# =========================
# НАСТРОЙКИ
# =========================

DATASET_ROOT = Path("data/ModelNet10")   # <-- замени на путь к своему датасету
CATEGORIES = ["chair", "sofa", "table", "bed", "toilet", "night_stand"]
POINTS_PER_MODEL = 3000

MESH_OUTPUT = "mesh_grid.png"
POINTCLOUD_OUTPUT = "pointcloud_grid.png"


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def find_sample_model(dataset_root: Path, category: str) -> Path:
    """
    Ищет один пример модели для категории.
    Сначала смотрит train, потом test.
    """
    train_dir = dataset_root / category / "train"
    test_dir = dataset_root / category / "test"

    train_files = sorted(train_dir.glob("*.off")) if train_dir.exists() else []
    test_files = sorted(test_dir.glob("*.off")) if test_dir.exists() else []

    if train_files:
        return train_files[0]
    if test_files:
        return test_files[0]

    raise FileNotFoundError(f"Не найдено .off файлов для категории: {category}")


def set_axes_equal(ax, points: np.ndarray):
    """
    Делает одинаковый масштаб по всем осям,
    чтобы 3D-объекты не искажались.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    if max_range < 1e-8:
        max_range = 1.0

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)


def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)


def plot_mesh(ax, model, title: str):
    """
    Рисует mesh-модель через matplotlib.
    """
    vertices = np.array(model.vertices, dtype=float)
    triangles = np.array(model.triangles, dtype=int)

    faces = vertices[triangles]  # shape: (num_triangles, 3, 3)

    mesh = Poly3DCollection(
        faces,
        linewidths=0.2,
        edgecolors="black",
        alpha=0.9
    )
    ax.add_collection3d(mesh)

    set_axes_equal(ax, vertices)
    clean_axes(ax)
    ax.set_title(title, fontsize=11, pad=10)
    ax.view_init(elev=20, azim=35)


def plot_point_cloud(ax, point_cloud, title: str):
    """
    Рисует облако точек.
    Ожидается, что point_cloud.points — ndarray shape (N, 3)
    """
    points = np.array(point_cloud.points_arr, dtype=float)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=1
    )

    set_axes_equal(ax, points)
    clean_axes(ax)
    ax.set_title(title, fontsize=11, pad=10)
    ax.view_init(elev=20, azim=35)


# =========================
# ОСНОВНЫЕ ФУНКЦИИ
# =========================

def generate_models_for_categories(dataset_root: Path, categories):
    """
    Загружает и нормализует по одной модели на категорию.
    """
    models = []

    for category in categories:
        file_path = find_sample_model(dataset_root, category)
        model = Model(str(file_path))
        model.normalize()
        models.append((category, model, file_path))

    return models


def create_mesh_grid(models, output_path: str):
    """
    Создает картинку-сетку из mesh-моделей.
    """
    fig = plt.figure(figsize=(12, 8))

    for i, (category, model, _) in enumerate(models, start=1):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        plot_mesh(ax, model, category)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_pointcloud_grid(models, output_path: str, n_points: int = 3000):
    """
    Создает картинку-сетку из облаков точек.
    """
    fig = plt.figure(figsize=(12, 8))

    for i, (category, model, _) in enumerate(models, start=1):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        pc = generate_point_cloud(model, n_points)
        plot_point_cloud(ax, pc, category)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# ЗАПУСК
# =========================

models = generate_models_for_categories(DATASET_ROOT, CATEGORIES)

print("Выбранные модели:")
for category, _, path in models:
    print(f"{category}: {path}")

create_mesh_grid(models, MESH_OUTPUT)
create_pointcloud_grid(models, POINTCLOUD_OUTPUT, POINTS_PER_MODEL)

print(f"\nСохранено:")
print(f" - {MESH_OUTPUT}")
print(f" - {POINTCLOUD_OUTPUT}")