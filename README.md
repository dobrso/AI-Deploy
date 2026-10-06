# Стек
```bash
Python 3.13
Библиотеки машинного обучения: PyTorch, Hugging Face, Scikit Learn
Разработка API: FastAPI
Тестирование: Pytest
Библиотеки для работы с данными: numpy, pandas, pillow
Библиотеки для сбора метрик: psutil, jiwer
```

# Быстрый старт
1. Клонирование репозитория  
```bash
git clone https://github.com/dobrso/AI-Deploy.git
```

2. Переход в директорию  
```bash
cd ~/AI-Deploy/
```

3. Создание виртуального окружения на версии Python 3.13  
```bash 
py -3.13 -m venv venv
```

4. Активация виртуального окружения
```bash
/venv/Scripts/activate
```

5. Установка зависимостей  
```bash
pip install -r requirements.txt
```

# Навигация по файлам

```bash
~/ai_deploy/api/ - Точка запуска API (fastapi dev)
~/ai_deploy/metrics/ - Пакет, в котором хранятся скрипты для сбора метрик для каждой модели
~/ai_deploy/models/ - Пакет, в котором лежат все модели
~/ai_deploy/tests/ - Точка запуска тестов к API (pytest)
~/rec_systems/ - Пакет, в котором находятся 4 реализованные рекомендательные системы и скрип для сбора метрик по ним
```

#### Важно! Для работы collect_audio_metrics необходимо установить ffmpeg