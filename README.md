# Мини‑BYOL (самообучение без отрицательных пар)

Репозиторий содержит реализацию метода Bootstrap Your Own Latent (BYOL) для самообучения представлений на датасете MNIST с последующим сравнением с supervised подходом.

## Инструкция запуска

### Предварительные требования
```bash
# Клонирование репозитория
git clone https://github.com/andreymokriev/MiniBYOL.git
cd MiniBYOL

# Установка зависимостей
pip install torch torchvision jupyter matplotlib numpy
```
### Запуск

1. Самообучение BYOL
```bash
jupyter notebook MiniBYOL.ipynb
```
2.  Контролируемое обучение
```bash
jupyter notebook Supervised.ipynb
```

## Результаты

### Производительность на MNIST

| Метод | τ | Точность | 
|-------|---|----------|
| **BYOL + Linear** | 0.995 | ~98.22% |
| **BYOL + Linear** | 0.999 | ~98.0% |
| **BYOL + Linear** | 0.980 | ~98.22% |
| **Supervised** | - | ~99.39% |