# Proyecto de Ciencia de Datos e IA

**Autor:** Enzo Gonzalez Caicedo

## Objetivo del proyecto

Realizar un análisis exploratorio de datos (EDA) del conjunto de pasajeros del Titanic e implementar un flujo reproducible de ciencia de datos que prepare los datos, analice patrones de supervivencia y entrene un modelo de clasificación.

El proyecto busca responder dos preguntas:

1. **EDA:** ¿Cómo se relacionan características como el sexo, la clase, la edad y la composición familiar con la supervivencia observada?
2. **Machine learning:** ¿Qué tan bien puede un modelo estimar la supervivencia a partir de información disponible sobre un pasajero?

Es un ejercicio educativo y de análisis histórico. Las asociaciones observadas no demuestran causalidad, y el modelo no debe utilizarse para tomar decisiones actuales sobre personas.

## Alcance

- Ingerir y conservar los datos originales en la capa **Bronze**, junto con metadatos y un reporte de calidad.
- Limpiar, validar e incorporar variables derivadas en **Silver**.
- Generar tablas descriptivas y estadísticas en **Gold**.
- Explorar los datos y documentar hallazgos, faltantes, sesgos y limitaciones.
- Comparar modelos con validación cruzada, evitar fuga de datos usando `Pipeline` y evaluar una sola vez en una partición de prueba reservada.
- Guardar el modelo final y sus metadatos para poder auditar y reproducir el resultado.

## Tecnologías

Python 3.10 o superior, pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, PyArrow, Joblib, pytest y Ruff. El proyecto utiliza notebooks para exploración y módulos Python probados para el pipeline productivo.

## Requisitos e instalación

Se requiere Python 3.10+ y `make`.

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
```

En Windows, activa el entorno virtual con `.venv\Scripts\activate`.

La fuente de datos tiene una URL predeterminada. Si necesitas cambiarla o usar un archivo local, copia `.env.example` a `.env` y ajusta sus variables:

```bash
cp .env.example .env
```

`TITANIC_URL` permite indicar otra fuente y `TITANIC_LOCAL_PATH` una ruta local. No compartas ni subas `.env`.

## Ejecución

Ejecuta el pipeline completo:

```bash
make pipeline
```

Las etapas también se pueden ejecutar por separado:

```bash
make ingest   # fuente -> data/bronze
make silver   # Bronze -> Silver
make gold     # Silver -> Gold
make train    # evaluación y modelo final
```

Verifica el proyecto con:

```bash
make test
make lint
```

Los datos y modelos generados se guardan en `data/` y `models/`; estos artefactos locales no se versionan.

## Estructura principal

```text
flows/       Pipeline de ingesta, transformación, agregación y modelado
tests/       Pruebas automáticas de las etapas
notebooks/   Exploración y experimentos de modelado
data/        Capas Bronze, Silver y Gold generadas localmente
models/      Modelo entrenado y metadatos generados localmente
reports/     Conclusiones, decisiones de modelado y ficha del modelo
docs/        Guías de instalación y desarrollo del taller
```

## Documentación y resultados

- [Guía del taller](docs/GUIA.md)
- [Instalación y configuración](docs/INSTALACION.md)
- [Conclusiones del análisis exploratorio](reports/conclusiones_eda.md)
- [Problema y evaluación de machine learning](reports/problema_ml.md)
- [Ficha del modelo](reports/model_card.md)
