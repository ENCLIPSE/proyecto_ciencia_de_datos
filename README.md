# Proyecto de ciencia de datos e IA

Taller de análisis exploratorio y clasificación con el conjunto de datos del Titanic. El pipeline procesa los datos en capas Bronze, Silver y Gold, ejecuta pruebas y guarda un modelo con metadatos.

## Inicio rápido

Requiere Python 3.10 o superior y `make`.

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make test
make pipeline
```

Consulta [la guía del taller](docs/GUIA.md) para el desarrollo paso a paso y [la guía de instalación](docs/INSTALACION.md) para la configuración del entorno.

El pipeline usa una URL predeterminada, configurable en `.env` a partir de `.env.example`. Si la fuente remota no está disponible, configura `TITANIC_LOCAL_PATH` en ese archivo local. No subas `.env` ni los datos o modelos generados.

Resultados y decisiones: [conclusiones del EDA](reports/conclusiones_eda.md), [problema y evaluación de ML](reports/problema_ml.md) y [ficha del modelo](reports/model_card.md).