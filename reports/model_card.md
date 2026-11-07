# Ficha del modelo: Titanic

## Identificación y uso previsto

- **Artefacto:** `models/titanic_model_v1.joblib`
- **Tarea:** clasificación binaria para estimar la probabilidad histórica de supervivencia.
- **Uso previsto:** formación, análisis histórico y demostración reproducible de un pipeline de ciencia de datos.
- **Usos no recomendados:** seguros, precios, selección de personas, asignación de recursos, seguridad o cualquier decisión individual con consecuencias reales.

## Datos y método

- **Fuente:** [dataset del Titanic configurado para la ingesta](https://paste.c-net.org/MarvelJinxed); 891 pasajeros etiquetados y 418 sin etiqueta.
- **Fecha de ingesta:** 2026-11-07.
- **Huella SHA-256 de Silver usada para entrenar:** `deb1826b83b79399f2df484be78c8767e5edcc6d8952cd8c550ba304b2c5bcba`.
- **Tamaño:** la evaluación reservó 179 registros (20 %); después de evaluar, el artefacto final se reajustó con los 891 registros etiquetados.
- **Objetivo:** `Survived`.
- **Variables usadas:** `Age`, `SibSp`, `Parch`, `Fare`, `Familia`, `Pclass`, `Sex`, `Embarked`, `TTitulo` y `Alone`.
- **Variables excluidas:** identificadores y texto crudo (`PassengerId`, `Ticket`, `Name`), el objetivo y derivados (`Survived`, `Survivor`), codificaciones ordinales (`NSex`, `NEmbarked`, `NCabin`, `NTitulo`), `Person` y las variables de cabina (`HasCabin`, `TCabin`).
- **Preprocesamiento:** imputación de edad por sexo y clase dentro del pipeline; edad y puerto originales con faltantes; imputación y codificación categórica aprendidas dentro de cada ajuste, sin calcularlas sobre la partición de prueba.
- **Estimador:** `HistGradientBoostingClassifier`, `learning_rate=0.03`, `max_depth=2`, `max_iter=100`, `l2_regularization=0.0`. Umbral de etiqueta: 0,5.
- **Selección:** validación cruzada estratificada de 5 pliegues en el conjunto de entrenamiento (712 casos; semilla 42). El mejor ROC-AUC de rejilla fue 0,872 ± 0,028; al ser el mejor resultado de una búsqueda, puede ser optimista.

## Resultados

La partición reservada se fijó con semilla 2026 y se evaluó una sola vez después de congelar las decisiones:

| Métrica de prueba | Resultado |
|---|---:|
| Exactitud | 0,832 |
| F1 (umbral 0,5) | 0,758 |
| ROC-AUC | 0,846 |
| IC bootstrap 95 % del ROC-AUC (1.000 remuestreos) | [0,777; 0,909] |

Las predicciones fuera de pliegue de entrenamiento muestran diferencias de recall: 0,915 para mujeres (n = 256) y 0,107 para hombres (n = 456); por clase: 0,691 en primera, 0,871 en segunda y 0,515 en tercera. Son análisis exploratorios, no una certificación de equidad, y no reemplazan la evaluación final.

## Limitaciones y riesgos

El conjunto representa un único naufragio histórico y contiene faltantes importantes, en especial en edad y cabina. Sexo y clase reflejan tanto condiciones sociales como decisiones de evacuación de 1912; una buena predicción no demuestra causalidad ni justicia. La prueba es pequeña y su intervalo de confianza es amplio. El umbral 0,5 es una convención: no se dispone de una función real de costos para justificarlo. No usar el modelo para tomar decisiones actuales sobre personas.

## Reproducibilidad

Versiones al guardar el modelo: Python 3.10.12, scikit-learn 1.7.2 y pandas 2.3.3. Desde el entorno del proyecto, ejecutar `make test` y `make pipeline`. La configuración concreta y la huella del archivo del modelo están en `models/titanic_model_v1.metadata.json`.
