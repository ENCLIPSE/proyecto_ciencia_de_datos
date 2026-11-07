# Problema de ML y decisiones de modelado

| Decisión | Tu respuesta |
|---|---|
| Tipo de problema | Clasificación binaria supervisada |
| Variable objetivo | `Survived` (1 = sobrevivió) |
| Unidad y momento de predicción | Un pasajero, con la información disponible al embarcar |
| Métrica principal | ROC-AUC: mide la capacidad de ordenar positivos por encima de negativos y no depende del umbral |
| Métricas secundarias | F1, precisión y recall con umbral 0,5; exactitud se interpreta junto con el baseline |
| Costo de errores | En un ejercicio de riesgo, falso positivo y falso negativo pueden tener costos diferentes; no hay una función de costos definida y este modelo no debe decidir cobertura, precio ni acceso a servicios |
| Criterio de éxito | Superar el baseline Dummy (ROC-AUC 0,50; exactitud ≈ 0,617 en entrenamiento) y contrastar el resultado con la regla de género (exactitud 0,788; F1 0,715 en el conjunto de entrenamiento) |

## Selección de modelos

Se reservó el 20 % de los 891 casos etiquetados como prueba (179 pasajeros; semilla fija 2026). La comparación de modelos y el ajuste se hicieron solo en los 712 casos de entrenamiento, con validación cruzada estratificada de 5 pliegues (semilla 42). La métrica para comparar fue ROC-AUC.

| Modelo (parámetros por defecto) | ROC-AUC CV media ± desv. | Brecha entrenamiento − validación | Lectura |
|---|---:|---:|---|
| HistGradientBoosting | 0,851 ± 0,014 | 0,141 | Mejor AUC entre los candidatos por defecto junto con logística; brecha alta |
| Regresión logística | 0,851 ± 0,026 | 0,016 | Rendimiento similar, más estable e interpretable |
| Random Forest | 0,846 ± 0,023 | 0,152 | Brecha alta; mayor costo que los modelos lineales |
| Árbol de decisión | 0,845 ± 0,027 | 0,054 | Fácil de explicar, pero menor AUC |
| K vecinos (KNN) | 0,836 ± 0,014 | 0,095 | Menor AUC que los principales candidatos |
| Dummy (clase mayoritaria) | 0,500 ± 0,000 | 0,000 | Referencia sin capacidad de ordenamiento |

Se eligió HistGradientBoosting para el ejercicio. La búsqueda en rejilla sobre el entrenamiento seleccionó `learning_rate=0.03`, `max_depth=2`, `max_iter=100` y `l2_regularization=0.0` (ROC-AUC CV 0,872 ± 0,028). Este puntaje proviene de la búsqueda y puede ser optimista; no se presenta como estimación final. Se conservó `Sex` por su valor predictivo histórico, pero se reconoce como variable sensible. Se excluyeron las variables de cabina: añadirlas cambió el ROC-AUC CV de 0,8717 a 0,8731, una diferencia de 0,0014 frente a desviaciones cercanas a 0,03, insuficiente para justificar el riesgo de artefacto de registro.

El umbral se dejó en 0,5 como referencia fija, no como umbral óptimo por costos: el objetivo principal es ROC-AUC y no se definió una función de costos con el usuario o dominio.

## Evaluación final

Después de congelar las decisiones se evaluó una sola vez en la partición reservada. HistGradientBoosting obtuvo exactitud **0,832**, F1 **0,758** y ROC-AUC **0,846** (n = 179). El IC bootstrap al 95 % del ROC-AUC fue **[0,777; 0,909]** (1.000 remuestreos). El intervalo amplio refleja la incertidumbre de una prueba pequeña. Tras esta evaluación, el pipeline definitivo se ajustó con los 891 casos etiquetados y se guardó junto a sus metadatos.

Las métricas por subgrupo se inspeccionaron con predicciones fuera de pliegue sobre entrenamiento, no sobre prueba. Con umbral 0,5, el recall fue 0,915 para mujeres (n = 256) y 0,107 para hombres (n = 456); por clase fue 0,691 en primera (n = 172), 0,871 en segunda (n = 140) y 0,515 en tercera (n = 400). Estas diferencias muestran que el rendimiento agregado no describe igual a todos los grupos y no constituyen una evaluación de equidad.

## Limitaciones y uso previsto

- El Titanic es un único evento histórico; 891 registros etiquetados no representan a otros naufragios, poblaciones ni decisiones actuales.
- Faltan 177 edades, 687 cabinas y 2 puertos de embarque. La ausencia puede ser sistemática; las imputaciones no recuperan información que no se observó.
- Sexo y clase son variables sensibles o vinculadas a condiciones sociales e históricas. El modelo aprende patrones del registro y de las decisiones de evacuación de 1912, no efectos causales ni criterios justos.
- El ROC-AUC tiene un intervalo amplio y el umbral 0,5 no se deriva de costos reales. Las métricas por subgrupo también son inciertas, especialmente para grupos pequeños.
- Uso previsto: taller educativo, reproducibilidad y análisis histórico. No usar para suscripción de seguros, fijación de precios, asignación de recursos, seguridad, elegibilidad ni decisiones individuales.