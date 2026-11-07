# Conclusiones del EDA

## 1. Hallazgos
- La tasa general de supervivencia fue **38,4 %** (342 de 891 pasajeros).
- La diferencia por sexo fue marcada: sobrevivió el **74,2 % de las mujeres** (233/314) y el **18,9 % de los hombres** (109/577). La prueba chi-cuadrado encontró asociación entre sexo y supervivencia (χ² = 260,72; gl = 1; p = 1,20 × 10⁻⁵⁸); esto no demuestra causalidad.
- La supervivencia descendió por clase: **63,0 %** en primera (136/216), **47,3 %** en segunda (87/184) y **24,2 %** en tercera (119/491).
- Sobrevivió el **50,6 %** de quienes viajaban con familiares (179/354), frente al **30,4 %** de quienes viajaban solos (163/537). La asociación no es uniforme por tamaño familiar: los grupos grandes son pequeños y sus tasas son inestables.
- El puerto C tuvo una tasa agregada de **55,4 %** (93/168), frente a **33,9 %** en S (219/646) y **39,0 %** en Q (30/77). La composición por clase difiere entre puertos, por lo que la comparación agregada no debe interpretarse como efecto del puerto.

## 2. Decisiones de limpieza y su sesgo
- **Edad:** hay 177 valores faltantes. Para el EDA se imputaron con la media por sexo; la desviación estándar bajó de 14,53 a 13,01 años. La imputación concentra artificialmente observaciones en la media y reduce la dispersión. Se conserva `AgeOriginal` para distinguir mediciones de imputaciones y el modelo vuelve a imputar dentro del pipeline de entrenamiento.
- **Puerto:** faltaban 2 valores de `Embarked`. Para el EDA se rellenan con la moda y se conserva `EmbarkedOriginal`; el modelo usa esta última columna, con sus faltantes, para que la imputación se ajuste solo en entrenamiento.
- **Títulos:** `Master` se agrupa con `Mr` (40 pasajeros originalmente). Es una simplificación defendible para evitar una categoría pequeña, pero mezcla niños varones con adultos. Edad y `Person` ayudan a conservar esa distinción durante el análisis; la agrupación no debe confundirse con equivalencia real.
- **Cabina:** 687 de 891 pasajeros (77,1 %) no tienen cabina registrada; se representan con `X` para no perder la ausencia. La supervivencia fue 66,7 % entre quienes tienen cabina registrada (136/204) y 30,0 % entre quienes no (206/687). Esto puede reflejar diferencias de registro o de clase y no prueba que tener una cabina causara mayor supervivencia.
- **Edad desconocida y `Person`:** la regla actual asigna `female` o `male` si falta la edad, por lo que puede dejar niños sin identificar. Entre las 714 personas con edad conocida, 83 (11,6 %) tienen menos de 16 años. Si se extrapola esa proporción a los 177 casos sin edad, se estiman aproximadamente **21 niños**; es una estimación simple, no un conteo observado.

## 3. Confusiones detectadas
- **Puerto y clase:** C concentra 85 pasajeros de primera clase de 168, mientras S tiene 129 de primera entre 646 y 353 de tercera. Al estratificar por clase, las tasas de C y S son, respectivamente, 69,4 % y 58,9 % en primera (n = 85 y 129), 52,9 % y 46,3 % en segunda (n = 17 y 164), y 37,9 % y 19,0 % en tercera (n = 66 y 353). La diferencia agregada se reduce al comparar grupos semejantes, pero la estratificación observacional no elimina todos los posibles factores de confusión. Q tiene solo 2 pasajeros de primera y 3 de segunda; sus tasas por clase son especialmente inciertas.

## 4. Qué NO podemos afirmar
- Estos datos describen un solo naufragio histórico; no permiten concluir que sexo, clase, puerto o composición familiar causaran la supervivencia ni generalizar directamente a otros desastres.
- Los datos están incompletos y algunas ausencias (en especial cabina y edad) pueden ser sistemáticas. Las tasas de grupos pequeños tienen alta incertidumbre; deben leerse junto con sus tamaños de muestra.