# Detección de Fraude en Canastas de Compra

![Banner](assets/banner_fraude.png)

> Modelo de clasificación supervisada que identifica transacciones fraudulentas en canastas de compra usando Árbol de Decisión y Bosque Aleatorio, con optimización del umbral de decisión para minimizar los fraudes no detectados.

---

## Resultado

| Modelo | Exactitud | Observación |
|---|---|---|
| Árbol de Decisión (sin límite) | 88.41% | Profundidad 57 — sobreajuste |
| **Árbol de Decisión (profundidad óptima)** | **88.41%** | Mejor balance precisión/generalización |
| **Bosque Aleatorio (1000 árboles)** | **Superior** | Más robusto y estable |

**Con umbral optimizado a 0.30:** los fraudes detectados aumentaron de 47 a 118 — más del doble — reduciendo los fraudes no detectados de 197 a 126.

---

## Tecnologías

| Herramienta | Uso |
|---|---|
| Scikit-learn | DecisionTreeClassifier, RandomForestClassifier, métricas |
| Pandas / NumPy | Exploración, limpieza y manipulación de datos |
| Matplotlib / Seaborn | Visualización de resultados y matrices de confusión |

---

---

## Tabla de Contenidos

- [1. Identificación del Problema](#1-identificación-del-problema)
- [2. Los Datos — Cómo se utilizan para resolver el problema](#2-los-datos--cómo-se-utilizan-para-resolver-el-problema)
- [3. Conclusiones Finales](#3-conclusiones-finales)

---

## 1. Identificación del Problema

### ¿Qué problema existe en el mundo real?

Cada vez que alguien hace una compra en línea, hay riesgo de fraude. Una transacción fraudulenta que pasa desapercibida representa una pérdida directa de dinero. Pero bloquear demasiadas transacciones legítimas también tiene un costo — clientes frustrados y ventas perdidas.

El reto no es solo detectar fraudes. Es detectarlos **sin afectar a los clientes honestos**.

Este proyecto responde una pregunta concreta:

**¿Podemos predecir si una transacción es fraudulenta a partir de las características de la canasta de compra?**

### ¿Por qué es un problema difícil?

Hay dos razones principales:

**1. Los datos están desbalanceados.** En cualquier sistema real, la gran mayoría de transacciones son legítimas. Un modelo que simplemente diga "todo es legítimo" tendría alta exactitud, pero sería completamente inútil para detectar fraudes.

**2. Los errores no tienen el mismo costo.** Hay dos tipos de error posibles:

| Tipo de error | Qué significa | Consecuencia |
|---|---|---|
| **Falso Negativo (FN)** | Fraude real que el modelo no detectó | Pérdida económica directa — el más grave |
| **Falso Positivo (FP)** | Transacción legítima bloqueada | Cliente insatisfecho |

Este proyecto prioriza reducir los **Falsos Negativos** — es decir, que el modelo no deje pasar fraudes sin detectar.

### ¿Qué propone este proyecto?

Comparar dos modelos y optimizar el umbral de clasificación:

- **Árbol de Decisión** — modelo interpretable. Funciona como un diagrama de flujo: hace preguntas sobre la transacción y llega a una decisión. Fácil de auditar y explicar.
- **Bosque Aleatorio** — construye 1000 árboles de decisión distintos y decide por mayoría de votos. Más robusto, menos propenso al sobreajuste.

> **Ejemplo que lo hace concreto:** Un árbol de decisión pregunta: ¿el costo de cumplimiento supera cierto valor? ¿el producto es de Apple? ¿el costo total es alto? Dependiendo de las respuestas, clasifica la transacción como fraude o no. El Bosque Aleatorio hace eso mismo 1000 veces con variaciones y promedia el resultado.

---

## 2. Los Datos — Cómo se utilizan para resolver el problema

### ¿Qué datos se usan?

El dataset `FraudeCanastas.csv` contiene transacciones de canastas de compra con una variable objetivo binaria:

- `fraud_flag = 1` → transacción fraudulenta
- `fraud_flag = 0` → transacción legítima

El dataset es **desbalanceado** — los fraudes son minoría, igual que en cualquier sistema real.

### ¿Cómo se limpian los datos?

Antes de entrenar el modelo, los datos pasan por dos pasos de limpieza:

**Paso 1 — Eliminación de columnas poco informativas**
Se identifican columnas con exactamente 2 valores únicos (excepto `fraud_flag`) y la columna `ID`. Estas variables no aportan poder predictivo y solo añaden ruido al modelo.

**Paso 2 — Exploración estadística**
Se construye un resumen completo del dataset: tipos de dato, porcentaje de nulos, media, mediana, desviación estándar, mínimo y máximo por columna. Esto permite entender la distribución de los datos antes de modelar.

### ¿Qué variables predicen mejor el fraude?

El análisis de importancia de características identifica las variables más influyentes:

| Ranking | Variable | Por qué importa |
|---|---|---|
| 1 | `FULFILMENT CHARGE | RETAILER` | El costo de cumplimiento de pedidos es la señal más fuerte de fraude |
| 2 | `costo_total` | Transacciones de alto costo total son más sospechosas |
| 3 | `costo_medio / costo_max / costo_min` | El perfil completo de costos del pedido |
| 4 | Productos Apple y Samsung | Dispositivos de alto valor aparecen frecuentemente en fraudes |

### ¿Cómo usa el modelo estos datos?

**División del dataset**
80% para entrenar el modelo, 20% para evaluar. Se mantiene la proporción de fraude en ambos conjuntos (`shuffle=True`, `random_state=8`).

**Entrenamiento del Árbol de Decisión**
Primero sin límite de profundidad — el árbol crece hasta 57 niveles, lo que indica sobreajuste (memoriza los datos de entrenamiento). Luego se prueba con profundidades de 3 a 18 y se selecciona la óptima.

**Optimización del umbral**
Por defecto, los modelos clasifican como fraude cuando la probabilidad supera 0.50. Este proyecto prueba un umbral de **0.30** — más sensible, detecta más fraudes a costa de algunos falsos positivos.

| Umbral | Fraudes detectados (TP) | Fraudes no detectados (FN) |
|---|---|---|
| 0.50 (por defecto) | 47 | 197 |
| **0.30 (optimizado)** | **118** | **126** |

Bajar el umbral de 0.50 a 0.30 más que **duplicó** los fraudes detectados.

### Flujo del proyecto (pipeline)

> **Pipeline** es la secuencia ordenada de pasos que siguen los datos desde que entran hasta que se obtiene el resultado final.

```
1. Carga y exploración del dataset
        ↓
2. Limpieza — eliminación de columnas irrelevantes y variables con 2 valores únicos
        ↓
3. Análisis estadístico — resumen de tipos, nulos y distribuciones
        ↓
4. División entrenamiento / prueba (80% / 20%)
        ↓
5. Árbol de Decisión sin límite de profundidad (baseline)
        ↓
6. Búsqueda de profundidad óptima (rango 3–18)
        ↓
7. Análisis de importancia de características (Top 20)
        ↓
8. Optimización del umbral de clasificación (0.50 → 0.30)
        ↓
9. Bosque Aleatorio con 1000 árboles
        ↓
10. Comparación final con Matriz de Confusión y curva ROC
```

### ¿Cómo se mide el resultado?

**Matriz de Confusión** — descompone los resultados en 4 categorías:

| | Predicción: No Fraude | Predicción: Fraude |
|---|---|---|
| **Real: No Fraude** | ✅ Verdadero Negativo (TN) — correctamente identificado como legítimo | ⚠️ Falso Positivo (FP) — legítimo bloqueado |
| **Real: Fraude** | 🚨 Falso Negativo (FN) — fraude no detectado | ✅ Verdadero Positivo (TP) — fraude correctamente detectado |

El objetivo es maximizar los **TP** y minimizar los **FN** — detectar el mayor número posible de fraudes reales.

---

## 3. Conclusiones Finales

### ¿Qué funcionó?

**El umbral optimizado fue el hallazgo más importante.** Pasar de 0.50 a 0.30 duplicó los fraudes detectados sin requerir cambios en el modelo. Esto demuestra que en problemas de detección de fraude, la configuración del umbral es tan crítica como el algoritmo elegido.

**El Bosque Aleatorio superó al Árbol de Decisión** en robustez y estabilidad. Al combinar 1000 árboles independientes, elimina el sobreajuste que presentó el árbol individual con profundidad 57.

**La variable `FULFILMENT CHARGE | RETAILER`** fue consistentemente la más importante en ambos modelos. Esto sugiere que el costo de cumplimiento tiene un patrón claro en las transacciones fraudulentas — información valiosa para el negocio más allá del modelo.

### ¿Qué aprendimos sobre el problema?

La exactitud sola no es una métrica suficiente para evaluar modelos de fraude. Un modelo con 92% de exactitud puede estar fallando en exactamente los casos más costosos — los fraudes. Las métricas correctas son Recall y la distribución de la Matriz de Confusión.

### ¿Qué limitaciones tiene?

- El dataset está desbalanceado. Técnicas como SMOTE (generación de muestras sintéticas de la clase minoritaria) podrían mejorar la detección.
- El Árbol sin límite de profundidad (57 niveles) presenta sobreajuste claro — 92.18% en entrenamiento vs 88.41% en prueba.
- No se aplicó validación cruzada — añadirla daría estimaciones más confiables del rendimiento real.

### ¿Qué sigue?

Los próximos pasos naturales son aplicar técnicas de balanceo de clases (SMOTE), probar modelos como XGBoost o LightGBM diseñados para datos desbalanceados, e implementar validación cruzada estratificada para una evaluación más robusta.

---

## Autor

Proyecto desarrollado como parte de un curso de **Machine Learning & AI**.
Si tienes preguntas o sugerencias, puedes abrir un *issue* en este repositorio.
