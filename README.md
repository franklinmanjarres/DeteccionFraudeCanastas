## DeteccionFraudeCanastas

Este proyecto implementa modelos de aprendizaje automático para detectar fraude en transacciones de compra, utilizando técnicas como Árboles de Decisión y Random Forest. El análisis se basa en un conjunto de datos reales, logrando resultados con más del **88% de precisión** en la clasificación de operaciones fraudulentas.

---

###  Objetivo del Proyecto

Desarrollar un modelo capaz de identificar canastas de compra potencialmente fraudulentas, basándose en atributos transaccionales y características del comportamiento del cliente.

---

###  Modelos utilizados

- **Árboles de Decisión (Decision Tree Classifier)**: Modelo base que permite interpretar fácilmente las reglas de clasificación.
- **Random Forest**: Modelo de ensamble que mejora la precisión y generalización al combinar múltiples árboles.

---

###  Metodología

- Análisis exploratorio del conjunto de datos.
- Limpieza y codificación de variables categóricas.
- Balanceo de clases si es necesario.
- Entrenamiento y validación cruzada.
- Evaluación del desempeño con métricas como:
  - Precisión (Accuracy)
  - Matriz de confusión
  - Curva ROC y AUC

---

###  Resultados

- Precisión obtenida: **≈ 88%**
- Random Forest mostró mejor desempeño que el árbol de decisión simple, con menor sobreajuste y mejor capacidad predictiva.

---

### Tecnologías usadas

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook
