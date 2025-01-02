import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
DATOSCRUDOS = pd.read_csv("housing.csv")
DATOSCRUDOSCOPY = DATOSCRUDOS.copy()
print(DATOSCRUDOSCOPY.info())
print(DATOSCRUDOSCOPY.describe())
print(DATOSCRUDOSCOPY.isnull().sum())
data_cleaned = DATOSCRUDOSCOPY.dropna()
print(data_cleaned.isnull().sum())


# Histograma de precios
data_cleaned['median_house_value'].hist(bins=30, figsize=(10, 6))
plt.title("Distribución de los Precios de las Viviendas")
plt.xlabel("Precio Medio")
plt.ylabel("Frecuencia")
plt.show()


# Filtrar solo columnas numéricas
numeric_data = data_cleaned.select_dtypes(include=['float64', 'int64'])

# Comprobar las columnas numéricas seleccionadas
print(numeric_data.columns)

# Calcular la matriz de correlación
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()


proximidad_ocenano_hogares = DATOSCRUDOSCOPY.groupby(by="ocean_proximity")['households'].sum().reset_index(name='num_hogares')
print(proximidad_ocenano_hogares)

## Trabajaremos con el DataFrame Copia
## Podemos ver numeros de registro de hogares
plt.figure(figsize=(10, 6))
bars = plt.bar(proximidad_ocenano_hogares['ocean_proximity'], proximidad_ocenano_hogares['num_hogares'])
plt.title('Número total de hogares por proximidad al océano', pad=20)
plt.xlabel('Proximidad al océano')
plt.ylabel('Número total de hogares')
plt.xticks(rotation=45)

# Añadir valores sobre las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom')

# Ajustar diseño
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Gráfico 1: Valor vs Ingresos
sns.scatterplot(data=DATOSCRUDOSCOPY, x='median_income', y='median_house_value', ax=axes[0,0])
axes[0,0].set_title('Valor vs Ingresos Medianos')

# Gráfico 2: Valor vs Edad
sns.scatterplot(data=DATOSCRUDOSCOPY, x='housing_median_age', y='median_house_value', ax=axes[0,1])
axes[0,1].set_title('Valor vs Edad de la Casa')

# Gráfico 3: Valor por ubicación
sns.boxplot(data=DATOSCRUDOSCOPY, x='ocean_proximity', y='median_house_value', ax=axes[1,0])
axes[1,0].set_title('Valor por Ubicación')
axes[1,0].tick_params(axis='x', rotation=45)

# Gráfico 4: Valor vs Habitaciones
sns.scatterplot(data=DATOSCRUDOSCOPY, x='total_rooms', y='median_house_value', ax=axes[1,1])
axes[1,1].set_title('Valor vs Habitaciones Totales')

plt.tight_layout()
plt.show()


## Análisis de Clústeres
from sklearn.cluster import KMeans

# Seleccionar variables relevantes
features = numeric_data[["latitude", "longitude", "median_house_value"]]

# Realizar clustering
kmeans = KMeans(n_clusters=3, random_state=42)
numeric_data["cluster"] = kmeans.fit_predict(features)

# Visualizar los clústeres
plt.scatter(numeric_data["longitude"], numeric_data["latitude"], c=numeric_data["cluster"], cmap="viridis")
plt.title("Clústeres de Viviendas")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.colorbar()
plt.show()
