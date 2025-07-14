# 🤖 Agente de IA para Gobierno de Datos en BigQuery

Este proyecto implementa un **Agente de Inteligencia Artificial** capaz de responder preguntas técnicas sobre los datasets, tablas, rutinas y esquemas de datos almacenados en **Google BigQuery**, mediante técnicas de **RAG (Retrieval-Augmented Generation)**.

---

## 🧠 Arquitectura General

- **Frontend**: Aplicación React desplegada en **Vercel**  
- **Backend**: API en Python con Flask desplegada en **Cloud Run**
- **Vector Store**: Embeddings almacenados en **Elasticsearch**
- **Motor de IA**: Agente basado en **LangChain + OpenAI GPT-4**
- **Origen de datos**: **BigQuery** (esquemas y rutinas)
- **Trazabilidad**: **LangSmith**

---

## 📁 Estructura del Repositorio

/frontend # App React (Vercel)

└── src/

└── components/

└── pages/

└── utils/

└── services/


/backend # Agente Flask + LangChain (Cloud Run)

└── app.py

└── Dockerfile

└── requirements.txt

└── tools/ # Herramientas personalizadas

└── embeddings/ # Scripts para chunking y carga

/APP_DATA_CARGA_BIGQUERY_PROYECTO.ipynb

/README.md

---

## 🚀 Frontend (React + Vercel)

### Funcionalidad

- Interfaz amigable para consultas al agente
- Envío de mensajes al backend y visualización de respuestas
- Visualización enriquecida de metadatos de tablas y rutinas

### Tecnologías

- React
- Axios
- TailwindCSS
- Desplegado en Vercel

---

##⚙️ Backend (Flask + LangChain + BigQuery)
### Funcionalidad
- API REST con endpoint /agent

- Integra herramientas para consultar BigQuery:

 * Esquema de tablas
 * Tamaño, filas y descripción
 * Rutinas asociadas
 * Información de particionado y clustering
 * Lógica de deduplicación y origen/destino

- Búsqueda semántica con embeddings almacenados en Elasticsearch

### Herramientas personalizadas (@tool)
*  obtener_esquema_tabla(tabla_id)
*  listar_tablas(dataset)
*  info_tabla(tabla_id)
*  listar_rutinas(dataset)
*  particionamiento_tabla(tabla_id)
*  obtener_procedimiento_asociado(tabla_id)
*  sugerir_select_bigquery(query)


---

##Tecnologías
Python 3.9

Flask

LangChain / LangGraph

OpenAI API

Google BigQuery

Elasticsearch

PostgreSQL (para trazabilidad con LangSmith)

---

##🧠 Embeddings y Chunking
Los scripts de embeddings analizan:

Columnas de tablas (esquema, descripción, tags)

Rutinas (bloques lógicos: inicio, carga, origen, filtros, deduplicación)

Comentarios técnicos (desarrollador, caso de uso, versión, etc.)

La información se almacena como vectores en Elasticsearch para recuperación semántica posterior.

---

##🌐 Endpoint de la API

GET /agent?idagente=<thread_id>&msg=<mensaje>

Ejemplo:

/agent?idagente=usuario123&msg=¿Cuál es el esquema de la tabla DS_XYZ.CUSTOMERS?