from langchain_openai import ChatOpenAI
import os
from flask import Flask, jsonify, request
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver, PostgresCheckpointer
from langgraph.prebuilt import create_react_agent
from google.cloud import bigquery
from langchain.agents import tool
from elasticsearch import Elasticsearch


## datos de trazabilidad
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "clave"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "gcpprojectbootcampdemo"
os.environ["OPENAI_API_KEY"] ="clave"


app = Flask(__name__)

@app.route('/agent', methods=['GET'])
def main():
    #Capturamos variables enviadas
    id_agente = request.args.get('idagente')
    msg = request.args.get('msg')
    #datos de configuracion
    DB_URI = os.environ.get(
        "DB_URI",
        "postgresql://clave"
    )
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    db_query = ElasticsearchStore(
        es_url="ip",
        es_user="elastic",
        es_password="clave",
        index_name="lg-proyectobigquery",
        embedding=OpenAIEmbeddings())

    # Herramienta RAG

    @tool
    def obtener_esquema_tabla(tabla_id: str) -> dict:
        """
        Recupera un resumen del esquema de una tabla de BigQuery.
        Incluye tipo, modo, descripción y etiqueta de política (policy tag) si existe.
        Muestra solo columnas relevantes en el resumen.
        Formato esperado: 'proyecto.dataset.tabla'
        """
        try:
            client = bigquery.Client(project="banded-badge-465303-k1")

            # Obtener metadata de la tabla
            tabla = client.get_table(tabla_id)

            columnas_completas = []
            columnas_relevantes = []

            for campo in tabla.schema:
                columna = {
                    "name": campo.name,
                    "type": campo.field_type,
                    "mode": campo.mode,
                    "description": campo.description or "",
                }

                # Añadir policy tag si existe
                if campo.policy_tags:
                    policy_tags = campo.policy_tags.names  # Lista de nombres
                    columna["policy_tags"] = policy_tags
                else:
                    columna["policy_tags"] = []

                columnas_completas.append(columna)

                # Heurística de relevancia para resumen
                if (
                    columna["description"]
                    or columna["policy_tags"]
                    or any(word in campo.name.lower() for word in ["id", "fecha", "date", "email", "monto", "amount", "user"])
                ):
                    columnas_relevantes.append(columna)

            resumen = {
                "tabla": tabla_id,
                "total_columnas": len(columnas_completas),
                "columnas_relevantes": columnas_relevantes[:10],  # top 10 para evitar desbordes
            }

            return resumen

        except Exception as e:
            return {"error": str(e)}

    @tool
    def sugerir_select_bigquery(query: str) -> str:
        """Analiza una consulta SQL y sugiere el SELECT principal que debería ejecutarse."""
        try:
            query_upper = query.strip().upper()

            # Si ya inicia con SELECT
            if query_upper.startswith("SELECT"):
                return f"Consulta sugerida: {query}"

            # Buscar SELECT principal en consultas complejas
            select_start = query_upper.find("SELECT")
            if select_start != -1:
                select_query = query[select_start:].strip()
                return f"Consulta sugerida: {select_query}"

            return "No se pudo identificar una instrucción SELECT en la consulta proporcionada."

        except Exception as e:
            return f"Error analizando la consulta: {str(e)}"

    @tool
    def listar_tablas(dataset: str) -> str:
        """Lista todas las tablas dentro de un dataset."""
        client = bigquery.Client(project="banded-badge-465303-k1")
        try:
            tables = client.list_tables(dataset)
            return "\n".join([table.table_id for table in tables])
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def info_tabla(tabla_id: str) -> str:
        """Devuelve tamaño, filas y descripción de una tabla de BigQuery."""
        client = bigquery.Client(project="banded-badge-465303-k1")
        try:
            table = client.get_table(tabla_id)
            return (
                f"Descripción: {table.description}\n"
                f"Filas: {table.num_rows}\n"
                f"Tamaño (bytes): {table.num_bytes}\n"
                f"Esquema: {[f.name for f in table.schema]}"
            )
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def listar_rutinas(dataset: str) -> str:
        """Devuelve las rutinas (UDFs o stored procedures) en un dataset."""
        client = bigquery.Client(project="banded-badge-465303-k1")
        try:
            routines = client.list_routines(dataset)
            return "\n".join([r.name for r in routines])
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def particionamiento_tabla(tabla_id: str) -> str:
        """Devuelve información de partición y clustering de una tabla."""
        client = bigquery.Client(project="banded-badge-465303-k1")
        try:
            table = client.get_table(tabla_id)
            part = table.time_partitioning
            clust = table.clustering_fields
            return f"Particionado por: {part.field if part else 'Ninguno'}\nClustering: {clust if clust else 'Ninguno'}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def obtener_procedimiento_asociado(tabla_id: str) -> dict:
        """
        Recupera el código del procedimiento almacenado asociado a una tabla de BigQuery.
        Si no lo encuentra y el dataset comienza con 'DS_RDV', asume que es una tabla de carga inicial.

        Args:
            tabla_id (str): Formato esperado 'proyecto.dataset.tabla'

        Returns:
            dict: Información del procedimiento o indicación de que es tabla de streaming.
        """
        try:
            # Inicializa el cliente con project explícito si lo deseas
            client = bigquery.Client(project="banded-badge-465303-k1")

            # Parsear tabla_id
            partes = tabla_id.split(".")
            if len(partes) != 3:
                return {"error": "Formato inválido. Usa: proyecto.dataset.tabla"}

            proyecto, dataset, nombre_tabla = partes
            nombre_procedimiento = f"USP_{nombre_tabla}"

            # Buscar procedimiento
            query = f"""
            SELECT routine_definition
            FROM `{proyecto}.{dataset}.INFORMATION_SCHEMA.ROUTINES`
            WHERE routine_type = 'PROCEDURE'
            AND routine_name = '{nombre_procedimiento}'
            """

            resultados = client.query(query).result()
            filas = list(resultados)

            if not filas:
                if dataset.startswith("DS_RDV"):
                    return {
                        "procedimiento": None,
                        "mensaje": f"No se encontró el procedimiento '{nombre_procedimiento}'. Dado que el dataset '{dataset}' comienza con 'DS_RDV', probablemente se trate de una tabla de carga inicial o de streaming."
                    }
                else:
                    return {
                        "procedimiento": None,
                        "mensaje": f"No se encontró el procedimiento '{nombre_procedimiento}' en el dataset '{dataset}'."
                    }

            return {
                "procedimiento": f"{proyecto}.{dataset}.{nombre_procedimiento}",
                "codigo": filas[0].routine_definition
            }

        except Exception as e:
            return {"error": str(e)}
    # Inicializamos la memoria
    with ConnectionPool(
            # Example configuration
            conninfo=DB_URI,
            max_size=20,
            kwargs=connection_kwargs,
    ) as pool:
        checkpointer = PostgresSaver(pool)
        checkpointer = PostgresCheckpointer.from_config(
            db_url=DB_URI,
            create_tables_if_not_exists=True
        )

        # Inicializamos el modelo
        model = ChatOpenAI(model="gpt-4.1-2025-04-14")

        # Agrupamos las herramientas
        tolkit = [obtener_esquema_tabla, sugerir_select_bigquery, listar_tablas, info_tabla, listar_rutinas, particionamiento_tabla, obtener_procedimiento_asociado]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 """
                Eres un asistente experto en ingeniería y gobierno de datos. Estás conectado a herramientas que permiten responder preguntas sobre metadata de tablas en BigQuery: como sus columnas, tipos de datos, políticas de clasificación (policy tags), tamaños, fechas de modificación, rutinas (stored procedures), y más.

                Tu rol es ayudar al usuario a explorar y entender los datasets, respondiendo de forma técnica pero clara.

                Usa únicamente las herramientas disponibles (como búsqueda semántica o consulta directa) para responder. Si no puedes resolver algo con las herramientas, dilo explícitamente y sugiere cómo podría obtenerse la información.

                Sé preciso, breve y profesional.
                 """),
                ("human", "{messages}"),
            ]
        )
        # inicializamos el agente
        agent_executor = create_react_agent(model, tolkit, checkpointer=checkpointer, prompt=prompt)
        # ejecutamos el agente
        config = {"configurable": {"thread_id": id_agente}}
        response = agent_executor.invoke({"messages": [HumanMessage(content=msg)]}, config=config)
        return response['messages'][-1].content


if __name__ == '__main__':
    # La aplicación escucha en el puerto 8080, requerido por Cloud Run
    app.run(host='0.0.0.0', port=8080)