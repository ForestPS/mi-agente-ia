import streamlit as st
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages # Mágico: ayuda a acumular mensajes
from langgraph.checkpoint.memory import MemorySaver # La memoria RAM
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. CONFIGURACIÓN E INICIALIZACIÓN
# ==========================================
load_dotenv()
st.set_page_config(page_title="Agente con Memoria", page_icon="🧠", layout="wide")
st.title("🧠 Agente Inteligente con Memoria")

traductor_de_texto = OpenAIEmbeddings(model="text-embedding-3-small")
base_de_datos = Chroma(persist_directory="./mi_base_de_datos", embedding_function=traductor_de_texto)
cerebro_ia = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

# ==========================================
# 2. EL CEREBRO DEL AGENTE (LANGGRAPH)
# ==========================================

class EstadoAgente(TypedDict):
    # Annotated + add_messages hace que los mensajes nuevos se peguen a los viejos
    messages: Annotated[list[BaseMessage], add_messages]
    libro_filtro: str 
    contexto_pdf: str
    tipo_de_pregunta: str

def nodo_clasificar_pregunta(estado: EstadoAgente):
    # Tomamos el último mensaje enviado por el usuario
    ultima_pregunta = estado["messages"][-1].content
    
    instruccion = (
        "Eres un clasificador. Si el usuario saluda o hace charla casual, responde: SALUDO. "
        "Si pregunta por información de libros o datos, responde: PDF."
    )
    mensajes_clasificacion = [SystemMessage(content=instruccion), HumanMessage(content=ultima_pregunta)]
    respuesta = cerebro_ia.invoke(mensajes_clasificacion)
    clasificacion = respuesta.content.strip().upper()
    if "SALUDO" not in clasificacion: clasificacion = "PDF"
    return {"tipo_de_pregunta": clasificacion}

def nodo_hablar_normal(estado: EstadoAgente):
    # Le pasamos TODO el historial de mensajes para que tenga memoria de la charla
    instruccion = SystemMessage(content="Eres un asistente amigable. Responde de forma natural y breve.")
    historial = [instruccion] + estado["messages"]
    respuesta = cerebro_ia.invoke(historial)
    return {"messages": [respuesta]} # Solo devolvemos la nueva respuesta

def nodo_investigar_pdf(estado: EstadoAgente):
    ultima_pregunta = estado["messages"][-1].content
    filtro = estado.get("libro_filtro", "Todos los libros")
    
    # Buscamos en Chroma
    if filtro == "Todos los libros":
        fragmentos = base_de_datos.similarity_search(ultima_pregunta, k=3)
    else:
        fragmentos = base_de_datos.similarity_search(ultima_pregunta, k=3, filter={"source": filtro})
        
    textos_armados = []
    for f in fragmentos:
        nombre_libro = f.metadata.get('source', 'Libro desconocido')
        textos_armados.append(f"[Libro: {nombre_libro}]\n{f.page_content}")
    contexto = "\n\n".join(textos_armados)
    
    instruccion = SystemMessage(content=(
        f"Responde basándote ÚNICAMENTE en este texto:\n\n{contexto}\n\n"
        "Si el usuario pregunta 'de dónde' o 'qué libro', usa las etiquetas [Libro: ...]."
    ))
    
    # IMPORTANTE: Le pasamos el historial + la instrucción de búsqueda
    historial_con_contexto = [instruccion] + estado["messages"]
    respuesta = cerebro_ia.invoke(historial_con_contexto)
    
    return {"contexto_pdf": contexto, "messages": [respuesta]}

def decidir_camino(estado: EstadoAgente):
    if estado.get("tipo_de_pregunta") == "SALUDO": return "charlador"
    else: return "investigador"

# --- CONSTRUCCIÓN DEL GRAFO CON MEMORIA ---
flujo = StateGraph(EstadoAgente)
flujo.add_node("clasificador", nodo_clasificar_pregunta)
flujo.add_node("charlador", nodo_hablar_normal)
flujo.add_node("investigador", nodo_investigar_pdf)

flujo.add_edge(START, "clasificador")
flujo.add_conditional_edges("clasificador", decidir_camino)
flujo.add_edge("charlador", END)
flujo.add_edge("investigador", END)

# 1. Creamos el objeto de memoria
memoria_interna = MemorySaver()

# 2. Compilamos el agente pasándole la memoria
agente = flujo.compile(checkpointer=memoria_interna)

# ==========================================
# 3. INTERFAZ VISUAL (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("📥 Biblioteca")
    archivo_subido = st.file_uploader("Subir PDF", type="pdf")
    if archivo_subido and st.button("Aprender Libro"):
        with open("temp.pdf", "wb") as f: f.write(archivo_subido.getbuffer())
        with st.spinner("Leyendo..."):
            lector = PyPDFLoader("temp.pdf")
            paginas = lector.load()
            
            # IA descubre el título
            texto_portada = "".join([p.page_content for p in paginas[:2]])
            res_titulo = cerebro_ia.invoke([SystemMessage(content="Dime el título real del libro. Solo el título."), HumanMessage(content=texto_portada)])
            nombre_real = res_titulo.content.strip()
            
            cortador = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            fragmentos = cortador.split_documents(paginas)
            for f in fragmentos: f.metadata["source"] = nombre_real
            base_de_datos.add_documents(fragmentos)
        st.success(f"Aprendido: {nombre_real}")
        st.rerun()

    st.divider()
    datos_bd = base_de_datos.get()
    libros_guardados = sorted(list(set([m['source'] for m in datos_bd['metadatas'] if m and 'source' in m])))
    libro_seleccionado = st.selectbox("Buscar en:", ["Todos los libros"] + libros_guardados)

    if st.button("✨ Nueva conversación"):
        # Al limpiar mensajes de Streamlit, el hilo seguirá en memoria pero vacío visualmente
        st.session_state.mensajes_visuales = []
        st.rerun()
    # ==========================================
    # ZONA DE PELIGRO (GESTIÓN DE LIBROS)
    # ==========================================
    st.divider()
    st.header("🗑️ Zona de peligro")

    # 1. Borrar un libro específico
    if len(libros_guardados) > 0:
        st.write("¿Quieres eliminar un libro?")
        libro_a_borrar = st.selectbox(
            "Selecciona el libro a eliminar:", 
            libros_guardados, 
            key="selector_borrado"
        )
        
        if st.button("❌ Eliminar libro seleccionado"):
            with st.spinner(f"Borrando {libro_a_borrar}..."):
                resultado_busqueda = base_de_datos.get(where={"source": libro_a_borrar})
                ids_a_borrar = resultado_busqueda['ids']
                
                if len(ids_a_borrar) > 0:
                    base_de_datos.delete(ids=ids_a_borrar)
                    st.success(f"Libro '{libro_a_borrar}' eliminado correctamente.")
                    st.session_state.mensajes_visuales = []
                    st.rerun()
                else:
                    st.error("No se encontraron fragmentos para borrar.")
    else:
        st.info("No hay libros para borrar.")

    # 2. Borrado total
    st.write("---")
    if st.button("🚨 Borrar toda la biblioteca"):
        base_de_datos.delete_collection()
        st.session_state.mensajes_visuales = []
        st.warning("Biblioteca destruida. Reinicia la app para crear una nueva.")
        st.rerun()

# ==========================================
# 4. CHAT PRINCIPAL
# ==========================================
if "mensajes_visuales" not in st.session_state:
    st.session_state.mensajes_visuales = []

for msj in st.session_state.mensajes_visuales:
    with st.chat_message(msj["rol"]): st.write(msj["contenido"])

pregunta_usuario = st.chat_input("Escribe aquí...")

if pregunta_usuario:
    st.session_state.mensajes_visuales.append({"rol": "user", "contenido": pregunta_usuario})
    with st.chat_message("user"): st.write(pregunta_usuario)
    
    with st.chat_message("assistant"):
        configuracion = {"configurable": {"thread_id": "usuario_unico_123"}}
        estado_input = {
            "messages": [HumanMessage(content=pregunta_usuario)],
            "libro_filtro": libro_seleccionado
        }
        
        # 1. Creamos una caja de texto "vacía" en la pantalla
        caja_texto = st.empty()
        respuesta_acumulada = ""
        
        # 2. Reemplazamos .invoke() por .stream() en modo "mensajes"
        # Esto hace que LangGraph nos lance las letras en vivo mientras la IA las genera
        for fragmento, metadatos in agente.stream(estado_input, config=configuracion, stream_mode="messages"):
            
            # Solo atrapamos los pedacitos de texto que vienen de nuestros expertos
            if metadatos.get("langgraph_node") in ["charlador", "investigador"]:
                
                # Si el fragmento contiene letras, las sumamos a nuestra respuesta
                if fragmento.content:
                    respuesta_acumulada += fragmento.content
                    
                    # Dibujamos el texto en vivo añadiendo un bloque negro (cursor) al final
                    caja_texto.markdown(respuesta_acumulada + " ▌")
        
        # 3. Al terminar, quitamos el cursor y dejamos el texto final limpio
        caja_texto.markdown(respuesta_acumulada)

    # 4. Guardamos la respuesta completa en la memoria visual de Streamlit
    st.session_state.mensajes_visuales.append({"rol": "assistant", "contenido": respuesta_acumulada})