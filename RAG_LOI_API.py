import os
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
import streamlit as st
import faiss
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np
import pickle
import requests
from typing import List, Dict, Optional, Union
import re


import datetime
import requests


class RAGLoader:
    def __init__(self, 
                 docs_folder: str = "./docs", 
                 splits_folder: str = "./splits",
                 index_folder: str = "./index"):
        """
        Initialise le RAG Loader
        
        Args:
            docs_folder: Dossier contenant les documents sources
            splits_folder: Dossier où seront stockés les morceaux de texte
            index_folder: Dossier où sera stocké l'index FAISS
        """
        self.docs_folder = Path(docs_folder)
        self.splits_folder = Path(splits_folder)
        self.index_folder = Path(index_folder)
        
        # Créer les dossiers s'ils n'existent pas
        self.splits_folder.mkdir(parents=True, exist_ok=True)
        self.index_folder.mkdir(parents=True, exist_ok=True)
        
        # Chemins des fichiers
        self.splits_path = self.splits_folder / "splits.json"
        self.index_path = self.index_folder / "faiss.index"
        self.documents_path = self.index_folder / "documents.pkl"
        
        # Initialiser le modèle
        # self.model = None
        self.index = None
        self.indexed_documents = None

    def encode(self,payload):
        API_URL = "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-large"
        headers = {"Authorization": "Bearer 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"} 
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
        
    def load_and_split_texts(self) -> List[Document]:
        """
        Charge les textes du dossier docs, les découpe en morceaux et les sauvegarde
        dans un fichier JSON unique.
        
        Returns:
            Liste de Documents contenant les morceaux de texte et leurs métadonnées
        """
        documents = []
        
        # Vérifier d'abord si les splits existent déjà
        if self._splits_exist():
            print("Chargement des splits existants...")
            return self._load_existing_splits()
        
        print("Création de nouveaux splits...")
        # Parcourir tous les fichiers du dossier docs
        for file_path in self.docs_folder.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
                # Découper le texte en phrases
                chunks = chunks = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                
                # Créer un Document pour chaque morceau
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': file_path.name,
                            'chunk_id': i,
                            'total_chunks': len(chunks)
                        }
                    )
                    documents.append(doc)
        
        # Sauvegarder tous les splits dans un seul fichier JSON
        self._save_splits(documents)
        
        print(f"Nombre total de morceaux créés: {len(documents)}")
        return documents
    
    def _splits_exist(self) -> bool:
        """Vérifie si le fichier de splits existe"""
        return self.splits_path.exists()
    
    def _save_splits(self, documents: List[Document]):
        """Sauvegarde tous les documents découpés dans un seul fichier JSON"""
        splits_data = {
            'splits': [
                {
                    'text': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in documents
            ]
        }
        
        with open(self.splits_path, 'w', encoding='utf-8') as f:
            json.dump(splits_data, f, ensure_ascii=False, indent=2)
    
    def _load_existing_splits(self) -> List[Document]:
        """Charge les splits depuis le fichier JSON unique"""
        with open(self.splits_path, 'r', encoding='utf-8') as f:
            splits_data = json.load(f)
            
        documents = [
            Document(
                page_content=split['text'],
                metadata=split['metadata']
            )
            for split in splits_data['splits']
        ]
        
        print(f"Nombre de splits chargés: {len(documents)}")
        return documents

    def load_index(self) -> bool:
        """
        Charge l'index FAISS et les documents associés s'ils existent
        
        Returns:
            bool: True si l'index a été chargé, False sinon
        """
        if not self._index_exists():
            print("Aucun index trouvé.")
            return False
            
        print("Chargement de l'index existant...")
        try:
            # Charger l'index FAISS
            self.index = faiss.read_index(str(self.index_path))
            
            # Charger les documents associés
            with open(self.documents_path, 'rb') as f:
                self.indexed_documents = pickle.load(f)
                
            print(f"Index chargé avec {self.index.ntotal} vecteurs")
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement de l'index: {e}")
            return False

           
    def create_index(self, documents: Optional[List[Document]] = None) -> bool:
        """
        Crée un nouvel index FAISS à partir des documents.
        Si aucun document n'est fourni, charge les documents depuis le fichier JSON.
        
        Args:
            documents: Liste optionnelle de Documents à indexer
            
        Returns:
            bool: True si l'index a été créé avec succès, False sinon
        """
        try:           
            # Charger les documents si non fournis
            if documents is None:
                documents = self.load_and_split_texts()
            
            if not documents:
                print("Aucun document à indexer.")
                return False
            
            print("Création des embeddings...")
            texts = [doc.page_content for doc in documents]
            embeddings = self.encode(texts)
            
            # Initialiser l'index FAISS
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            
            # Ajouter les vecteurs à l'index
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Sauvegarder l'index
            print("Sauvegarde de l'index...")
            faiss.write_index(self.index, str(self.index_path))
            
            # Sauvegarder les documents associés
            self.indexed_documents = documents
            with open(self.documents_path, 'wb') as f:
                pickle.dump(documents, f)
                
            print(f"Index créé avec succès : {self.index.ntotal} vecteurs")
            return True
            
        except Exception as e:
            print(f"Erreur lors de la création de l'index: {e}")
            return False
    
    def _index_exists(self) -> bool:
        """Vérifie si l'index et les documents associés existent"""
        return self.index_path.exists() and self.documents_path.exists()
    
    def get_retriever(self, k: int = 5):
        """
        Crée un retriever pour l'utilisation avec LangChain
        
        Args:
            k: Nombre de documents similaires à retourner
            
        Returns:
            Callable: Fonction de recherche compatible avec LangChain
        """
        if self.index is None:
            if not self.load_index():
                if not self.create_index():
                    raise ValueError("Impossible de charger ou créer l'index")
                    
            
        def retriever_function(query: str) -> List[Document]:
            # Créer l'embedding de la requête
            query_embedding = self.encode([query])[0]
            
            # Rechercher les documents similaires
            distances, indices = self.index.search(
                np.array([query_embedding]).astype('float32'), 
                k
            )
            
            # Retourner les documents trouvés
            results = []
            for idx in indices[0]:
                if idx != -1:  # FAISS retourne -1 pour les résultats invalides
                    results.append(self.indexed_documents[idx])
                    
            return results
            
        return retriever_function



# Import de notre classe RAGLoader
# from rag_loader import RAGLoader  # Assurez-vous que le fichier précédent est nommé rag_loader.py


# rag_loader = RAGLoader()

# Configuration de la page Streamlit
st.set_page_config(
    page_title="هذا برنامج الاجابة عن الأسئلة المتعلقة بالقانون المغربي",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    /* Import de la police depuis Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    
    /* ou plusieurs polices */
    @import url('https://fonts.googleapis.com/css2?family=Almarai:wght@400;700&family=Cairo:wght@400;700&display=swap');

    /* Utilisation avec fallback */
    .question, .answer, .history {
        background-color: #f2ffcf;
        color: black;
        padding: 3px;
        border-radius: 10px;
        margin: 4px 0;
        direction: rtl;
        text-align: right;
        /* Police principale avec fallbacks */
        font-family: 'Tajawal', 'Almarai', 'Traditional Arabic', sans-serif;
        font-size: 17px;
    }

    .title {
        background-color: #ffeccf;
        color: black;
        padding: 5px;
        border-radius: 20px;
        margin: 10px 0;
        direction: rtl;
        text-align: center;
        /* Police principale avec fallbacks */
        font-family: 'Tajawal';
        font-size: 30px;
    }

    
    .stTextInput input {
        direction: rtl;
        text-align: right;
        font-family: 'Almarai', 'Tajawal', 'Traditional Arabic', sans-serif;
        font-size: 20px;
        padding: 10px;
        border-radius: 20px;
        border: 1px solid #fa0000;
        color: black;
        background-color: white;
    }

    /* Style pour le conteneur du formulaire */
    [data-testid="stForm"] {
        direction: rtl;

    
    .stButton button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)


class RAGChatBot:
    def __init__(self):
        # Initialisation du modèle LLM
        self.llm = ChatMistralAI(model="mistral-large-latest", mistral_api_key="QK0ZZpSxQbCEVgOLtI6FARQVmBYc6WGP")
        
        # Initialisation du RAG
        self.rag_loader = RAGLoader()
        
        # Chargement ou création de l'index si nécessaire
        if not self.rag_loader.load_index():
            self.rag_loader.create_index()
            
        # Obtention du retriever
        self.retriever = self.rag_loader.get_retriever(k=15)
        
        # Template du prompt en arabe
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """أنت مساعد مفيد يجيب على الأسئلة باللغة العربية باستخدام المعلومات المقدمة.
            استخدم المعلومات التالية للإجابة على السؤال:
            
            {context}
            
            إذا لم تكن المعلومات كافية للإجابة على السؤال بشكل كامل، قم بتوضيح ذلك.
            أجب بشكل دقيق دقيق."""),
            ("human", "{question}")
        ])

    def get_response(self, question: str) -> str:
        """Obtient une réponse à partir d'une question"""
        try:
            # Récupération des documents pertinents
            relevant_docs = self.retriever(question)
            
            # Préparation du contexte
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Création du prompt
            prompt = self.prompt_template.format_messages(
                context=context,
                question=question
            )
            
            # Obtention de la réponse
            response = self.llm(prompt)
            
            return response.content
        except Exception as e:
            return f"عذراً، حدث خطأ: {str(e)}"


def initialize_session_state():
    """Initialise les variables de session"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatBot()
        
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""

def handle_question():
    """Gère le traitement d'une nouvelle question"""
    # Récupère la question depuis session_state
    question = st.session_state.question_input
    
    if question:
        # Obtention de la réponse
        response = st.session_state.chatbot.get_response(question)
        
        # Ajout à l'historique
        st.session_state.chat_history.append({
            'time': datetime.datetime.now().strftime("%H:%M:%S"),
            'question': question,
            'answer': response
        })
        
        # Réinitialisation du champ de question
        st.session_state.question_input = ""

def main():
    # Initialisation des variables de session
    initialize_session_state()
    
    # Titre de l'application
    st.markdown('<h1 class="title">هذا برنامج الاجابة عن الأسئلة المتعلقة بالقانون المغربي</h1>', unsafe_allow_html=True)
 
    # Formulaire pour la question
    with st.form(key='question_form'):
        # Zone de saisie de la question
        st.text_input(
            label='',
            placeholder='اكتب سؤالك هنا...',
            key='question_input'
        )
        
        # Bouton d'envoi
        submit_button = st.form_submit_button("إرسال", on_click=handle_question)
    
    # Affichage de l'historique
    st.markdown('<div class="history">', unsafe_allow_html=True)
    for chat in reversed(st.session_state.chat_history):
        # Affichage de la question
        st.markdown(f"""
        <div class="question">
            <strong>السؤال ({chat['time']}):</strong><br>
            {chat['question']}
        </div>
        """, unsafe_allow_html=True)
        
        # Affichage de la réponse
        st.markdown(f"""
        <div class="answer">
            <strong>الجواب:</strong><br>
            {chat['answer']}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()