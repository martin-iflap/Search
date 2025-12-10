from config_loader import get_all_stopwords, get_vector_search_config, load_config
from functools import lru_cache
from pypdf import PdfReader
from docx import Document
import unicodedata
import logging
import spacy
import math
import re

SENTENCE_SPLIT_PATTERN = re.compile(r'[.!?]+')

@lru_cache(maxsize=1000)
def remove_diacritics(text: str) -> str:
    """Remove diacritics from the input text string
     - use this to be able to search without diacritics
    """
    if type(text) != str:
        raise ValueError("This function accepts only string inputs!")
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')

class VectorCompare:
    def __init__(self, config: dict = None) -> None:
        """Initialize the VectorCompare with an empty IDF dictionary"""
        if config is None:
            config = load_config()

        self._stop_words = get_all_stopwords(config)
        search_config = get_vector_search_config(config)
        self._boost_factor = search_config["boost_factor"]
        self._use_lemmatization = search_config["use_lemmatization"]
        self._use_file_lemmatization = search_config["use_file_lemmatization"]
        self._max_search_results = search_config["max_search_results"]

        self.idf = {}
        self._nlp = None
        self.file_vector_cache: dict[int, dict[str, float]] = {}

    @property
    def nlp(self):
        """Lazy load the spaCy NLP model when first needed"""
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logging.error("Error loading spaCy model: %s", e)
                self._nlp = False
        return self._nlp if self._nlp is not False else None

    def compute_idf(self, index: dict[int, dict]) -> None:
        """Compute the inverse document frequency (IDF) for each word in the index and store it in self.idf
         - index: a dictionary where keys are document indexes and values are dictionaries
         - IDF = log(Total number of documents / Number of documents containing the word)
        """
        if not isinstance(index, dict):
            raise ValueError("This function accepts only dictionary inputs!")

        doc_num = len(index)
        words_in_doc_count = {}

        for i_value in index.values():
            unique_words = set(i_value["concordance"].keys())
            for word in unique_words:
                words_in_doc_count[word] = words_in_doc_count.get(word, 0) + 1

        for word, count in words_in_doc_count.items():
            self.idf[word] = math.log((doc_num + 1) / (count + 1))

    def tf_idf_vector(self, concordance: dict[str, int], query_w: set = None, boost: float = None) -> dict[str, float]:
        """Compute the TF-IDF vector for a given concordance dictionary
         - TF = (number the word occurs in the concordance) / (length of the concordance)
         - IDF is retrieved from self.idf
         - if query_w is provided and the word is in query_w, multiply TF-IDF by boost to increase weight
         - return a dic representing the TF-IDF vector (TF-IDF = TF * IDF)
        """
        if not isinstance(concordance, dict):
            raise ValueError("This function accepts only dictionary inputs!")

        if boost is None:
            boost = self._boost_factor

        total_words = sum(concordance.values())
        tf_idf_vector = {}

        for word, count in concordance.items():
            tf = count / total_words
            idf = self.idf.get(word, 0)
            tf_idf = tf * idf
            if query_w and word in query_w:
                tf_idf *= boost
                # logging.info("Boosted TF-IDF for word '%s': %.4f", word, tf_idf)
            tf_idf_vector[word] = tf_idf
        return tf_idf_vector

    def magnitude(self, vector: dict) -> float:
        """Compute the magnitude of a vector represented as a dictionary"""
        if type(vector) != dict:
            raise ValueError("This function accepts only dictionary inputs!")
        return math.sqrt(sum(val ** 2 for val in vector.values()))

    def relation(self, vector1: dict, vector2: dict) -> float:
        """Compute the cosine similarity between two vectors represented as dictionaries"""
        if not isinstance(vector1, dict) or not isinstance(vector2, dict):
            raise ValueError("This function accepts only dictionary inputs!")

        common_words = set(vector1.keys()) & set(vector2.keys())
        if not common_words:
            return 0.0

        top_value = sum(vector1[word] * vector2[word] for word in common_words)
        magnitude_v1 = self.magnitude(vector1)
        magnitude_v2 = self.magnitude(vector2)

        return top_value / (magnitude_v1 * magnitude_v2) if magnitude_v1 * magnitude_v2 != 0 else 0.0

    def concordance(self, document: str, use_lemmatization: bool = None) -> dict[str, int]:
        """Generate a concordance dictionary from the input document string
         - return a dictionary with words as keys and their count as values
         - exclude stop words defined in STOP_WORDS
        """
        if not isinstance(document, str):
            raise ValueError("This function accepts only string inputs!")

        if use_lemmatization is None:
            use_lemmatization = self._use_lemmatization
        con = {}

        is_non_eng: bool = any(char in document.lower() for char in set('ľščťžýáíéôúäöüß'))
        if use_lemmatization and self.nlp and not is_non_eng:
                doc = self.nlp(document)
                words = [token.lemma_ for token in doc]
        else:
            words = document.split()

        for word in words:
            word = word.strip(".,!?;:\"'()[]{}<>").lower()
            word = remove_diacritics(word)
            if word and word not in self._stop_words:
                con[word] = con.get(word, 0) + 1
        return con

    def search_file(self, filepath: str, query_words: str,
                    max_results: int = None, use_file_lemmatization: bool = None) -> list[str]:
        """Search for the query_words in the file specified by filepath
         - return a list of the top max_results sentences containing the query_words, ranked by relevance
         - support .txt, .docx, and .pdf files
         - return an empty list if an error occurs
         - relevance is determined by the number of overlapping words between the query_words and each sentence
        """
        if max_results is None:
            max_results = self._max_search_results
        if use_file_lemmatization is None:
            use_file_lemmatization = self._use_file_lemmatization

        try:
            content: str = ""
            if filepath.endswith(".txt"):
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()
            elif filepath.endswith(".docx"):
                doc = Document(filepath)
                content = "\n".join([par.text for par in doc.paragraphs])
            elif filepath.endswith(".pdf"):
                reader = PdfReader(filepath)
                content = "".join(page.extract_text() or "" for page in reader.pages)

            sentences: list[str] = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(content)
                                    if s.strip()]
            if not query_words:
                logging.warning("No valid search words found in search term: %s", search_term)
                return []
            scored = []

            for sentence in sentences:
                sentence_lower = sentence.lower()
                if use_file_lemmatization and self.nlp:
                    doc = self.nlp(sentence_lower)
                    sentence_words = set(
                                remove_diacritics(token.lemma_)
                                for token in doc
                                if token.lemma_ not in self._stop_words
                    )
                else:
                    sentence_words = set(
                                remove_diacritics(w)
                                for w in sentence_lower.split()
                                if w not in self._stop_words
                    )
                overlap = len(query_words & sentence_words)
                if overlap > 0:
                    scored.append((overlap / len(query_words), sentence))

            scored.sort(reverse=True)
            return [sentence for _, sentence in scored[:max_results]]

        except Exception as e:
            logging.error("Error searching file %s: %s", filepath, e)
            return []