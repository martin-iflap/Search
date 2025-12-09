from config_loader import get_all_stopwords, get_vector_search_config, load_config
import unicodedata
import logging
import spacy
import math


class VectorCompare:
    def __init__(self, config: dict = None) -> None:
        """Initialize the VectorCompare with an empty IDF dictionary"""
        if config is None:
            config = load_config()

        self._stop_words = get_all_stopwords(config)
        search_config = get_vector_search_config(config)
        self._boost_factor = search_config["boost_factor"]
        self._use_lemmatization = search_config["use_lemmatization"]

        self.idf = {}
        self._nlp = None

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
        total = 0
        for count in vector.values():
            total += count ** 2
        return total ** 0.5

    def relation(self, vector1: dict, vector2: dict) -> float:
        """Compute the cosine similarity between two vectors represented as dictionaries"""
        if not isinstance(vector1, dict) or not isinstance(vector2, dict):
            raise ValueError("This function accepts only dictionary inputs!")
        relevance = 0
        top_value = 0
        for word in vector1.keys():
            if vector2.get(word):
                top_value += vector1[word] * vector2[word]
        if self.magnitude(vector1) * self.magnitude(vector2) != 0:
            relevance = top_value / (self.magnitude(vector1) * self.magnitude(vector2))
        return relevance

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
            word = self.remove_diacritics(word)
            if word in self._stop_words or word == "":
                continue
            con[word] = con.get(word, 0) + 1
        return con

    def remove_diacritics(self, text: str) -> str:
        """Remove diacritics from the input text string
         - use this to be able to search without diacritics
        """
        if type(text) != str:
            raise ValueError("This function accepts only string inputs!")
        normalized = unicodedata.normalize('NFD', text)
        return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')