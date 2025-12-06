from docx import Document
import unicodedata
import logging
import math
import os


DIR_PATH = "C:\\Users\\User\\Documents\\Martin"
SUPPORTED_FILE_TYPES = {".txt", ".docx"}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

STOP_WORDS = { # exclude these from the concordance but keep an eye on them for future improvements
    # English
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "about", "as", "into", "through", "during", "before", "after",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
    # Slovak
    "a", "aj", "ale", "aby", "ak", "ako", "alebo", "by", "bol", "bola", "bolo",
    "boli", "byť", "do", "je", "jeho", "jej", "ich", "im", "ja", "ku", "k", "ma",
    "mi", "mna", "mne", "na", "nad", "ním", "niečo", "nič", "o", "od", "on", "ona",
    "ono", "oni", "po", "pod", "pre", "pri", "s", "sa", "sem", "si", "sme", "som",
    "so", "som", "sú", "ta", "tak", "tam", "ten", "tí", "to", "tú", "tu", "ty",
    "v", "vo", "z", "za", "že"
}

class VectorCompare:
    def __init__(self):
        """Initialize the VectorCompare with an empty IDF dictionary"""
        self.idf = {}

    def compute_idf(self, index: dict[int, dict]) -> None:
        """Compute the inverse document frequency (IDF) for each word in the index and store it in self.idf
         - index: a dictionary where keys are document indexes and values are dictionaries
         - IDF = log(Total number of documents / Number of documents containing the word)
        """
        doc_num = len(index)
        words_in_doc_count = {}

        for i_value in index.values():
            unique_words = set(i_value["concordance"].keys())
            for word in unique_words:
                words_in_doc_count[word] = words_in_doc_count.get(word, 0) + 1

        for word, count in words_in_doc_count.items():
            self.idf[word] = math.log((doc_num + 1) / (count + 1))

    def tf_idf_vector(self, s_concordance: dict[str, int], query_w: set = None, boost: float = 2.0) -> dict[str, float]:
        """Compute the TF-IDF vector for a given concordance dictionary
         - TF = (number the word occurs in the concordance) / (length of the concordance)
         - IDF is retrieved from self.idf
         - if query_w is provided and the word is in query_w, multiply TF-IDF by boost to increase weight
         - return a dic representing the TF-IDF vector (TF-IDF = TF * IDF)
        """
        total_words = sum(s_concordance.values())
        tf_idf_vector = {}

        for word, count in s_concordance.items():
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
        if type(vector1) != dict or type(vector2) != dict:
            raise ValueError("This function accepts only dictionary inputs!")
        relevance = 0
        top_value = 0
        for word in vector1.keys():
            if vector2.get(word):
                top_value += vector1[word] * vector2[word]
        if self.magnitude(vector1) * self.magnitude(vector2) != 0:
            relevance = top_value / (self.magnitude(vector1) * self.magnitude(vector2))
        return relevance

    def concordance(self, document: str) -> dict[str, int]:
        """Generate a concordance dictionary from the input document string
         - return a dictionary with words as keys and their count as values
         - exclude stop words defined in STOP_WORDS
        """
        if type(document) != str:
            raise ValueError("This function accepts only string inputs!")
        con = {}
        for word in document.split():
            word = word.strip(".,!?;:\"'()[]{}<>").lower()
            word = self.remove_diacritics(word)
            if word in STOP_WORDS or word == "":
                continue
            else:
                if con.get(word):
                    con[word] += 1
                else:
                    con[word] = 1
        return con

    def remove_diacritics(self, text: str) -> str:
        """Remove diacritics from the input text string
         - use this to be able to search without diacritics
        """
        normalized = unicodedata.normalize('NFD', text)
        return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')

def load_files(folder, v: VectorCompare) -> dict[int, dict]:
    """Load files from the specified folder and return a concordance index
     - return a dictionary with indexes as keys and dicts as values where:
     -    1. concordance is a dict of word counts from the concordance method
     -    2. filepath is the path to the file
    """
    index = {}
    try:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)

            if os.path.isdir(filepath): # Handle subdirectories by recursion and merging results
                inside = load_files(filepath, v)
                for value in inside.values():
                    index[len(index)] = value
                continue

            if not filename.endswith(tuple(SUPPORTED_FILE_TYPES)):
                continue

            content = ""

            if filename.endswith(".txt"):
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()
            elif filename.endswith(".docx"):
                doc = Document(filepath)
                content = "\n".join([par.text for par in doc.paragraphs])

            index[len(index)] = {
                "concordance" : v.concordance(content),
                "filepath" : filepath
            }
        # logging.info("Loaded %d files from %s successfully", len(index), folder)
        return index
    except Exception as e:
        logging.error("Error loading files: %s", e)
        return {}


def main() -> None:
    """Main function to execute the search"""
    v = VectorCompare()
    index = load_files(DIR_PATH, v)
    v.compute_idf(index)

    while True:
        search_term = input("Enter search term: ")
        search_concordance = v.concordance(search_term)
        search_vector = v.tf_idf_vector(search_concordance)
        query_words = set(search_concordance.keys())
        matches = []

        for i in range(len(index)):
            file_vector = v.tf_idf_vector(index[i]["concordance"], query_w=query_words, boost=2.5)
            score = v.relation(search_vector, file_vector)
            if score > 0.005:
                matches.append((score, index[i]["filepath"]))
                # logging.info("Score for file n %d: %.4f", i+1, score)

        matches.sort(reverse=True)
        for score, filepath in matches:
            print(f"Score: {score:.4f} - File: {filepath}")
        cont = input("Type exit to exit. Press any key to continue... ")
        if cont.lower() == "exit":
            break

if __name__ == "__main__":
    main()