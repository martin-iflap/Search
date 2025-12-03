from docx import Document
import logging
import os


DIR_PATH = "C:\\Users\\User\\Documents\\Martin"
SUPPORTED_FILE_TYPES = {".txt", ".docx"}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

class VectorCompare:
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
        """
        if type(document) != str:
            raise ValueError("This function accepts only string inputs!")
        con = {}
        for word in document.split():
            if con.get(word):
                con[word] += 1
            else:
                con[word] = 1
        return con

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
                index.update(inside)
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
                "concordance" : v.concordance(content.lower()),
                "filepath" : filepath
            }
        logging.info("Loaded %d files from %s successfully", len(index), folder)
        return index
    except Exception as e:
        logging.error("Error loading files: %s", e)
        return {}

def main():
    v = VectorCompare()
    index = load_files(DIR_PATH, v)

    search_term = input("Enter search term: ").lower()
    matches = []

    for i in range(len(index)):
        score = v.relation(v.concordance(search_term), index[i]["concordance"])
        logging.info("Score for file n %d: %.4f", i+1, score)
        if score != 0:
            matches.append((score, index[i]["filepath"]))

    matches.sort(reverse=True)
    for score, filepath in matches:
        print(f"Score: {score:.4f} - File: {filepath}")

def test():
    """Random test function"""
    for filename in os.listdir("C:\\Users\\User\\Documents\\Martin"):
        print(filename)

if __name__ == "__main__":
    main()