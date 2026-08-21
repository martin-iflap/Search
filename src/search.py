from .config_loader import get_search_config
from .vector_search import VectorCompare
import pymupdf
from docx import Document
import threading
import argparse
import logging
import fnmatch
import os

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

search_config = get_search_config()
DIR_PATH: str = search_config.get("dir_path", "C:\\Users\\User\\Documents")
SUPPORTED_FILE_TYPES: set[str] = set(search_config.get("supported_file_types", []))
SEARCH_THRESHOLD: float = search_config.get("search_threshold", 0.1)
EXCLUDE_PATTERNS = search_config.get("exclude_patterns", [])

def load_files(folder: str, v: VectorCompare) -> dict[int, dict]:
    """Load files from the specified folder and return a concordance index
     - return a dictionary with indexes as keys and dicts as values where:
     -    1. concordance is a dict of word counts from the concordance method
     -    2. filepath is the path to the file
    """
    index = {}

    try:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)

            if any(fnmatch.fnmatch(filepath, pattern) for pattern in EXCLUDE_PATTERNS):
                continue

            if os.path.isdir(filepath): # Handle subdirectories by recursion and merging results
                inside = load_files(filepath, v)
                for value in inside.values():
                    index[len(index)] = value
                continue

            if not filename.endswith(tuple(SUPPORTED_FILE_TYPES)):
                continue

            content = ""

            if filename.endswith(".txt"):
                try:
                    with open(filepath, encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    logging.error("Error reading TXT file %s: %s", filepath, e)
                    continue
            elif filename.endswith(".docx"):
                try:
                    doc = Document(filepath)
                    content = "\n".join([par.text for par in doc.paragraphs])
                except Exception as e:
                    logging.error("Error reading DOCX file %s: %s", filepath, e)
                    continue
            elif filename.endswith(".pdf"):
                try:
                    with pymupdf.open(filepath) as doc:
                        for page in doc:
                            text = page.get_text()
                            if text:
                                content += text + "\n"
                except Exception as e:
                    logging.error("Error reading PDF file %s: %s", filepath, e)
                    continue

            index[len(index)] = {
                "concordance" : v.concordance(content.lower()),
                "filepath" : filepath
            }
        # logging.info("Loaded %d files from %s successfully", len(index), folder)
        return index
    except PermissionError as e:
        logging.error("Error loading files: %s", e)
        return {}

def thread_load_files_index(folder: str, v: VectorCompare, index_container: dict) -> None:
    """Thread target function to load files and compute IDF while waiting for user input
     - update the result dict in main with the index and a finished flag
     - why multithreading? to avoid waits and because I felt like it
    """
    if not isinstance(index_container, dict):
        raise ValueError("This function accepts only dictionary inputs!")

    index = load_files(folder, v)
    v.compute_idf(index)
    index_container['index'] = index
    index_container["finished"] = True


def ask_and_display_search(v: VectorCompare, index: dict[int, str], search_term: str) -> bool:
    """Ask user for a search term, perform the search, and display results
     - return the post_search_options function to handle user options after displaying results to main
     - return True to continue searching, False to exit
    """
    search_concordance: dict[str, int] = v.concordance(search_term)
    search_vector: dict[str, float] = v.tf_idf_vector(search_concordance)
    query_words: set[str] = set(search_concordance.keys())
    matches: list[tuple[float, str]] = []

    for i in range(len(index)):
        if i not in v.file_vector_cache:
            v.file_vector_cache[i] = v.tf_idf_vector(index[i]["concordance"], query_w=query_words)
        file_vector = v.file_vector_cache[i]
        score = v.relation(search_vector, file_vector)
        if score > SEARCH_THRESHOLD:
            matches.append((score, index[i]["filepath"]))
            # logging.info("Score for file n %d: %.4f", i+1, score)

    matches.sort(reverse=True)

    if not matches:
        print("No matches found.")

    for num, (score, filepath) in enumerate(matches, start=1):
        print(f"{num}. Score: {score:.4f} - File: {filepath}")

    return post_search_options(v, matches, query_words)

def post_search_options(v: VectorCompare, matches: list[tuple[float, str]], query_words: set[str]) -> bool:
    """Provide options to the user after displaying search results
     - Allow user to open files, exit search loop or continue searching
     - Return True to continue searching, False to exit
    """
    while True:
        cont = input("Options: 'exit' | 'open <num>' | 'search <num>' | press enter to continue... ")
        if cont.lower() == "exit":
            return False

        if cont.strip() == "":
            return True

        if cont.lower().startswith("open "):
            try:
                file_number = int(cont.split()[1]) - 1
                if 0 <= file_number < len(matches):
                    try:
                        os.startfile(matches[file_number][1])
                    except FileNotFoundError:
                        logging.error("File not found: %s", matches[file_number][1])
                    except Exception as e:
                        logging.error("Error opening file: %s", e)
                else:
                    print("Invalid file number.")
            except (IndexError, ValueError):
                print("Usage: open <file_number>")

        elif cont.lower().startswith("search "):
            try:
                file_number = int(cont.split()[1]) - 1
                if 0 <= file_number < len(matches):
                    found = v.search_file(matches[file_number][1], query_words)
                    if found:
                        print("Top results from file:")
                        for s in found:
                            print(s)
                    else:
                        print("No valid results found in the file.")
                else:
                    print("Invalid file number.")
            except (IndexError, ValueError):
                print("Usage: search <file_number>")


def main() -> None:
    """Main function to execute the search
     - optional argparse argument --dir to specify directory path else DIR_PATH is used
     - loads files, computes IDF, and enters a loop to call ask_and_display_search
     - continue until user decides to exit (ask_and_display_search returns False)
    """
    parser = argparse.ArgumentParser(description="Search through text and docx files in a specified directory.")
    parser.add_argument("--dir", type=str, help="Directory path to search files in.")
    args = parser.parse_args()

    search_dir: str = args.dir if args.dir and os.path.isdir(args.dir) else DIR_PATH
    if args.dir and not os.path.exists(args.dir):
        logging.warning("Invalid directory path provided. Using default path: %s", DIR_PATH)


    v = VectorCompare()
    result: dict = {"index": None, "finished": False}
    index_thread = threading.Thread(target=thread_load_files_index, args=(search_dir, v, result))
    index_thread.start()

    while True:
        search_term = input("Enter search term: ").lower()
        while search_term.strip() == "":
            search_term = input("Enter non-empty search term: ")

        if not result["finished"]:
            index_thread.join()

        index = result["index"]
        if index is None or len(index) == 0:
            print("No files loaded to search.")
            logging.warning("Index is empty, cannot perform search")
            break

        if not ask_and_display_search(v, index, search_term):
            break


if __name__ == "__main__":
    main()

# optimize stop words for all languages, optimize performance
# write some tests, check error cases(logging vs raise)
# fuzzy matching??, multiprocessing for loading files,
# cache file content?, check the stupid folder structure handling