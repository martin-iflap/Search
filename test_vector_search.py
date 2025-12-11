from vector_search import VectorCompare, remove_diacritics
import tempfile
import pytest
import math
import os


@pytest.fixture
def default_config() -> dict:
    """Return the default config dict for testing"""
    return {
        "stopwords": {"test_words": ["this", "is", "a"]}, # make test words a set??
        "vector_search": {
            "boost_factor": 2.0,
            "use_lemmatization": False,
            "use_file_lemmatization": False,
            "max_search_results": 5
        }
    }

@pytest.fixture
def vc(default_config: dict) -> VectorCompare:
    """Return a VectorCompare object with default config"""
    return VectorCompare(config=default_config)

def test_remove_diacritics() -> None:
    """Test the remove_diacritics function from vector_search"""
    test_text1: str = "Ráno rád programuješ ale to aj večer."
    expected_text2: str = "Rano rad programujes ale to aj vecer."
    assert remove_diacritics(test_text1) == expected_text2

    test_text2: str = "Café naïve façade coöperate jalapeño"
    expected_text2: str = "Cafe naive facade cooperate jalapeno"
    assert remove_diacritics(test_text2) == expected_text2

    test_text3: str = "No diacritics here!"
    expected_text3: str = "No diacritics here!"
    assert remove_diacritics(test_text3) == expected_text3

def test_nlp_property(vc: VectorCompare, default_config: dict) -> None:
    """Test the nlp property of VectorCompare"""
    vc_no_spacy = VectorCompare(config=default_config)
    assert vc_no_spacy._nlp is None
    assert vc_no_spacy._spacy is None
    nlp_result = vc_no_spacy.nlp
    assert nlp_result is None
    assert vc_no_spacy._nlp is False

    # Test with spaCy (lemmatization enabled)
    config_with_spacy = {
        "stopwords": {"files": []},
        "vector_search": {
            "boost_factor": 2.0,
            "use_lemmatization": True,
            "use_file_lemmatization": False,
            "max_search_results": 5
        }
    }
    vc_with_spacy = VectorCompare(config=config_with_spacy)
    assert vc_with_spacy._nlp is None
    assert vc_with_spacy._spacy is not None
    nlp_instance = vc_with_spacy.nlp
    assert nlp_instance is not None
    assert vc_with_spacy._nlp is nlp_instance
    assert vc_with_spacy.nlp is nlp_instance

def test_relation(vc: VectorCompare) -> None:
    """Test the relation (cosine similarity) calculation of VectorCompare"""
    vector1: dict[str, int] = {'word1': 1.0, 'word2': 2.0, 'word3': 3.0}
    vector2: dict[str, int] = {'word1': 4.0, 'word2': 5.0, 'word3': 6.0}
    expected_relation = (1*4 + 2*5 + 3*6) / (vc.magnitude(vector1) * vc.magnitude(vector2))
    assert vc.relation(vector1, vector2) == pytest.approx(expected_relation)

    vector3 = {'word4': 0.0, 'word5': 2.0}
    vector4 = {'word6': 3.0, 'word7': 0.0}
    assert vc.relation(vector3, vector4) == 0.0

def test_magnitude(vc: VectorCompare) -> None:
    """Test the magnitude calculation of VectorCompare"""
    test_vector: dict[str, int] = {'word1': 3.0, 'word2': 4.0}
    assert vc.magnitude(test_vector) == 5.0

def test_concordance(vc: VectorCompare) -> None:
    """Test the concordance generation of VectorCompare"""

    test_document: str = "This is a test. This test is only a test."
    expected_concordance: dict[str, int] = {'test': 3, 'only': 1}
    assert vc.concordance(test_document) == expected_concordance

def test_compute_idf(vc: VectorCompare) -> None:
    """Test the IDF computation of VectorCompare"""
    index: dict[str, int] = {
        0: {"concordance": {'word1': 2, 'word2': 1}},
        1: {"concordance": {'word2': 3, 'word3': 1}},
        2: {"concordance": {'word1': 1, 'word3': 2, 'word2': 1}},
    }
    vc.compute_idf(index)
    assert 'word1' in vc.idf and vc.idf['word1'] == pytest.approx(math.log((3 + 1) / (2 + 1)))
    assert 'word2' in vc.idf and vc.idf['word2'] == pytest.approx(math.log((3 + 1) / (3 + 1)))
    assert 'word3' in vc.idf and vc.idf['word3'] == pytest.approx(math.log((3 + 1) / (2 + 1)))

def test_tf_idf_vector(vc: VectorCompare) -> None:
    """Test the TF-IDF vector computation of VectorCompare"""
    test_query: set[str] = {'word1', 'word3'}
    vc.idf = {'word1': 1.0, 'word2': 0.5, 'word3': 2.0}
    concordance: dict[str, int] = {'word1': 2, 'word2': 1, 'word3': 1}
    tf_idf_vector = vc.tf_idf_vector(concordance, query_w=test_query, boost=5.0)
    total_words = sum(concordance.values())
    expected_word1 = (2 / total_words) * vc.idf['word1'] * 5.0
    expected_word2 = (1 / total_words) * vc.idf['word2']
    expected_word3 = (1 / total_words) * vc.idf['word3'] * 5.0
    assert tf_idf_vector['word1'] == pytest.approx(expected_word1)
    assert tf_idf_vector['word2'] == pytest.approx(expected_word2)
    assert tf_idf_vector['word3'] == pytest.approx(expected_word3)

def test_search_file(vc: VectorCompare) -> None:
    """Test the search_file method of VectorCompare"""
    vc = VectorCompare()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                     delete=False, encoding='utf-8') as tmp:
        tmp.write("Python is great. I love programming in Python. "
                  "Machine learning is a fascinating field. Python has many libraries for ML.")
        tmp_path = tmp.name

    try:
        # Basic search
        results = vc.search_file(tmp_path, {"python", "machine", "learning"}, max_results=3)
        assert len(results) <= 3
        assert results[0] == "Machine learning is a fascinating field"

        # No matches
        results = vc.search_file(tmp_path, {"nonexistentword"}, max_results=3)
        assert len(results) == 0

        # Empty search terms
        results = vc.search_file(tmp_path, set())
        assert results == []
    finally:
        os.unlink(tmp_path)