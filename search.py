def concordance(document):
    """Generate a concordance dictionary from the input document string"""
    if type(document) != str:
        raise ValueError("This function accepts only string inputs!")
    con = {}
    for word in document.split():
        if con.has_key(word):
            con[word] += 1
        else:
            con[word] = 1
    return con