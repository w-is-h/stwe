from nltk.corpus import stopwords

class DocumentTimePair(object):
    def __init__(self, path):
        self.path = path
        self.preproc = Preprocessing()

    def __iter__(self):
        for line in open(self.path):
            parts = line.split("\t")
            try:
                yield ( int(parts[0]), self.preproc.pp_doc(parts[1]) )
            except:
                continue


class Preprocessing(object):
    def __init__(self, doc_min_len=2):
        self.stopwords = stopwords.words('english')
        self.doc_min_len = doc_min_len
    
    def pp_doc(self, doc):
        return [x for x in doc.split() if x not in self.stopwords and len(x) > self.doc_min_len]

        
