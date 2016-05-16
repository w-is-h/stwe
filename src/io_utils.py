from nltk.corpus import stopwords

class DocumentTimePair(object):
    def __init__(self, path, time_transform=-1, start_time=-1):
        self.path = path
        self.preproc = Preprocessing()
        self.time_transform = time_transform
        self.start_time = start_time

    def __iter__(self):
        for line in open(self.path):
            parts = line.split("\t")
            if self.time_transform >= 0 and self.start_time >= 0:
                parts[0] = (float(parts[0]) - self.start_time) / self.time_transform
            try:
                yield ( float(parts[0]), self.preproc.pp_doc(parts[1]) )
            except:
                continue


class Preprocessing(object):
    def __init__(self, doc_min_len=2):
        self.stopwords = stopwords.words('english')
        self.doc_min_len = doc_min_len
    
    def pp_doc(self, doc):
        return [x for x in doc.split() if x not in self.stopwords and len(x) > self.doc_min_len]

        
