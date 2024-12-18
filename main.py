import argparse

from depedit import DepEdit
from depedit.depedit import ParsedToken, Entity
from tqdm import tqdm
import os

"""
1. Extract all entities
2. Select referents with either:
    a. Pronominal reference
    b. Definite NP reference (the guy, the policy, ...)
3. Create pairs with most recent antecedent
4. Measure distance in number of words, number of clauses, and number of other entities
"""


def func2case(_f: str) -> str:
    if 'subj' in _f:
        case = "nom"
    elif 'obj' in _f:
        case = "acc"
    elif 'obl' in _f:
        case = "obl"
    elif 'poss' in _f:
        case = "gen"
    else:
        case = "other"
    return case


# Calculate competitor distance, i.e. number of distance between referent (t1) and anaphora (t2)
def get_competitor_dist(t1: Entity, t2: Entity, _d: DepEdit):
    i1 = t1.head.id
    i2 = t2.head.id
    comp_dist = {"nom+": 0,
                 "acc+": 0,
                 "obl+": 0,
                 "gen+": 0,
                 "other+": 0,
                 "nom-": 0,
                 "acc-": 0,
                 "obl-": 0,
                 "gen-": 0,
                 "other-": 0,
                 }
    competitors = [r for r in _d.mentions if float(r.tokens[-1].id) < float(i2) and float(r.tokens[0].id) > float(i1)]
    for c in competitors:
        case = func2case(c.head.func)
        humanness = "+" if c.annos["etype"] == "person" else "-"
        label = f"{case}{humanness}"
        comp_dist[label] += 1

    return comp_dist


class Pair:
    def __init__(self, referent: Entity, anaphora: Entity, _competitor_dist, _cp_heads):

        self.referent = referent
        self.referent.tokens = [t for t in self.referent.tokens if '-' not in t.id]
        self.anaphora = anaphora
        self.anaphora.tokens = [t for t in self.anaphora.tokens if '-' not in t.id]
        self.word_dist = self.get_word_dist()
        self.competitor_dist = _competitor_dist
        self.cp_heads = _cp_heads
        self.clause_dist = self.get_clause_dist()
        self.case = self.get_case()
        self.humanness = self.get_humanness()
        self.pronominality = self.get_pronominality()

    def get_case(self):
        return func2case(self.referent.head.func)

    def get_humanness(self):
        if self.referent.annos["etype"] == "person":
            return "human"
        else:
            return "non-human"

    def get_pronominality(self):
        if self.anaphora.head.cpos in ["PRP", "PRP$"]:
            return "pronominal"
        else:
            return "non-pronominal"

    def get_word_dist(self):
        return float(self.anaphora.head.id) - float(self.referent.head.id)

    def get_clause_dist(self):
        return len([n for n in self.cp_heads if float(self.referent.head.id) < n < float(self.anaphora.head.id)])

    def write(self):
        cd = self.competitor_dist
        return '\t'.join([' '.join([t.text for t in self.referent.tokens if '-' not in t.id]),
                          ' '.join([t.text for t in self.anaphora.tokens if '-' not in t.id]),
                          self.case, self.humanness, self.pronominality,
                          str(self.word_dist), str(self.clause_dist),
                          str(cd["nom+"]), str(cd["acc+"]), str(cd["obl+"]), str(cd["gen+"]), str(cd["other+"]),
                          str(cd["nom-"]), str(cd["acc-"]), str(cd["obl-"]), str(cd["gen-"]), str(cd["other-"])]) + "\n"


def document_processing(filename):
    document_pairs = []

    class Options:
        def __init__(self, kill=None, quiet=None):
            self.kill = kill
            self.quiet = quiet

    options = Options(kill="supertoks")

    d = DepEdit(options=options)

    conllu = open(filename, encoding="utf-8").read()

    d.run_depedit(conllu, parse_entities=True)

    # Parse over entity clusters, find pairs!
    pairs = []
    for c in d.entities.keys():
        mentions = [m for m in d.mentions if m.cluster == c]
        prev = None
        for m in mentions:
            if (len(m) > 1 and m.tokens[0].text.lower() == "the" and m.tokens[1].cpos == "NN") or (
                    len(m) == 1 and m.head.cpos in ["PRP", "PRP$"]) and prev is not None:
                pairs.append([prev, m])
            prev = m

    # Calculate CP Head positions, for clausal distance
    _cp_heads = []  # beginning of sentence, and at rel, acl:relcl, advcl, ccomp, xcomp, parataxis, conj, advmod, appos, ref.
    for s in d.sentences:
        i = 0
        bos = s.tokens[i]
        while '-' in bos.id: # that is a supertoken
            i += 1
            bos = s.tokens[i]

        _cp_heads.append(float(bos.id))  # beginning of sentences
        _cp_heads += [float(t.id) for t in s.tokens if
                      t.func in ["rel", "acl:relcl", "advcl", "ccomop", "xcomp", "parataxis", "conj", "advmod", "appos",
                                 "ref"]]

    # Organize into class object
    for _r, _a in pairs:
        try:
            competitor_dist = get_competitor_dist(_r, _a, d)
            pair = Pair(_r, _a, competitor_dist, _cp_heads)
            document_pairs.append(pair)
        except:
            continue

    return document_pairs


def corpus_processing():
    filenames = os.listdir("gum/dep")
    corpus_pairs = []
    f = open("anaphoric_pair_distribution.tsv", "w", encoding="utf-8")
    f.write('\t'.join(["referent", "anaphora", "case", "humanness", "pronominality", "word_d", "clause_d", "nomp", "accp", "oblp", "genp", "othp", "nomn", "accn", "obln", "genn", "othn"]) + "\n")
    for filename in tqdm(filenames):
        document_pairs = document_processing("gum/dep/" + filename)
        corpus_pairs += document_pairs

        for pair in document_pairs:
            f.write(pair.write())

    f.close()
    return corpus_pairs


if __name__ == "__main__":
    corpus_pairs = corpus_processing()
