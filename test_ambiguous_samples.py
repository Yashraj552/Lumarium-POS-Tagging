import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformer_tagger import TransformerTagger

ib = TransformerTagger("models/indicbert_pos")
hr = TransformerTagger("models/hingroberta_pos")

test_sents = [
    "Kal maine dog-ne ka video dekha , he was running so fast yaar !".split(),
    "usne bola ki main nahi aaunga par usko aana pada".split(),
    "welcome to mumbai".split(),
    "tum to pagal ho yaar".split(),
    "yeh book uski hai".split(),
    "modi ji ne parliament me speech diya".split(),
]

for s in test_sents:
    print("\nSENTENCE:", " ".join(s))
    ib_p = ib.predict(s)
    hr_p = hr.predict(s)
    for w, p1, p2 in zip(s, ib_p, hr_p):
        mark = "==" if p1["tag"] == p2["tag"] else "!="
        print(f"  {w:14s} | IndicBERT: {p1['tag']:6s} ({p1['conf']:.2f}) | HingRoBERTa: {p2['tag']:6s} ({p2['conf']:.2f}) | {mark}")

