"""
create_dataset.py
─────────────────
Generates 50 candidate factual statements for GPT-2 Small factual-recall
research.  Each entry has:
    prompt    – text fed to the model
    correct   – ground-truth next-word completion (space-prefixed)
    incorrect – up to 3 human-specified foils (also space-prefixed)

Design notes
────────────
Prompt style is chosen based on pilot experiments with GPT-2 Small:

• Geography  – 2-shot "The capital of X is Y. The capital of Z is" format.
  Seeding with 2 known capitals lifts probability for the target capital
  into the 50-70% range while leaving room for the major-city foil.

• History    – Partial-name completions where the partial token strongly
  suggests one historical figure but leaves a small competing option.
  Full-context attributions (e.g. "invented by Isaac Newton") make the
  model too confident, eliminating all foils.

• Attributes – Composition / comparison / sequence facts where a semantically
  plausible alternative (a different element, unit, or planet) naturally
  competes with the correct answer.

NOTE: `incorrect` values here are human-specified priors.  The validation
script (validate_dataset.py) ALSO derives foils empirically from GPT-2's
actual top-k distribution, which may differ from these guesses.
"""
import json

def cap2(a, ac, b, bc, target):
    return (f"The capital of {a} is {ac}. "
            f"The capital of {b} is {bc}. "
            f"The capital of {target} is")


dataset = [

    # ══════════════════════════════════════════════════════════════════════
    # GEOGRAPHY — 20 entries
    # Seeds chosen so the target capital lands in the 55-75% range while
    # a major non-capital city (Marseille, Milan, Barcelona, Frankfurt…)
    # competes at >10%.
    # ══════════════════════════════════════════════════════════════════════

    {"prompt": cap2("Italy",  "Rome",   "Spain", "Madrid", "France"),
     "correct": " Paris",   "incorrect": [" Marseille", " Lyon",    " Bordeaux"]},

    {"prompt": cap2("Spain",  "Madrid", "Germany", "Berlin", "France"),
     "correct": " Paris",   "incorrect": [" Marseille", " Lyon",    " Strasbourg"]},

    {"prompt": cap2("Italy",  "Rome",   "Germany", "Berlin", "France"),
     "correct": " Paris",   "incorrect": [" Marseille", " Nice",    " Bordeaux"]},

    {"prompt": cap2("UK",     "London", "Germany", "Berlin", "France"),
     "correct": " Paris",   "incorrect": [" Marseille", " Lyon",    " Nice"]},

    {"prompt": cap2("UK",     "London", "Italy",   "Rome",   "France"),
     "correct": " Paris",   "incorrect": [" Marseille", " Lyon",    " Bordeaux"]},

    {"prompt": cap2("Germany","Berlin", "France",  "Paris",  "Spain"),
     "correct": " Madrid",  "incorrect": [" Barcelona", " Seville", " Valencia"]},

    {"prompt": cap2("UK",     "London", "Germany", "Berlin", "Spain"),
     "correct": " Madrid",  "incorrect": [" Barcelona", " Seville", " Valencia"]},

    {"prompt": cap2("UK",     "London", "France",  "Paris",  "Spain"),
     "correct": " Madrid",  "incorrect": [" Barcelona", " Seville", " Valencia"]},

    {"prompt": cap2("Russia", "Moscow", "Italy",   "Rome",   "Spain"),
     "correct": " Madrid",  "incorrect": [" Barcelona", " Valencia"," Seville"]},

    {"prompt": cap2("Russia", "Moscow", "Spain",   "Madrid", "Italy"),
     "correct": " Rome",    "incorrect": [" Milan",    " Naples",  " Florence"]},

    {"prompt": cap2("UK",     "London", "Spain",   "Madrid", "Italy"),
     "correct": " Rome",    "incorrect": [" Milan",    " Naples",  " Florence"]},

    {"prompt": cap2("UK",     "London", "France",  "Paris",  "Italy"),
     "correct": " Rome",    "incorrect": [" Milan",    " Florence"," Naples"]},

    {"prompt": cap2("UK",     "London", "Italy",   "Rome",   "Germany"),
     "correct": " Berlin",  "incorrect": [" Munich",   " Hamburg", " Frankfurt"]},

    {"prompt": cap2("China",  "Beijing","Russia",  "Moscow", "Germany"),
     "correct": " Berlin",  "incorrect": [" Munich",   " Frankfurt"," Hamburg"]},

    {"prompt": cap2("Spain",  "Madrid", "France",  "Paris",  "Italy"),
     "correct": " Rome",    "incorrect": [" Milan",    " Florence"," Venice"]},

    # Supplementary — varied targets
    {"prompt": cap2("France", "Paris",  "Spain",   "Madrid", "Germany"),
     "correct": " Berlin",  "incorrect": [" Munich",   " Frankfurt"," Cologne"]},

    {"prompt": cap2("Italy",  "Rome",   "France",  "Paris",  "Russia"),
     "correct": " Moscow",  "incorrect": [" St Petersburg"," Kiev"," Minsk"]},

    {"prompt": cap2("Japan",  "Tokyo",  "China",   "Beijing","Russia"),
     "correct": " Moscow",  "incorrect": [" St Petersburg"," Kiev"," Vladivostok"]},

    {"prompt": cap2("Germany","Berlin", "France",  "Paris",  "Russia"),
     "correct": " Moscow",  "incorrect": [" St Petersburg"," Kiev"," Warsaw"]},

    {"prompt": cap2("France", "Paris",  "Germany", "Berlin", "Japan"),
     "correct": " Tokyo",   "incorrect": [" Osaka",   " Kyoto",   " Yokohama"]},


    # ══════════════════════════════════════════════════════════════════════
    # HISTORY — 15 entries
    # Partial-name and single-token attribution prompts where GPT-2 is
    # moderately confident but retains a meaningful competing token.
    # ══════════════════════════════════════════════════════════════════════

    # Partial-name completions (work well because the model assigns
    # probability to the next NAME TOKEN rather than a full name)
    {"prompt": "Sir Isaac",
     "correct": " Newton",  "incorrect": [" As",       " Alexander"," Asimov"]},

    {"prompt": "The Mona Lisa was painted by Leonardo",
     "correct": " da",      "incorrect": [" Di",       " De",       " del"]},

    {"prompt": "Leibniz and Newton both developed calculus. Newton is also known as Sir Isaac",
     "correct": " Newton",  "incorrect": [" As",       " Alexander"," And"]},

    {"prompt": "The Principia Mathematica of Sir Isaac",
     "correct": " Newton",  "incorrect": [" As",       " Alexander"," Asimov"]},

    {"prompt": "Vasco da",
     "correct": " Gama",    "incorrect": [" Gamma",    " Vila",     " da"]},

    # Sentence-completion attributions
    {"prompt": "The Mona Lisa and The Last Supper were painted by Leonardo da",
     "correct": " Vinci",   "incorrect": [" Caprio",   " Savio",    " Fabbri"]},

    {"prompt": "The Communist Manifesto was written by Karl Marx and Friedrich",
     "correct": " Engels",  "incorrect": [" Nietzsche"," Hegel",    " Hess"]},

    {"prompt": "Romeo and Juliet was written by William",
     "correct": " Shakespeare","incorrect":[" Golding", " Wordsworth"," Blake"]},

    {"prompt": "Hamlet is a tragedy by William",
     "correct": " Shakespeare","incorrect":[" Golding", " Blake",    " Thackeray"]},

    {"prompt": "The Communist Manifesto was co-authored by Marx and",
     "correct": " Engels",  "incorrect": [" Lenin",    " Trotsky",  " Stalin"]},

    # Date-completion chains (anchored to eliminate ambiguity)
    {"prompt": "World War I ended in 1918. World War II ended in",
     "correct": " 1945",    "incorrect": [" 1944",     " 1946",     " 1943"]},

    {"prompt": "The Declaration of Independence was signed in 1776. "
               "The French Revolution began in",
     "correct": " 1789",    "incorrect": [" 1776",     " 1799",     " 1815"]},

    {"prompt": "The Berlin Wall was built in 1961. The Berlin Wall fell in",
     "correct": " 1989",    "incorrect": [" 1987",     " 1991",     " 1985"]},

    {"prompt": "The Berlin Wall fell in 1989. The Soviet Union collapsed in",
     "correct": " 1991",    "incorrect": [" 1989",     " 1993",     " 1988"]},

    {"prompt": "World War Two ran from 1939 to",
     "correct": " 1945",    "incorrect": [" 1944",     " 1946",     " 1943"]},


    # ══════════════════════════════════════════════════════════════════════
    # ATTRIBUTES — 15 entries
    # Physical composition, comparison, and sequence facts where the correct
    # answer and a close alternative both receive significant probability.
    # ══════════════════════════════════════════════════════════════════════

    # Chemistry composition (analogous element competes)
    {"prompt": "Water is composed of hydrogen and",
     "correct": " oxygen",  "incorrect": [" helium",   " nitrogen", " carbon"]},

    {"prompt": "The freezing point of water is 0 degrees",
     "correct": " Celsius", "incorrect": [" Fahrenheit"," Kelvin",  " Centigrade"]},

    {"prompt": "In the metric system, temperature is measured in degrees",
     "correct": " Celsius", "incorrect": [" Fahrenheit"," Kelvin",  " Centigrade"]},

    # Atomic structure (competing subatomic particle)
    {"prompt": "The nucleus of an atom contains protons and",
     "correct": " neutrons","incorrect": [" electrons", " quarks",  " photons"]},

    # Speed comparison (the competing option is the other well-known speed)
    {"prompt": "The speed of sound is slower than the speed of",
     "correct": " light",   "incorrect": [" sound",    " electricity"," radio"]},

    # Planet size chain (next planet in size order is the foil)
    {"prompt": "Saturn is larger than Uranus, which is larger than",
     "correct": " Neptune", "incorrect": [" Jupiter",  " Earth",    " Mars"]},

    {"prompt": "Jupiter is the largest planet. Saturn is the second largest. Uranus is the",
     "correct": " third",   "incorrect": [" fourth",   " second",   " fifth"]},

    # More attribute facts
    {"prompt": "The boiling point of water is 100 degrees",
     "correct": " Celsius", "incorrect": [" Fahrenheit"," Kelvin",  " Centigrade"]},

    {"prompt": "Electrolysis of water produces hydrogen and",
     "correct": " oxygen",  "incorrect": [" helium",   " nitrogen", " carbon"]},

    {"prompt": "The Sun and Jupiter are both composed mostly of hydrogen and",
     "correct": " helium",  "incorrect": [" oxygen",   " nitrogen", " carbon"]},

    {"prompt": "Carbon dioxide is a compound of carbon and",
     "correct": " oxygen",  "incorrect": [" nitrogen", " hydrogen", " helium"]},

    {"prompt": "Protons have a positive charge. Electrons have a",
     "correct": " negative","incorrect": [" positive", " neutral",  " zero"]},

    {"prompt": "Every magnet has a north pole and a",
     "correct": " south",   "incorrect": [" negative", " positive", " east"]},

    {"prompt": "Matter exists in three main states: solid, liquid, and",
     "correct": " gas",     "incorrect": [" plasma",   " solid",    " liquid"]},

    {"prompt": "The currency of the United Kingdom is the",
     "correct": " pound",   "incorrect": [" euro",     " dollar",   " franc"]},
]

assert len(dataset) == 50, f"Expected 50, got {len(dataset)}"

for i, ex in enumerate(dataset):
    assert ex["correct"].startswith(" "), f"Entry {i}: correct must start with space"
    for inc in ex["incorrect"]:
        assert inc.startswith(" "), f"Entry {i}: incorrect '{inc}' must start with space"

with open("factual_recall_raw.json", "w") as f:
    json.dump(dataset, f, indent=2)

counts = {}
for ex in dataset:
    cat = "geography" if any(kw in ex["prompt"] for kw in ["capital", "The capital"]) else \
          "history"   if any(ex["prompt"].startswith(p) for p in
                             ["Sir","The Mona","Leibniz","Vasco","The Communist","Romeo",
                              "Hamlet","The Principia","World War","The Declaration",
                              "The Berlin","World War Two"]) else "attributes"
    counts[cat] = counts.get(cat, 0) + 1

print(f"Generated {len(dataset)} candidate prompts → factual_recall_raw.json")
for cat, n in sorted(counts.items()):
    print(f"  {cat}: {n}")
