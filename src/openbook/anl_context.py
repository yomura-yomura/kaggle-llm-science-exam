import pandas as pd
import plotly.express as px

df = pd.read_csv("valid_with_context.csv")

count_df = df["category_top1"].value_counts().rename("count").reset_index()
fig = px.bar(count_df, x="index", y="count")
fig.show()


import wikipediaapi


def print_categorymembers(categorymembers, level=0, max_level=1):
    for c in categorymembers.values():
        # if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
        # for title in print_categorymembers(c.categorymembers, level=level + 1, max_level=max_level):
        #     yield title
        # else:
        #     print("%s: %s (ns: %d)" % ("*" * (level + 1), c.title, c.ns))
        #     yield c.title
        if c.ns == wikipediaapi.Namespace.MAIN:
            yield c.title
        else:
            try:
                for title in print_categorymembers(c.categorymembers, level=level + 1, max_level=max_level):
                    yield title
            except KeyError:
                pass


wiki_wiki = wikipediaapi.Wikipedia("MyProjectName (merlin@example.com)", "en")
# cat = wiki_wiki.page("Category:Physical cosmology")
# cats = list(print_categorymembers(cat.categorymembers))

import tqdm

cats = set()
for stem_cat in tqdm.tqdm([cat for cats in STEM.values() for cat in cats]):
    # cat = wiki_wiki.page(f"{stem_cat}").categories.get(f"{stem_cat}", None)
    # if cat is None or cat.ns != wikipediaapi.Namespace.CATEGORY:
    #     print(f"skip {stem_cat}")
    #     continue
    cat = wiki_wiki.page(f"{stem_cat}")
    cats |= set(print_categorymembers(cat.categorymembers))


stem_categories = [
    "Aerospace engineering",
    "Agriculture",
    "Archaeology",
    "Architecture",
    "Artificial intelligence",
    "Astronomy",
    "Biology",
    "Botany",
    "Calculus",
    "Cell biology",
    "Chemistry",
    "Civil engineering",
    "Clinical research",
    "Computer hardware",
    "Computer science",
    "Developmental and reproductive biology",
    "Ecology",
    "Economics",
    "Electrical and electronics engineering",
    "Engineering",
    "Entomology",
    "Environmental science",
    "Evolutionary biology",
    "Genetics",
    "Geography ",
    "Geology",
    "Ichthyology",
    "Machine vision",
    "Mathematics",
    "Mechanical engineering",
    "Medicine",
    "Meteorology",
    "Mycology",
    "Nanotechnology",
    "Ornithology",
    "Physics",
    "Probability and statistics",
    "Psychiatry",
    "Quantum computing",
    "Robotics",
    "Scientific naming",
    "Structural engineering",
    "Virology",
]


STEM = {
    "S": ["Category:Applied_sciences", "Category:Biotechnology", "Category:Biology", "Category:Natural_history"],
    "T": [
        "Category:Technology_strategy",
        "Category:Technical_specifications",
        "Category:Technology_assessment",
        "Category:Technology_hazards",
        "Category:Technology_systems",
        "Category:Hypothetical_technology",
        "Category:Mobile_technology",
        "Category:Obsolete_technologies",
        "Category:Philosophy_of_technology",
        "Category:Real-time_technology",
        "Category:Software",
        "Category:Technology_development",
        "Category:Computing",
        "Category:Artificial_objects",
        "Category:Technological_change",
        "Category:Technical_communication",
        "Category:Technological_comparisons",
    ],
    "E": [
        "Category:Engineering_disciplines",
        "Category:Engineering_concepts",
        "Category:Industrial_equipment",
        "Category:Manufacturing",
    ],
    "M": ["Category:Fields_of_mathematics", "Category:Physical_sciences"],
}
