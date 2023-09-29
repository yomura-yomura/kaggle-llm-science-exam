import pandas as pd
import wikipediaapi

matched_cohere_df = pd.read_parquet("additional_cohere.parquet")
matched_cohere_df["section"] = ""
matched_cohere_df[["title", "section", "text"]]

wiki_wiki = wikipediaapi.Wikipedia(
    "MyProjectName (merlin@example.com)", "en", extract_format=wikipediaapi.ExtractFormat.WIKI
)


def print_sections(sections, level=0):
    for s in sections:
        print("%s: %s - %s" % ("*" * (level + 1), s.title, s.text[0:40]))
        yield s.text
        if len(s.sections) > 0:
            for text in print_sections(s.sections, level + 1):
                yield text


for title, df in matched_cohere_df.groupby("title", sort=False):
    print(title)
    page = wiki_wiki.page(title)

    paragraphs = page.summary.splitlines()
    texts = list(print_sections(page.sections))
    break
