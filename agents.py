from autogen_agentchat import AssitantAgent
from autogen_ext.models import OpenAIChatCompletionClient
import os
import arxiv
gemini = OpenAIChatCompletionClient(
    model="gemini-1.5",
    api_key=os.getenv("GEMINI_API_KEY"))


def arxiv_search(query: str, max_results: int = 5):
    client = arxiv.Client()
    search = arxiv.Search(
        query = query,
        max_results = max_results,
        sort_by= arxiv.SortCriterion.Relevance,
    )

    papers: List[Dict] = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "published": result.published.isoformat(),
            "summary": result.summary,
            "url": result.pdf_url
        })
    return papers

researcher_agent = AssitantAgent(
    name = "ResearcherAgent",
    description = "An agent that searches for research papers on a given topic.",
    model_client = gemini,
    tools = [arxiv_search]
    system_message = (
        "You are an expert researcher. When you receive a topic, "
        "search for the latest research papers on that topic and return a JSON list of "
        "papers with their titles, authors, publication dates, and abstracts."
    )
)

summarizer_agent = AssitantAgent(
    name = "SummarizerAgent",
    description = "An agent that summarizes the content of a given document.",
    model_client = gemini
    system_message = (
        "you are an expert researcher and summarizer. when you recieve a JSON list of "
        "papers, write a literatur review of the papers in the list in Markdown format.\n"
        "1. start with a 2-3 lines of brief description of the topic"
        "2. Then include on bullet per page with : title as Markdown link,"
        "authors and the specific problem tackled by the paper. and it key contributions."
        "3. close the line with a single line takeaway"
        "And at last give 1 must read paper about the papers present in the list."
    )
    )