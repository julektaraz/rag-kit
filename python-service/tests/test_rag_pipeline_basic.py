from pipeline.rag_pipeline import RAGPipeline


def test_query_without_documents_returns_helpful_message():
    pipeline = RAGPipeline(
        auto_save=False,
        persistence_path=None,
    )

    result = pipeline.query("What is this project about?")

    assert "No documents have been ingested" in result["answer"]
    assert result["contexts"] == []
    assert result["sources"] == []


